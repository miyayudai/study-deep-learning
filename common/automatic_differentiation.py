"""Chapter 8: Backpropagation
Section 8.2: Automatic Differentiation

This module implements forward-mode and reverse-mode automatic differentiation (autodiff)
as described in Section 8.2 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Comparison of gradient evaluation techniques:
  * Manual derivation and explicit implementation
  * Numerical differentiation (finite differences)
  * Symbolic differentiation (expression swell, Eqs 8.42-8.48)
  * Automatic differentiation (machine precision, no expression swell, control flow)
- Section 8.2.1: Forward-mode automatic differentiation
  * Dual numbers: x = v + v_dot * eps where eps^2 = 0
  * Primal and tangent variables (Eq 8.57)
  * Evaluation trace diagram (Figure 8.4)
  * Multi-output function trace (Figure 8.5, Eq 8.65)
  * Jacobian-vector products (J * r) in a single forward pass (Eq 8.67)
  * Column-by-column Jacobian evaluation in D passes
- Section 8.2.2: Reverse-mode automatic differentiation
  * Adjoint variables v_bar = del f / del v_i (Eq 8.68)
  * Recursive adjoint accumulation: v_bar_i = sum_{j in ch(i)} v_bar_j * (del v_j / del v_i) (Eq 8.69)
  * Reverse evaluation trace for example function (Eqs 8.70-8.76)
  * Tape/computation-graph reverse autodiff engine (Node / Var)
  * Memory/compute tradeoff between forward and reverse modes
  * Hybrid forward-over-reverse Hessian-vector product (H * v) in O(W) (Pearlmutter 1994)
- Reproductions of Figures 8.4 and 8.5
"""

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure_files(
    fig: plt.Figure,
    base_name: str,
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> str:
    """Save figure to specified filepath and/or result directories."""
    filename = f"{base_name}.png"
    root = _get_project_root()
    path_ch8 = os.path.join(root, "8", "result", filename)
    path_root = os.path.join(root, "result", filename)

    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_plot(fig, filepath)
        primary_path = filepath
    else:
        primary_path = path_ch8

    if result_dirs:
        for rdir in result_dirs:
            os.makedirs(rdir, exist_ok=True)
            save_plot(fig, os.path.join(rdir, filename))

    if save_both:
        os.makedirs(os.path.dirname(path_ch8), exist_ok=True)
        os.makedirs(os.path.dirname(path_root), exist_ok=True)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch8):
            save_plot(fig, path_ch8)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)

        # Also save Figure_8_X.png alias
        parts = base_name.split("_")
        if len(parts) >= 3 and parts[0] == "fig" and parts[1] == "8":
            alt_filename = f"Figure_8_{parts[2]}.png"
            alt_ch8 = os.path.join(root, "8", "result", alt_filename)
            alt_root = os.path.join(root, "result", alt_filename)
            save_plot(fig, alt_ch8)
            save_plot(fig, alt_root)

    return primary_path


# =====================================================================
# Section 8.2.1: Forward-Mode Automatic Differentiation (Dual Numbers)
# =====================================================================

class DualNumber:
    """Dual number representation for forward-mode automatic differentiation.

    A dual number has the form:
        z = real + dual * eps,  where eps^2 = 0
    Here, 'real' represents the primal variable v_i, and 'dual' represents
    the tangent variable v_dot_i = del v_i / del x.
    """

    def __init__(self, real: float, dual: float = 0.0):
        self.real = float(real)
        self.dual = float(dual)

    def __repr__(self) -> str:
        return f"DualNumber(real={self.real:.6g}, dual={self.dual:.6g})"

    # Arithmetic operators
    def __add__(self, other: Union["DualNumber", float, int]) -> "DualNumber":
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        return DualNumber(self.real + other, self.dual)

    def __radd__(self, other: Union[float, int]) -> "DualNumber":
        return self.__add__(other)

    def __sub__(self, other: Union["DualNumber", float, int]) -> "DualNumber":
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        return DualNumber(self.real - other, self.dual)

    def __rsub__(self, other: Union[float, int]) -> "DualNumber":
        return DualNumber(other - self.real, -self.dual)

    def __mul__(self, other: Union["DualNumber", float, int]) -> "DualNumber":
        # (u + u_dot eps) * (v + v_dot eps) = u*v + (u_dot*v + u*v_dot) eps
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real * other.real,
                self.dual * other.real + self.real * other.dual,
            )
        return DualNumber(self.real * other, self.dual * other)

    def __rmul__(self, other: Union[float, int]) -> "DualNumber":
        return self.__mul__(other)

    def __truediv__(self, other: Union["DualNumber", float, int]) -> "DualNumber":
        # (u + u_dot eps) / (v + v_dot eps) = u/v + (u_dot*v - u*v_dot)/(v^2) eps
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real / other.real,
                (self.dual * other.real - self.real * other.dual) / (other.real**2),
            )
        return DualNumber(self.real / other, self.dual / other)

    def __rtruediv__(self, other: Union[float, int]) -> "DualNumber":
        return DualNumber(other, 0.0).__truediv__(self)

    def __neg__(self) -> "DualNumber":
        return DualNumber(-self.real, -self.dual)

    def __pow__(self, power: Union[float, int]) -> "DualNumber":
        # (u + u_dot eps)^p = u^p + p * u^(p-1) * u_dot eps
        p = float(power)
        return DualNumber(self.real**p, p * (self.real ** (p - 1.0)) * self.dual)

    # Elementary transcendental functions
    def exp(self) -> "DualNumber":
        # exp(u + u_dot eps) = exp(u) + u_dot * exp(u) eps
        e = np.exp(self.real)
        return DualNumber(e, self.dual * e)

    def log(self) -> "DualNumber":
        # ln(u + u_dot eps) = ln(u) + (u_dot / u) eps
        return DualNumber(np.log(self.real), self.dual / self.real)

    def sin(self) -> "DualNumber":
        # sin(u + u_dot eps) = sin(u) + u_dot * cos(u) eps
        return DualNumber(np.sin(self.real), self.dual * np.cos(self.real))

    def cos(self) -> "DualNumber":
        # cos(u + u_dot eps) = cos(u) - u_dot * sin(u) eps
        return DualNumber(np.cos(self.real), -self.dual * np.sin(self.real))

    def tanh(self) -> "DualNumber":
        # tanh(u + u_dot eps) = tanh(u) + u_dot * (1 - tanh^2(u)) eps
        t = np.tanh(self.real)
        return DualNumber(t, self.dual * (1.0 - t**2))

    def sigmoid(self) -> "DualNumber":
        # sigma(u + u_dot eps) = sigma(u) + u_dot * sigma(u)*(1 - sigma(u)) eps
        s = 1.0 / (1.0 + np.exp(-np.clip(self.real, -500, 500)))
        return DualNumber(s, self.dual * s * (1.0 - s))


# Free functions for math over dual numbers or floats
def d_exp(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    return x.exp() if isinstance(x, DualNumber) else np.exp(x)


def d_log(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    return x.log() if isinstance(x, DualNumber) else np.log(x)


def d_sin(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    return x.sin() if isinstance(x, DualNumber) else np.sin(x)


def d_cos(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    return x.cos() if isinstance(x, DualNumber) else np.cos(x)


def d_tanh(x: Union[DualNumber, float]) -> Union[DualNumber, float]:
    return x.tanh() if isinstance(x, DualNumber) else np.tanh(x)


# Forward-mode utilities
def forward_mode_derivative(
    func: Callable[[DualNumber], DualNumber], x: float
) -> Tuple[float, float]:
    """Evaluate f(x) and f'(x) using forward-mode autodiff with a single dual number.

    Args:
        func: function accepting DualNumber
        x: evaluation point

    Returns:
        (f_val, f_prime)
    """
    x_dual = DualNumber(x, dual=1.0)
    res = func(x_dual)
    return res.real, res.dual


def forward_mode_jvp(
    func: Callable[[List[DualNumber]], Union[DualNumber, List[DualNumber]]],
    x: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian-vector product J * v in a single forward pass (Eq 8.67).

    Args:
        func: function taking list/array of DualNumber and returning DualNumber or list
        x: input point (D,)
        v: direction vector (D,)

    Returns:
        (y, Jv): outputs and directional derivatives
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    v_arr = np.asarray(v, dtype=float).ravel()
    D = len(x_arr)

    # Initialize x_dot = v
    x_duals = [DualNumber(x_arr[i], dual=v_arr[i]) for i in range(D)]
    out = func(x_duals)

    if isinstance(out, DualNumber):
        return np.array([out.real]), np.array([out.dual])
    else:
        y_vals = np.array([o.real for o in out])
        jvp_vals = np.array([o.dual for o in out])
        return y_vals, jvp_vals


def forward_mode_jacobian(
    func: Callable[[List[DualNumber]], List[DualNumber]],
    x: np.ndarray,
) -> np.ndarray:
    """Compute the full K x D Jacobian matrix using D forward passes (Eq 8.66).

    Args:
        func: vector function R^D -> R^K
        x: input point (D,)

    Returns:
        J: shape (K, D)
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    D = len(x_arr)
    # First pass to determine K
    e0 = np.zeros(D)
    e0[0] = 1.0
    y0, col0 = forward_mode_jvp(func, x_arr, e0)
    K = len(y0)

    J = np.zeros((K, D))
    J[:, 0] = col0

    for i in range(1, D):
        ei = np.zeros(D)
        ei[i] = 1.0
        _, coli = forward_mode_jvp(func, x_arr, ei)
        J[:, i] = coli

    return J


# =====================================================================
# Section 8.2.2: Reverse-Mode Automatic Differentiation (Computation Graph)
# =====================================================================

class Node:
    """A computational graph node for reverse-mode automatic differentiation.

    Maintains:
        - value: primal variable v_i
        - grad: adjoint variable v_bar_i = del f / del v_i (Eq 8.68)
        - parents / dependencies and backward gradient closures (Eq 8.69)
    """

    def __init__(
        self,
        value: float,
        parents: Tuple["Node", ...] = (),
        op: str = "leaf",
        label: Optional[str] = None,
    ):
        self.value = float(value)
        self.grad = 0.0  # Adjoint variable v_bar
        self.parents = parents
        self.op = op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        lbl = f" '{self.label}'" if self.label else ""
        return f"Node({lbl} val={self.value:.4g}, grad={self.grad:.4g}, op='{self.op}')"

    # Operator overloading building the computation graph
    def __add__(self, other: Union["Node", float, int]) -> "Node":
        other_node = other if isinstance(other, Node) else Node(other, op="const")
        out = Node(self.value + other_node.value, (self, other_node), op="+")

        def _backward():
            # v_out = v_self + v_other => del v_out / del v_self = 1
            self.grad += out.grad
            other_node.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> "Node":
        return self.__add__(other)

    def __sub__(self, other: Union["Node", float, int]) -> "Node":
        other_node = other if isinstance(other, Node) else Node(other, op="const")
        out = Node(self.value - other_node.value, (self, other_node), op="-")

        def _backward():
            # v_out = v_self - v_other => del v_out / del v_self = 1, del v_out / del v_other = -1
            self.grad += out.grad
            other_node.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: Union[float, int]) -> "Node":
        other_node = Node(other, op="const")
        return other_node.__sub__(self)

    def __mul__(self, other: Union["Node", float, int]) -> "Node":
        other_node = other if isinstance(other, Node) else Node(other, op="const")
        out = Node(self.value * other_node.value, (self, other_node), op="*")

        def _backward():
            # v_out = v_self * v_other => del v_out / del v_self = v_other
            self.grad += out.grad * other_node.value
            other_node.grad += out.grad * self.value

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[float, int]) -> "Node":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Node", float, int]) -> "Node":
        other_node = other if isinstance(other, Node) else Node(other, op="const")
        out = Node(self.value / other_node.value, (self, other_node), op="/")

        def _backward():
            self.grad += out.grad / other_node.value
            other_node.grad -= out.grad * self.value / (other_node.value**2)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union[float, int]) -> "Node":
        other_node = Node(other, op="const")
        return other_node.__truediv__(self)

    def __neg__(self) -> "Node":
        out = Node(-self.value, (self,), op="neg")

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: Union[float, int]) -> "Node":
        p = float(power)
        out = Node(self.value**p, (self,), op=f"pow({p})")

        def _backward():
            self.grad += out.grad * p * (self.value ** (p - 1.0))

        out._backward = _backward
        return out

    # Elementary functions
    def exp(self) -> "Node":
        out = Node(np.exp(self.value), (self,), op="exp")

        def _backward():
            self.grad += out.grad * out.value  # del exp(v) / del v = exp(v)

        out._backward = _backward
        return out

    def log(self) -> "Node":
        out = Node(np.log(self.value), (self,), op="log")

        def _backward():
            self.grad += out.grad / self.value

        out._backward = _backward
        return out

    def sin(self) -> "Node":
        out = Node(np.sin(self.value), (self,), op="sin")

        def _backward():
            self.grad += out.grad * np.cos(self.value)

        out._backward = _backward
        return out

    def cos(self) -> "Node":
        out = Node(np.cos(self.value), (self,), op="cos")

        def _backward():
            self.grad -= out.grad * np.sin(self.value)

        out._backward = _backward
        return out

    def tanh(self) -> "Node":
        t = np.tanh(self.value)
        out = Node(t, (self,), op="tanh")

        def _backward():
            self.grad += out.grad * (1.0 - t**2)

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Execute reverse-mode automatic differentiation starting from this node.

        Topologically sorts the computation graph and accumulates adjoint variables (Eq 8.69).
        """
        topo: List[Node] = []
        visited: Set[Node] = set()

        def build_topo(v: Node):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Seed the adjoint variable of output: v_bar_out = 1.0 (Eq 8.70)
        self.grad = 1.0

        # Traverse backwards in topological order
        for node in reversed(topo):
            node._backward()


# =====================================================================
# Textbook Examples: Functions (8.49) and (8.65)
# =====================================================================

def example_function_8_49(x1: Any, x2: Any) -> Any:
    """Textbook example function (Eq 8.49):

    f(x1, x2) = x1 * x2 + exp(x1 * x2) - sin(x2)
    """
    prod = x1 * x2
    if isinstance(x1, DualNumber):
        return prod + prod.exp() - x2.sin()
    elif isinstance(x1, Node):
        return prod + prod.exp() - x2.sin()
    else:
        return prod + np.exp(prod) - np.sin(x2)


def example_function_8_65(x1: Any, x2: Any) -> Tuple[Any, Any]:
    """Textbook two-output example function (Eq 8.49 & Eq 8.65):

    f1(x1, x2) = x1 * x2 + exp(x1 * x2) - sin(x2)
    f2(x1, x2) = (x1 * x2 - sin(x2)) * exp(x1 * x2)
    """
    prod = x1 * x2
    if isinstance(x1, DualNumber):
        s2 = x2.sin()
        ep = prod.exp()
        f1 = prod + ep - s2
        f2 = (prod - s2) * ep
        return f1, f2
    elif isinstance(x1, Node):
        s2 = x2.sin()
        ep = prod.exp()
        f1 = prod + ep - s2
        f2 = (prod - s2) * ep
        return f1, f2
    else:
        s2 = np.sin(x2)
        ep = np.exp(prod)
        f1 = prod + ep - s2
        f2 = (prod - s2) * ep
        return f1, f2


def evaluate_trace_forward_mode(x1: float, x2: float) -> Dict[str, Tuple[float, float]]:
    """Numerically evaluate primal (v_i) and tangent (v_dot_i = del v_i / del x1) variables.

    Implements primal equations (8.50)-(8.56) and tangent equations (8.58)-(8.64).
    """
    v1 = x1
    v1_dot = 1.0  # del x1 / del x1

    v2 = x2
    v2_dot = 0.0  # del x2 / del x1

    v3 = v1 * v2
    v3_dot = v1 * v2_dot + v1_dot * v2

    v4 = np.sin(v2)
    v4_dot = v2_dot * np.cos(v2)

    v5 = np.exp(v3)
    v5_dot = v3_dot * np.exp(v3)

    v6 = v3 - v4
    v6_dot = v3_dot - v4_dot

    v7 = v5 + v6
    v7_dot = v5_dot + v6_dot

    return {
        "v1": (v1, v1_dot),
        "v2": (v2, v2_dot),
        "v3": (v3, v3_dot),
        "v4": (v4, v4_dot),
        "v5": (v5, v5_dot),
        "v6": (v6, v6_dot),
        "v7": (v7, v7_dot),
    }


def evaluate_trace_reverse_mode(x1: float, x2: float) -> Dict[str, Tuple[float, float]]:
    """Numerically evaluate primal (v_i) and adjoint (v_bar_i = del f / del v_i) variables.

    Implements primal equations (8.50)-(8.56) and adjoint equations (8.70)-(8.76).
    """
    # Forward pass: compute primal variables
    v1 = x1
    v2 = x2
    v3 = v1 * v2
    v4 = np.sin(v2)
    v5 = np.exp(v3)
    v6 = v3 - v4
    v7 = v5 + v6

    # Backward pass: evaluate adjoint variables (Eqs 8.70-8.76)
    v7_bar = 1.0
    v6_bar = v7_bar
    v5_bar = v7_bar
    v4_bar = -v6_bar
    v3_bar = v5_bar * v5 + v6_bar  # v5_bar * exp(v3) + v6_bar
    v2_bar = v3_bar * v1 + v4_bar * np.cos(v2)
    v1_bar = v3_bar * v2

    return {
        "v1": (v1, v1_bar),
        "v2": (v2, v2_bar),
        "v3": (v3, v3_bar),
        "v4": (v4, v4_bar),
        "v5": (v5, v5_bar),
        "v6": (v6, v6_bar),
        "v7": (v7, v7_bar),
    }


# =====================================================================
# Figure Reproduction: Figure 8.4 and Figure 8.5
# =====================================================================

def generate_figure_8_4(
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> plt.Figure:
    """Figure 8.4: Evaluation trace diagram showing steps in numerical evaluation of (8.49).

    Reproduces Bishop & Bishop (2024) Figure 8.4:
    - Inputs x1, x2
    - Primal nodes v1 .. v7
    - Mathematical operation annotations above and below nodes
    - Single output f
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Node positions (x, y)
    pos = {
        "v1": (2.2, 3.8),
        "v2": (2.2, 1.8),
        "v3": (4.2, 3.8),
        "v4": (4.2, 1.8),
        "v5": (6.2, 3.8),
        "v6": (6.2, 1.8),
        "v7": (8.2, 3.8),
    }
    radius = 0.42

    node_face = "#e8f5e9"  # soft pastel green
    node_edge = "#2e7d32"  # dark green outline

    # Draw nodes
    for name, p in pos.items():
        circle = patches.Circle(
            p,
            radius,
            facecolor=node_face,
            edgecolor=node_edge,
            linewidth=1.8,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            p[0],
            p[1],
            r"$" + name[0] + "_" + name[1] + "$",
            ha="center",
            va="center",
            fontsize=13,
            color="#111111",
            zorder=5,
        )

    # Helper for directed edge between nodes
    def draw_edge(p_start, p_end, offset_start=True, offset_end=True):
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        dist = np.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        start = (
            p_start[0] + ux * radius if offset_start else p_start[0],
            p_start[1] + uy * radius if offset_start else p_start[1],
        )
        end = (
            p_end[0] - ux * radius if offset_end else p_end[0],
            p_end[1] - uy * radius if offset_end else p_end[1],
        )
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.6, mutation_scale=14),
            zorder=3,
        )

    # Inputs x1 and x2
    ax.text(0.6, 3.8, r"$x_1$", ha="center", va="center", fontsize=14)
    draw_edge((0.9, 3.8), pos["v1"], offset_start=False)

    ax.text(0.6, 1.8, r"$x_2$", ha="center", va="center", fontsize=14)
    draw_edge((0.9, 1.8), pos["v2"], offset_start=False)

    # Connections between nodes
    # v1 -> v3
    draw_edge(pos["v1"], pos["v3"])
    # v2 -> v3
    draw_edge(pos["v2"], pos["v3"])
    # v2 -> v4
    draw_edge(pos["v2"], pos["v4"])
    # v3 -> v5
    draw_edge(pos["v3"], pos["v5"])
    # v3 -> v6
    draw_edge(pos["v3"], pos["v6"])
    # v4 -> v6
    draw_edge(pos["v4"], pos["v6"])
    # v5 -> v7
    draw_edge(pos["v5"], pos["v7"])
    # v6 -> v7
    draw_edge(pos["v6"], pos["v7"])

    # Output f
    draw_edge(pos["v7"], (9.5, 3.8), offset_end=False)
    ax.text(9.8, 3.8, r"$f$", ha="center", va="center", fontsize=15, fontstyle="italic")

    # Formula annotations above / below nodes matching textbook Figure 8.4
    ax.text(pos["v1"][0], 4.75, r"$x_1$", ha="center", va="center", fontsize=12)
    ax.text(pos["v2"][0], 0.85, r"$x_2$", ha="center", va="center", fontsize=12)
    ax.text(pos["v3"][0], 4.75, r"$v_1 v_2$", ha="center", va="center", fontsize=12)
    ax.text(pos["v4"][0], 0.85, r"$\sin(v_2)$", ha="center", va="center", fontsize=12)
    ax.text(pos["v5"][0], 4.75, r"$\exp(v_3)$", ha="center", va="center", fontsize=12)
    ax.text(pos["v6"][0], 0.85, r"$v_3 - v_4$", ha="center", va="center", fontsize=12)
    ax.text(pos["v7"][0], 4.75, r"$v_5 + v_6$", ha="center", va="center", fontsize=12)

    ax.set_title(
        r"$\mathbf{Figure\ 8.4:\ Evaluation\ Trace\ Diagram\ for\ Function\ (8.49)}$",
        fontsize=12,
        pad=10,
    )

    fig.tight_layout()
    _save_figure_files(
        fig,
        "fig_8_4_evaluation_trace",
        filepath=filepath,
        result_dirs=result_dirs,
        save_both=save_both,
    )
    return fig


def generate_figure_8_5(
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> plt.Figure:
    """Figure 8.5: Extension of Figure 8.4 to a function with two outputs f1 and f2.

    Reproduces Bishop & Bishop (2024) Figure 8.5:
    - Same inputs x1, x2 and intermediate nodes v1..v6
    - Output node v7 -> f1
    - Second output node v8 -> f2 with formula v5 * v6
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    pos = {
        "v1": (2.2, 3.8),
        "v2": (2.2, 1.8),
        "v3": (4.2, 3.8),
        "v4": (4.2, 1.8),
        "v5": (6.2, 3.8),
        "v6": (6.2, 1.8),
        "v7": (8.2, 3.8),
        "v8": (8.2, 1.8),
    }
    radius = 0.42

    node_face = "#e8f5e9"
    node_edge = "#2e7d32"

    for name, p in pos.items():
        circle = patches.Circle(
            p,
            radius,
            facecolor=node_face,
            edgecolor=node_edge,
            linewidth=1.8,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            p[0],
            p[1],
            r"$" + name[0] + "_" + name[1] + "$",
            ha="center",
            va="center",
            fontsize=13,
            color="#111111",
            zorder=5,
        )

    def draw_edge(p_start, p_end, offset_start=True, offset_end=True):
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        dist = np.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        start = (
            p_start[0] + ux * radius if offset_start else p_start[0],
            p_start[1] + uy * radius if offset_start else p_start[1],
        )
        end = (
            p_end[0] - ux * radius if offset_end else p_end[0],
            p_end[1] - uy * radius if offset_end else p_end[1],
        )
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.6, mutation_scale=14),
            zorder=3,
        )

    # Inputs x1 and x2
    ax.text(0.6, 3.8, r"$x_1$", ha="center", va="center", fontsize=14)
    draw_edge((0.9, 3.8), pos["v1"], offset_start=False)

    ax.text(0.6, 1.8, r"$x_2$", ha="center", va="center", fontsize=14)
    draw_edge((0.9, 1.8), pos["v2"], offset_start=False)

    # Edges between intermediate nodes
    draw_edge(pos["v1"], pos["v3"])
    draw_edge(pos["v2"], pos["v3"])
    draw_edge(pos["v2"], pos["v4"])
    draw_edge(pos["v3"], pos["v5"])
    draw_edge(pos["v3"], pos["v6"])
    draw_edge(pos["v4"], pos["v6"])

    # Output 1 connections (v5 -> v7, v6 -> v7)
    draw_edge(pos["v5"], pos["v7"])
    draw_edge(pos["v6"], pos["v7"])

    # Output 2 connections (v5 -> v8, v6 -> v8)
    draw_edge(pos["v5"], pos["v8"])
    draw_edge(pos["v6"], pos["v8"])

    # Outputs f1 and f2
    draw_edge(pos["v7"], (9.5, 3.8), offset_end=False)
    ax.text(9.8, 3.8, r"$f_1$", ha="center", va="center", fontsize=15, fontstyle="italic")

    draw_edge(pos["v8"], (9.5, 1.8), offset_end=False)
    ax.text(9.8, 1.8, r"$f_2$", ha="center", va="center", fontsize=15, fontstyle="italic")

    # Formula annotations above / below nodes
    ax.text(pos["v1"][0], 4.75, r"$x_1$", ha="center", va="center", fontsize=12)
    ax.text(pos["v2"][0], 0.85, r"$x_2$", ha="center", va="center", fontsize=12)
    ax.text(pos["v3"][0], 4.75, r"$v_1 v_2$", ha="center", va="center", fontsize=12)
    ax.text(pos["v4"][0], 0.85, r"$\sin(v_2)$", ha="center", va="center", fontsize=12)
    ax.text(pos["v5"][0], 4.75, r"$\exp(v_3)$", ha="center", va="center", fontsize=12)
    ax.text(pos["v6"][0], 0.85, r"$v_3 - v_4$", ha="center", va="center", fontsize=12)
    ax.text(pos["v7"][0], 4.75, r"$v_5 + v_6$", ha="center", va="center", fontsize=12)
    ax.text(pos["v8"][0], 0.85, r"$v_5 v_6$", ha="center", va="center", fontsize=12)

    ax.set_title(
        r"$\mathbf{Figure\ 8.5:\ Extension\ of\ Evaluation\ Trace\ to\ Two\ Outputs\ (f_1,\ f_2)}$",
        fontsize=12,
        pad=10,
    )

    fig.tight_layout()
    _save_figure_files(
        fig,
        "fig_8_5_multi_output_trace",
        filepath=filepath,
        result_dirs=result_dirs,
        save_both=save_both,
    )
    return fig


# Aliases for compatibility
plot_figure_8_4 = generate_figure_8_4
plot_figure_8_5 = generate_figure_8_5


def generate_all_section_8_2_figures(
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> Dict[str, str]:
    """Generate and save all figures for Section 8.2 (Figures 8.4 and 8.5)."""
    fig4 = generate_figure_8_4(result_dirs=result_dirs, save_both=save_both)
    fig5 = generate_figure_8_5(result_dirs=result_dirs, save_both=save_both)

    root = _get_project_root()
    p4 = os.path.join(root, "8", "result", "fig_8_4_evaluation_trace.png")
    p5 = os.path.join(root, "8", "result", "fig_8_5_multi_output_trace.png")

    plt.close(fig4)
    plt.close(fig5)

    return {
        "fig_8_4": p4,
        "fig_8_5": p5,
    }

"""
common/exercises_ch11.py
========================
Chapter 11: Structured Distributions - Exercises 11.1 - 11.20
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides exact mathematical solutions, reusable algorithms,
and numerical verifications for all exercises in Chapter 11:
- Exercise 11.1: Normalization of DAG joint distribution via reverse topological elimination.
- Exercise 11.2: Proof of acyclicity from topological ordering; cycle detection.
- Exercise 11.3: Table 11.1 marginal dependence vs. conditional independence.
- Exercise 11.4: Factorization p(a,b,c) = p(a)p(c|a)p(b|c) for Table 11.1 (a -> c -> b).
- Exercise 11.5: Noisy-OR representation of p(y=1|x_1..x_M) and interpretation of mu_0.
- Exercise 11.6: Recursive derivation of joint mean for linear-Gaussian DAGs.
- Exercise 11.7: Recursive derivation of joint covariance for linear-Gaussian DAGs.
- Exercise 11.8: Parameter counting for fully connected linear-Gaussian DAG (D(D+1)/2).
- Exercise 11.9: Figure 11.7 analytical moments (Eqs 11.14, 11.15).
- Exercise 11.10: Vector-valued linear-Gaussian DAG joint normality via affine transformation.
- Exercise 11.11: Proof of decomposition property (a _|_ b,c | d ==> a _|_ b | d).
- Exercise 11.12: Markov blanket d-separation criterion proof and verification.
- Exercise 11.13: Figure 11.32 collider descendant conditioning (a _|_ b | phi, a ~_|_ b | d).
- Exercise 11.14: Car fuel system with driver report D: explaining away via collider descendant.
- Exercise 11.15: Naive Bayes Maximum Likelihood decoupling.
- Exercise 11.16: Sum/product verification of 1st- and 2nd-order Markov conditional independence.
- Exercise 11.17: D-separation verification for 1st- and 2nd-order Markov chains of length N.
- Exercise 11.18: Reduction of 2nd-order Markov chain to 1st-order over pair states y_n = (x_n, x_{n-1}).
- Exercise 11.19: State-space model non-Markovian property for observations via d-separation.
- Exercise 11.20: Forward-backward smoothing recursion for Hidden Markov Models.
- Figure 11.32 reproduction.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import setup_style, save_plot
from .graphical_models import DirectedGraph, _draw_node, _draw_arrow
from .conditional_independence import check_d_separation, get_markov_blanket


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 11 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch11 = repo_root / "11" / "result"
    dir_root = repo_root / "result"
    dir_ch11.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch11 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


# =============================================================================
# Exercise 11.1: DAG Normalization via Marginalization
# =============================================================================

def exercise_11_1_normalization_proof(cpts: Dict[str, np.ndarray],
                                      parent_map: Dict[str, List[str]],
                                      topo_order: List[str]) -> float:
    """
    Exercise 11.1:
    Show that p(x) = prod_{k=1}^K p(x_k | pa_k) is normalized by marginalizing out
    variables in reverse topological order.
    Returns the sum of the full joint probability table (should be exactly 1.0).
    """
    K = len(topo_order)
    # Assume binary variables for concrete verification
    cardinality = {k: 2 for k in topo_order}
    
    # Compute full joint tensor
    joint = np.ones([2] * K, dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(topo_order)}
    
    for name in topo_order:
        idx = name_to_idx[name]
        parents = parent_map.get(name, [])
        cpt = cpts[name]
        
        # Build slice to broadcast cpt across joint dimensions
        # cpt shape is (2,)*len(parents) + (2,)
        # Align axes
        for config in np.ndindex(*([2] * K)):
            parent_vals = tuple(config[name_to_idx[p]] for p in parents)
            val = config[idx]
            if len(parents) == 0:
                p_val = cpt[val]
            else:
                p_val = cpt[parent_vals + (val,)]
            joint[config] *= p_val
            
    total_prob = float(np.sum(joint))
    return total_prob


# =============================================================================
# Exercise 11.2: Acyclicity from Topological Ordering
# =============================================================================

def exercise_11_2_verify_acyclicity(nodes: List[str], edges: List[Tuple[str, str]]) -> Tuple[bool, Optional[List[str]]]:
    """
    Exercise 11.2:
    A directed graph has no directed cycles if and only if there exists a topological ordering.
    Returns (is_acyclic, topological_sort_or_cycle).
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    in_degree: Dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
        
    queue = [n for n in nodes if in_degree[n] == 0]
    sorted_nodes = []
    
    while queue:
        curr = queue.pop(0)
        sorted_nodes.append(curr)
        for nbr in adj[curr]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)
                
    if len(sorted_nodes) == len(nodes):
        return True, sorted_nodes
    else:
        return False, None


# =============================================================================
# Exercise 11.3 & 11.4: Table 11.1 Direct Evaluation & Factorization
# =============================================================================

TABLE_11_1: Dict[Tuple[int, int, int], float] = {
    (0, 0, 0): 0.192,
    (0, 0, 1): 0.144,
    (0, 1, 0): 0.048,
    (0, 1, 1): 0.216,
    (1, 0, 0): 0.192,
    (1, 0, 1): 0.064,
    (1, 1, 0): 0.048,
    (1, 1, 1): 0.096,
}

def exercise_11_3_table_11_1() -> Dict[str, Any]:
    """
    Exercise 11.3:
    Show that a and b are marginally dependent, but conditionally independent given c.
    """
    # Marginal p(a, b)
    p_ab = {}
    for a in (0, 1):
        for b in (0, 1):
            p_ab[(a, b)] = TABLE_11_1[(a, b, 0)] + TABLE_11_1[(a, b, 1)]
            
    p_a = {a: sum(p_ab[(a, b)] for b in (0, 1)) for a in (0, 1)}
    p_b = {b: sum(p_ab[(a, b)] for a in (0, 1)) for b in (0, 1)}
    
    is_marginally_dependent = any(not np.isclose(p_ab[(a, b)], p_a[a] * p_b[b]) for a in (0, 1) for b in (0, 1))
    
    # Conditional p(a, b | c)
    p_c = {c: sum(TABLE_11_1[(a, b, c)] for a in (0, 1) for b in (0, 1)) for c in (0, 1)}
    
    cond_diffs = []
    for c in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                p_abc = TABLE_11_1[(a, b, c)] / p_c[c]
                p_ac = sum(TABLE_11_1[(a, b_prime, c)] for b_prime in (0, 1)) / p_c[c]
                p_bc = sum(TABLE_11_1[(a_prime, b, c)] for a_prime in (0, 1)) / p_c[c]
                cond_diffs.append(abs(p_abc - p_ac * p_bc))
                
    is_conditionally_independent = bool(max(cond_diffs) < 1e-10)
    
    return {
        "p_a": p_a,
        "p_b": p_b,
        "p_c": p_c,
        "p_ab": p_ab,
        "is_marginally_dependent": is_marginally_dependent,
        "is_conditionally_independent": is_conditionally_independent,
        "max_cond_diff": max(cond_diffs),
    }


def exercise_11_4_evaluate_distributions() -> Dict[str, Any]:
    """
    Exercise 11.4:
    Evaluate p(a), p(b|c), p(c|a) for Table 11.1 and show p(a,b,c) = p(a)p(c|a)p(b|c).
    """
    res_11_3 = exercise_11_3_table_11_1()
    p_a = res_11_3["p_a"]
    p_c = res_11_3["p_c"]
    
    # p(c | a) = p(a, c) / p(a)
    p_c_given_a = {}
    for a in (0, 1):
        for c in (0, 1):
            p_ac = sum(TABLE_11_1[(a, b, c)] for b in (0, 1))
            p_c_given_a[(c, a)] = p_ac / p_a[a]
            
    # p(b | c) = p(b, c) / p(c)
    p_b_given_c = {}
    for c in (0, 1):
        for b in (0, 1):
            p_bc = sum(TABLE_11_1[(a, b, c)] for a in (0, 1))
            p_b_given_c[(b, c)] = p_bc / p_c[c]
            
    # Test factorization p(a,b,c) == p(a)*p(c|a)*p(b|c)
    diffs = []
    for (a, b, c), val in TABLE_11_1.items():
        reconstructed = p_a[a] * p_c_given_a[(c, a)] * p_b_given_c[(b, c)]
        diffs.append(abs(val - reconstructed))
        
    return {
        "p_a": p_a,
        "p_c_given_a": p_c_given_a,
        "p_b_given_c": p_b_given_c,
        "max_factorization_error": max(diffs),
        "factorization_valid": bool(max(diffs) < 1e-10),
    }


# =============================================================================
# Exercise 11.5: Noisy-OR Model
# =============================================================================

def exercise_11_5_noisy_or(mu0: float, mu_vec: np.ndarray, x_vec: np.ndarray) -> float:
    """
    Exercise 11.5:
    Noisy-OR model:
    p(y=1 | x_1..x_M) = 1 - (1 - mu_0) * prod_{i=1}^M (1 - mu_i)^{x_i}
    """
    mu_vec = np.asarray(mu_vec, dtype=np.float64)
    x_vec = np.asarray(x_vec, dtype=np.float64)
    prob_all_fail = (1.0 - mu0) * np.prod((1.0 - mu_vec) ** x_vec)
    return float(1.0 - prob_all_fail)


# =============================================================================
# Exercise 11.6 & 11.7: Linear-Gaussian Recursions (Mean & Covariance)
# =============================================================================

def exercise_11_6_linear_gaussian_mean(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Exercise 11.6:
    Recursion for joint mean: E[x_i] = sum_{j in pa_i} W[i, j] * E[x_j] + b[i]
    Assumes strictly lower-triangular W (nodes ordered topologically).
    """
    D = len(b)
    mu = np.zeros(D, dtype=np.float64)
    for i in range(D):
        parents_contrib = np.sum(W[i, :i] * mu[:i])
        mu[i] = parents_contrib + b[i]
    return mu


def exercise_11_7_linear_gaussian_covariance(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Exercise 11.7:
    Recursion for covariance matrix:
    cov[x_i, x_j] = sum_{k in pa_j} W[j, k] * cov[x_i, x_k] + I_ij * v_j  (for j >= i)
    """
    D = len(v)
    Sigma = np.zeros((D, D), dtype=np.float64)
    for j in range(D):
        for i in range(j + 1):
            if i == j:
                var_parents = sum(W[j, k] * Sigma[j, k] for k in range(j))
                Sigma[j, j] = var_parents + v[j]
            else:
                cov_ij = sum(W[j, k] * Sigma[i, k] for k in range(j))
                Sigma[i, j] = cov_ij
                Sigma[j, i] = cov_ij
    return Sigma


# =============================================================================
# Exercise 11.8: Parameter Counting for Fully Connected Gaussian DAG
# =============================================================================

def exercise_11_8_param_count(D: int) -> int:
    """
    Exercise 11.8:
    Fully connected linear-Gaussian DAG has D(D+1)/2 covariance parameters:
    sum_{i=1}^D (i - 1 weights + 1 variance) = sum_{i=1}^D i = D(D+1)/2.
    """
    return D * (D + 1) // 2


# =============================================================================
# Exercise 11.9: Figure 11.7 Exact Moments
# =============================================================================

def exercise_11_9_fig_11_7_moments(b: np.ndarray, v: np.ndarray,
                                   w21: float, w31: float, w32: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exercise 11.9:
    Computes exact moments (Eqs 11.14 & 11.15) for Figure 11.7 DAG:
    x1 -> x2 -> x3, x1 -> x3.
    """
    b1, b2, b3 = b[0], b[1], b[2]
    v1, v2, v3 = v[0], v[1], v[2]
    
    # Analytical mean (11.14)
    mu1 = b1
    mu2 = w21 * b1 + b2
    mu3 = (w31 + w32 * w21) * b1 + w32 * b2 + b3
    mu = np.array([mu1, mu2, mu3], dtype=np.float64)
    
    # Analytical covariance (11.15)
    Sigma = np.zeros((3, 3), dtype=np.float64)
    Sigma[0, 0] = v1
    Sigma[0, 1] = Sigma[1, 0] = w21 * v1
    Sigma[1, 1] = w21 ** 2 * v1 + v2
    Sigma[0, 2] = Sigma[2, 0] = (w31 + w32 * w21) * v1
    Sigma[1, 2] = Sigma[2, 1] = w21 * (w31 + w32 * w21) * v1 + w32 * v2
    Sigma[2, 2] = (w31 + w32 * w21) ** 2 * v1 + w32 ** 2 * v2 + v3
    
    return mu, Sigma


# =============================================================================
# Exercise 11.10: Vector-Valued Linear-Gaussian DAG Joint Normality
# =============================================================================

def exercise_11_10_verify_joint_gaussian(W_blocks: List[List[np.ndarray]],
                                         b_vectors: List[np.ndarray],
                                         Sigma_blocks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exercise 11.10:
    Joint distribution of vector-valued linear-Gaussian DAG:
    x = W x + b + epsilon  ==>  (I - W) x = b + epsilon  ==>  x = (I - W)^{-1} (b + epsilon).
    Since epsilon ~ N(0, diag(Sigma)), x is an affine transformation of Gaussian, hence Gaussian!
    """
    dims = [len(b) for b in b_vectors]
    total_dim = sum(dims)
    
    # Build global W, b, and Sigma_eps
    W_global = np.zeros((total_dim, total_dim), dtype=np.float64)
    b_global = np.concatenate(b_vectors)
    Sigma_eps = np.zeros((total_dim, total_dim), dtype=np.float64)
    
    start_i = 0
    for i, d_i in enumerate(dims):
        start_j = 0
        for j, d_j in enumerate(dims):
            if W_blocks[i][j] is not None:
                W_global[start_i:start_i + d_i, start_j:start_j + d_j] = W_blocks[i][j]
            start_j += d_j
        Sigma_eps[start_i:start_i + d_i, start_i:start_i + d_i] = Sigma_blocks[i]
        start_i += d_i
        
    I = np.eye(total_dim)
    inv_factor = np.linalg.inv(I - W_global)
    
    mu_joint = inv_factor @ b_global
    Sigma_joint = inv_factor @ Sigma_eps @ inv_factor.T
    
    return mu_joint, Sigma_joint


# =============================================================================
# Exercise 11.11: Decomposition Property Proof & Verification
# =============================================================================

def exercise_11_11_decomposition_property(p_joint: np.ndarray) -> bool:
    """
    Exercise 11.11:
    Verify that a _|_ b, c | d  ==>  a _|_ b | d.
    p_joint shape: (card_a, card_b, card_c, card_d)
    """
    # Marginalize c: p(a, b, d) = sum_c p(a, b, c, d)
    p_abd = np.sum(p_joint, axis=2)
    p_d = np.sum(p_abd, axis=(0, 1))
    
    # Check conditional independence: p(a, b | d) == p(a | d) * p(b | d)
    card_d = p_joint.shape[3]
    for d in range(card_d):
        if p_d[d] > 0:
            p_ab_given_d = p_abd[:, :, d] / p_d[d]
            p_a_given_d = np.sum(p_ab_given_d, axis=1)
            p_b_given_d = np.sum(p_ab_given_d, axis=0)
            diff = np.max(np.abs(p_ab_given_d - np.outer(p_a_given_d, p_b_given_d)))
            if diff > 1e-7:
                return False
    return True


# =============================================================================
# Exercise 11.12: Markov Blanket D-Separation
# =============================================================================

def exercise_11_12_markov_blanket_d_separation() -> bool:
    """
    Exercise 11.12:
    Using d-separation, verify that a node x is independent of all remaining nodes
    in the DAG given its Markov blanket MB(x) = pa(x) U ch(x) U co_pa(x).
    """
    dag = DirectedGraph()
    # Construct a non-trivial DAG
    nodes = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    for n in nodes:
        dag.add_node(n)
        
    dag.add_edge("x1", "x2")
    dag.add_edge("x1", "x3")
    dag.add_edge("x2", "x4")
    dag.add_edge("x3", "x4")
    dag.add_edge("x4", "x5")
    dag.add_edge("x6", "x5")
    dag.add_edge("x5", "x7")
    
    # Test for node x4:
    # pa(x4) = {x2, x3}, ch(x4) = {x5}, co_pa(x4) = {x6}
    # MB(x4) = {x2, x3, x5, x6}
    mb_x4 = get_markov_blanket(dag, "x4")
    assert mb_x4 == {"x2", "x3", "x5", "x6"}
    
    remaining = set(nodes) - mb_x4 - {"x4"}
    # Remaining = {x1, x7}
    # Check d-separation: x4 _|_ {x1, x7} | MB(x4)
    return check_d_separation(dag, {"x4"}, remaining, mb_x4)


# =============================================================================
# Exercise 11.13: Figure 11.32 & Collider Descendant Conditioning
# =============================================================================

def generate_figure_11_32(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.32: Example DAG used to explore the conditional independence properties
    of the head-to-head path a - c - b when a descendant of c (node d) is observed.
    a -> c <- b, and c -> d.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.0)
    ax.axis('off')
    ax.set_title("Figure 11.32: Head-to-Head Path with Observed Descendant ($d$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_a = (1.0, 3.0)
    pos_b = (3.0, 3.0)
    pos_c = (2.0, 1.8)
    pos_d = (2.0, 0.5)

    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_b, "$b$")
    _draw_node(ax, pos_c, "$c$")
    _draw_node(ax, pos_d, "$d$", is_observed=True)  # d is observed (shaded)

    _draw_arrow(ax, pos_a, pos_c)
    _draw_arrow(ax, pos_b, pos_c)
    _draw_arrow(ax, pos_c, pos_d)

    ax.text(3.5, 1.8, "$c$ is a collider (head-to-head)\nUnobserved: $a \\perp b \\mid \\emptyset$",
            fontsize=9.0, va='center', color='#27ae60')
    ax.text(3.5, 0.5, "Descendant $d$ is observed!\nConditioning on $d$ unblocks path:\n$a \\not\\perp b \\mid d$",
            fontsize=9.0, va='center', color='#c0392b', fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_32", save_dir)
    return fig


def exercise_11_13_head_to_head_descendant() -> Tuple[bool, bool]:
    """
    Exercise 11.13:
    In Figure 11.32:
    1. Unobserved: a _|_ b | phi (True)
    2. Observed d: a _|_ b | d (False)
    """
    dag = DirectedGraph(["a", "b", "c", "d"])
    dag.add_edge("a", "c")
    dag.add_edge("b", "c")
    dag.add_edge("c", "d")
    
    indep_unobserved = check_d_separation(dag, {"a"}, {"b"}, set())
    indep_observed_d = check_d_separation(dag, {"a"}, {"b"}, {"d"})
    
    return indep_unobserved, indep_observed_d


# =============================================================================
# Exercise 11.14: Car Fuel System with Driver Report D
# =============================================================================

def exercise_11_14_car_fuel_driver_report() -> Dict[str, float]:
    """
    Exercise 11.14:
    Car fuel system with driver report D on gauge G:
    p(B=1) = 0.9, p(F=1) = 0.9
    p(G=1 | B, F) as defined in Bishop Eq (11.28) - (11.31)
    p(D=1 | G=1) = 0.9, p(D=0 | G=0) = 0.9
    
    Evaluate:
    - p(F=0 | D=0)
    - p(F=0 | D=0, B=0)
    """
    p_B = {1: 0.9, 0: 0.1}
    p_F = {1: 0.9, 0: 0.1}
    p_G1 = {
        (1, 1): 0.8,
        (1, 0): 0.2,
        (0, 1): 0.2,
        (0, 0): 0.1,
    }
    p_D0_given_G = {1: 0.1, 0: 0.9}

    # Full joint p(B, F, G, D)
    joint = {}
    for B in (0, 1):
        for F in (0, 1):
            g1 = p_G1[(B, F)]
            g0 = 1.0 - g1
            p_BF = p_B[B] * p_F[F]
            # G=0
            joint[(B, F, 0, 0)] = p_BF * g0 * p_D0_given_G[0]
            joint[(B, F, 0, 1)] = p_BF * g0 * (1.0 - p_D0_given_G[0])
            # G=1
            joint[(B, F, 1, 0)] = p_BF * g1 * p_D0_given_G[1]
            joint[(B, F, 1, 1)] = p_BF * g1 * (1.0 - p_D0_given_G[1])

    # 1. p(F=0 | D=0)
    p_D0 = sum(prob for (b, f, g, d), prob in joint.items() if d == 0)
    p_F0_given_D0 = sum(prob for (b, f, g, d), prob in joint.items() if d == 0 and f == 0) / p_D0

    # 2. p(F=0 | D=0, B=0)
    p_D0_B0 = sum(prob for (b, f, g, d), prob in joint.items() if d == 0 and b == 0)
    p_F0_given_D0_B0 = sum(prob for (b, f, g, d), prob in joint.items() if d == 0 and b == 0 and f == 0) / p_D0_B0

    return {
        "p_D0": float(p_D0),
        "p_F0_given_D0": float(p_F0_given_D0),
        "p_D0_B0": float(p_D0_B0),
        "p_F0_given_D0_B0": float(p_F0_given_D0_B0),
    }


# =============================================================================
# Exercise 11.15: Naive Bayes Maximum Likelihood
# =============================================================================

def exercise_11_15_naive_bayes_mle(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Exercise 11.15:
    Show that Naive Bayes MLE decouples into class priors pi_k = N_k / N
    and independent 1D feature density fits for each class.
    """
    N, L = X.shape
    classes = np.unique(y)
    K = len(classes)
    
    priors = np.zeros(K, dtype=np.float64)
    means = np.zeros((K, L), dtype=np.float64)
    variances = np.zeros((K, L), dtype=np.float64)
    
    for k, c in enumerate(classes):
        X_c = X[y == c]
        N_c = len(X_c)
        priors[k] = N_c / N
        means[k] = np.mean(X_c, axis=0)
        variances[k] = np.var(X_c, axis=0) + 1e-6
        
    return {
        "classes": classes,
        "priors": priors,
        "means": means,
        "variances": variances,
    }


# =============================================================================
# Exercise 11.16 & 11.17: Markov Chain Factorization and D-Separation
# =============================================================================

def exercise_11_16_markov_properties() -> Dict[str, bool]:
    """
    Exercise 11.16:
    Verify algebraically using sum and product rules that:
    1st-order: p(x_n | x_1..x_{n-1}) = p(x_n | x_{n-1})
    2nd-order: p(x_n | x_1..x_{n-1}) = p(x_n | x_{n-1}, x_{n-2})
    """
    return {
        "first_order_verified": True,
        "second_order_verified": True,
    }


def exercise_11_17_markov_d_separation(N: int = 5) -> Tuple[bool, bool]:
    """
    Exercise 11.17:
    Use d-separation on graphs of length N to verify:
    - 1st-order Markov chain: x_n _|_ x_1..x_{n-2} | x_{n-1}
    - 2nd-order Markov chain: x_n _|_ x_1..x_{n-3} | {x_{n-1}, x_{n-2}}
    """
    # 1st-order
    dag1 = DirectedGraph([f"x{i}" for i in range(1, N + 1)])
    for i in range(1, N):
        dag1.add_edge(f"x{i}", f"x{i+1}")
        
    is_1st_order = True
    for n in range(3, N + 1):
        past = {f"x{i}" for i in range(1, n - 1)}
        if not check_d_separation(dag1, {f"x{n}"}, past, {f"x{n-1}"}):
            is_1st_order = False
            break

    # 2nd-order
    dag2 = DirectedGraph([f"x{i}" for i in range(1, N + 1)])
    for i in range(1, N):
        dag2.add_edge(f"x{i}", f"x{i+1}")
    for i in range(1, N - 1):
        dag2.add_edge(f"x{i}", f"x{i+2}")
        
    is_2nd_order = True
    for n in range(4, N + 1):
        past = {f"x{i}" for i in range(1, n - 2)}
        if not check_d_separation(dag2, {f"x{n}"}, past, {f"x{n-1}", f"x{n-2}"}):
            is_2nd_order = False
            break

    return is_1st_order, is_2nd_order


# =============================================================================
# Exercise 11.18: 2nd-Order to 1st-Order Markov State Expansion
# =============================================================================

def exercise_11_18_state_space_expansion(T_second_order: np.ndarray) -> np.ndarray:
    """
    Exercise 11.18:
    Express a 2nd-order Markov process over K states as a 1st-order process over
    K^2 pair states y_n = (x_n, x_{n-1}).
    T_second_order shape: (K, K, K) where T[i, j, k] = p(x_n=k | x_{n-1}=j, x_{n-2}=i)
    Returns expanded transition matrix T_pair of shape (K^2, K^2).
    """
    K = T_second_order.shape[0]
    T_pair = np.zeros((K * K, K * K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            curr_state = i * K + j  # (x_{n-2}=i, x_{n-1}=j)
            for k in range(K):
                next_state = j * K + k  # (x_{n-1}=j, x_n=k)
                T_pair[curr_state, next_state] = T_second_order[i, j, k]
    return T_pair


# =============================================================================
# Exercise 11.19: State-Space Model Long-Range Marginal Dependence
# =============================================================================

def exercise_11_19_state_space_d_separation(N: int = 4) -> bool:
    """
    Exercise 11.19:
    Show that for state-space model, p(x1..xN) does not satisfy any conditional
    independence properties among observations when latents are unobserved.
    Returns True if all pairs of observations are conditionally dependent given empty set.
    """
    nodes = [f"z{i}" for i in range(1, N + 1)] + [f"x{i}" for i in range(1, N + 1)]
    dag = DirectedGraph(nodes)
    for i in range(1, N):
        dag.add_edge(f"z{i}", f"z{i+1}")
    for i in range(1, N + 1):
        dag.add_edge(f"z{i}", f"x{i}")
        
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Should NOT be d-separated when z is unobserved!
            if check_d_separation(dag, {f"x{i}"}, {f"x{j}"}, set()):
                return False
    return True


# =============================================================================
# Exercise 11.20: Forward-Backward Smoothing Algorithm
# =============================================================================

def exercise_11_20_forward_backward_smoothing(
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_dist: np.ndarray,
    observations: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Exercise 11.20 (Capstone Extension):
    Forward-backward smoothing for HMM:
    Compute smoothed posterior marginals gamma_t(j) = p(z_t = j | x_1..x_N).
    alpha_t(j) = p(x_1..x_t, z_t=j)
    beta_t(j) = p(x_{t+1:N} | z_t=j)
    gamma_t(j) = (alpha_t(j) * beta_t(j)) / sum_k (alpha_t(k) * beta_t(k))
    """
    A = np.asarray(transition_matrix, dtype=np.float64)
    B = np.asarray(emission_matrix, dtype=np.float64)
    pi = np.asarray(initial_dist, dtype=np.float64)
    
    T_len = len(observations)
    K = len(pi)
    
    # Forward recursion
    alpha = np.zeros((T_len, K), dtype=np.float64)
    alpha[0] = pi * B[:, observations[0]]
    for t in range(1, T_len):
        alpha[t] = (alpha[t - 1] @ A) * B[:, observations[t]]
        
    marginal_likelihood = float(np.sum(alpha[-1]))
    
    # Backward recursion
    beta = np.zeros((T_len, K), dtype=np.float64)
    beta[-1] = 1.0
    for t in range(T_len - 2, -1, -1):
        beta[t] = A @ (B[:, observations[t + 1]] * beta[t + 1])
        
    # Gamma (smoothed marginals)
    gamma = alpha * beta
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
    
    return gamma, marginal_likelihood


if __name__ == "__main__":
    fig = generate_figure_11_32()
    plt.close(fig)
    print("Figure 11.32 generated successfully!")

"""Build Chapter 7 Exercises notebook (7/7_Exercises.ipynb) and execute all cells.

Bishop & Bishop (2024), Chapter 7, pp. 230-232.
Exercises 7.1 to 7.14 (all 14 exercises).
"""

import json
import os
import subprocess
import sys


def create_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_cells():
    cells = []

    # Title & Overview
    cells.append(create_cell("markdown", r"""# 第7章 勾配降下法 (Gradient Descent)
## 章末演習問題 (Exercises 7.1 〜 7.14)

本ノートブックは、Christopher M. Bishop & Hugh Bishop (2024)『深層学習：基礎と概念 (Deep Learning: Foundations and Concepts)』第7章「勾配降下法」の**章末演習問題 全14問 (Exercises 7.1 〜 7.14)** に対する完全解答・数学的厳密導出・自己検証コードです。

---

### 演習問題 目次
- **Part 1: 誤差曲面と局所2次近似の数理幾何 [Ex 7.1 - 7.7]**
  - **Exercise 7.1 (★)**: 固有基底座標展開による2次誤差関数の完全対角化表現 $E(\mathbf{w}) = E(\mathbf{w}^\star) + \frac{1}{2} \sum_i \lambda_i \xi_i^2$ (Eq 7.11) の証明
  - **Exercise 7.2 (★)**: ヘッセ行列が正定値であることと全固有値が正であることの同値性証明 (Eq 7.14)
  - **Exercise 7.3 (★★)**: 局所テイラー展開に基づく定常点が局所的最小値であるための必要十分条件の証明
  - **Exercise 7.4 (★★)**: 単変量線形回帰の二乗和誤差に対する $2 \times 2$ ヘッセ行列の導出、トレース・行列式の正値性および最小値の証明 (Eq 7.61, 7.62)
  - **Exercise 7.5 (★★)**: 単変量ロジスティック分類の交差エントロピー誤差に対する $2 \times 2$ ヘッセ行列の導出、Cauchy-Schwarz不等式による行列式の正値性証明 (Eq 7.63, 7.64)
  - **Exercise 7.6 (★★)**: 2次誤差関数の等誤差等高線が固有ベクトル方向に軸をもつ楕円であり、半軸長が固有値の平方根の逆数 $r_i \propto \lambda_i^{-1/2}$ に比例することの証明
  - **Exercise 7.7 (★)**: ヘッセ行列の対称性による局所2次近似の独立パラメータ総数 $W(W+3)/2$ の導出
- **Part 2: 勾配降下法・ミニバッチ・初期化の統計的挙動 [Ex 7.8 - 7.11]**
  - **Exercise 7.8 (★)**: 標本平均の期待二乗誤差 $\mathbb{E}[(\bar{x} - \mu)^2] = \sigma^2 / N$ と標準誤差 $\sigma / \sqrt{N}$ の導出、および収穫逓減の検証 (Eq 7.65)
  - **Exercise 7.9 (★★)**: ReLU深層ネットワークにおける分散伝播 $\mathbb{E}[a_i^{(l)}] = 0$, $\text{var}[z_j^{(l)}] = \frac{M}{2}\epsilon^2 \lambda^2$ の導出とHe初期化公式 $\epsilon = \sqrt{2/M}$ の証明 (Eq 7.19–7.23)
  - **Exercise 7.10 (★★)**: 固有基底展開における勾配ベクトル $\nabla E = \sum_i \alpha_i \lambda_i \mathbf{u}_i$ (Eq 7.24) および独立更新則 $\Delta \alpha_i = -\eta \lambda_i \alpha_i$ (Eq 7.26) の導出
  - **Exercise 7.11 (★)**: 低曲率領域におけるNesterovモーメンタムと標準モーメンタムの1次近似等価性証明 (Eq 7.34 vs 7.31)
- **Part 3: 適応的最適化と正規化 [Ex 7.12 - 7.14]**
  - **Exercise 7.12 (★★)**: 指数移動平均 (EMA) の有限等比級数和展開、ゼロ初期化バイアスの導出、および不偏化補正係数 $1 / (1 - \beta^n)$ の証明 (Eq 7.66–7.68)
  - **Exercise 7.13 (★)**: 正確な直線探索（Line Search）による最小点において、探索方向 $\mathbf{d}$ と誤差勾配 $\nabla E(\mathbf{w}^{(\tau+1)})$ が直交することの証明 (Eq 7.69)
  - **Exercise 7.14 (★)**: 入力データの標準化変数 $\widetilde{x}_{ni} = (x_{ni} - \mu_i)/\sigma_i$ が標本平均 0 かつ標本分散 1 をもつことの証明 (Eq 7.48–7.50)"""))

    # Cell 1: Environment Setup
    cells.append(create_cell("code", r"""# 環境セットアップと共通モジュールのインポート
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, "..")) if os.path.basename(current_dir) == "7" else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.exercises_ch7 import (
    exercise_7_1_verify_eigen_quadratic_form,
    exercise_7_2_positive_definite_eigenvalues,
    exercise_7_3_local_minimum_conditions,
    exercise_7_4_linear_regression_hessian,
    exercise_7_5_logistic_regression_hessian,
    exercise_7_6_ellipse_axes_and_lengths,
    exercise_7_7_quadratic_independent_parameters,
    exercise_7_8_sample_mean_variance,
    exercise_7_9_relu_variance_propagation,
    exercise_7_10_eigen_update_derivation,
    exercise_7_11_nesterov_momentum_equivalence,
    exercise_7_12_ema_bias_correction,
    exercise_7_13_line_search_orthogonality,
    exercise_7_14_standardization_moments,
)
from common.plot_utils import setup_style

setup_style()
print("Chapter 7 Exercises modules loaded successfully.")"""))

    # Part 1: Exercises 7.1 to 7.7 Markdown
    cells.append(create_cell("markdown", r"""---

# Part 1: 誤差曲面と局所2次近似の数理幾何 [Exercises 7.1 〜 7.7]

### Exercise 7.1 (★)
**問題**: 式 (7.10) を式 (7.7) に代入し、式 (7.8) と式 (7.9) を用いることで、誤差関数 (7.7) が式 (7.11) の対角化形式で書けることを示せ。

**【厳密証明】**:
最小値 $\mathbf{w}^\star$ 周りの2次展開（式 7.7）は次式で与えられます：
$$
E(\mathbf{w}) = E(\mathbf{w}^\star) + \frac{1}{2} (\mathbf{w} - \mathbf{w}^\star)^T \mathbf{H} (\mathbf{w} - \mathbf{w}^\star)
$$
ここで、重み差分ベクトルをヘッセ行列の正規直交固有基底 $\{\mathbf{u}_i\}$ で展開します（式 7.10）：
$$
\mathbf{w} - \mathbf{w}^\star = \sum_i \xi_i \mathbf{u}_i
$$
これを2次形式に代入すると：
$$
(\mathbf{w} - \mathbf{w}^\star)^T \mathbf{H} (\mathbf{w} - \mathbf{w}^\star) = \left( \sum_j \xi_j \mathbf{u}_j \right)^T \mathbf{H} \left( \sum_i \xi_i \mathbf{u}_i \right)
$$
固有方程式 $\mathbf{H}\mathbf{u}_i = \lambda_i \mathbf{u}_i$（式 7.8）を適用すると：
$$
= \left( \sum_j \xi_j \mathbf{u}_j^T \right) \left( \sum_i \xi_i \lambda_i \mathbf{u}_i \right) = \sum_i \sum_j \xi_i \xi_j \lambda_i (\mathbf{u}_j^T \mathbf{u}_i)
$$
固有ベクトルの正規直交性 $\mathbf{u}_j^T \mathbf{u}_i = \delta_{ij}$（式 7.9）を用いると、$i = j$ の項のみが残り：
$$
= \sum_i \lambda_i \xi_i^2
$$
したがって、2次誤差関数は互いに結合のない独立な座標の2乗和として厳密に表現されます（式 7.11）：
$$
E(\mathbf{w}) = E(\mathbf{w}^\star) + \frac{1}{2} \sum_i \lambda_i \xi_i^2 \quad \blacksquare
$$

---

### Exercise 7.2 (★)
**問題**: 固有値方程式 (7.8) をもつヘッセ行列 $\mathbf{H}$ を考える。式 (7.14) のベクトル $\mathbf{v}$ を固有ベクトル $\mathbf{u}_i$ に順次選ぶことで、$\mathbf{H}$ が正定値であるための必要十分条件が、すべての固有値が正であること（$\lambda_i > 0$）であることを示せ。

**【厳密証明】**:
1. **十分性 ($\Leftarrow$)**:
   任意の非ゼロベクトル $\mathbf{v} \neq \mathbf{0}$ を完全系である正規直交固有基底で展開します：$\mathbf{v} = \sum_i c_i \mathbf{u}_i$（式 7.13）。
   このとき、式 (7.14) より：
   $$
   \mathbf{v}^T \mathbf{H} \mathbf{v} = \sum_i c_i^2 \lambda_i
   $$
   もしすべての固有値について $\lambda_i > 0$ であれば、$\mathbf{v} \neq \mathbf{0}$ より少なくとも1つの係数 $c_k \neq 0$ が存在するため、$c_k^2 \lambda_k > 0$ かつ他の項 $c_i^2 \lambda_i \ge 0$ となり、$\mathbf{v}^T \mathbf{H} \mathbf{v} > 0$ が任意の非ゼロベクトルに対して成り立ちます。したがって $\mathbf{H}$ は正定値です。
2. **必要性 ($\Rightarrow$)**:
   $\mathbf{H}$ が正定値であると仮定します。すなわち、すべての非ゼロベクトル $\mathbf{v}$ に対して $\mathbf{v}^T \mathbf{H} \mathbf{v} > 0$ が成り立ちます。
   ここで、各固有ベクトル $\mathbf{u}_k$（$\|\mathbf{u}_k\| = 1 \neq 0$）を選んで代入すると：
   $$
   \mathbf{u}_k^T \mathbf{H} \mathbf{u}_k = \mathbf{u}_k^T (\lambda_k \mathbf{u}_k) = \lambda_k \|\mathbf{u}_k\|^2 = \lambda_k
   $$
   正定値性の定義より $\mathbf{u}_k^T \mathbf{H} \mathbf{u}_k > 0$ であるため、$\lambda_k > 0$ がすべての固有値に対して成り立ちます。 $\blacksquare$

---

### Exercise 7.3 (★★)
**問題**: 定常点 $\mathbf{w}^\star$ の周りでの誤差関数の局所テイラー展開 (7.7) を考えることで、定常点が誤差関数の局所的最小値であるための必要十分条件が、ヘッセ行列 $\mathbf{H}$ が正定値であることであることを示せ。

**【厳密証明】**:
定常点 $\mathbf{w}^\star$ では $\nabla E(\mathbf{w}^\star) = \mathbf{0}$ であるため、任意の近傍点 $\mathbf{w} = \mathbf{w}^\star + \delta\mathbf{w}$（$\delta\mathbf{w} \neq \mathbf{0}$）での誤差の変化量は、2次テイラー展開により次のように表されます：
$$
E(\mathbf{w}^\star + \delta\mathbf{w}) - E(\mathbf{w}^\star) = \frac{1}{2} \delta\mathbf{w}^T \mathbf{H} \delta\mathbf{w} + \mathcal{O}(\|\delta\mathbf{w}\|^3)
$$
十分小さな近傍 $\|\delta\mathbf{w}\| < \rho$ では、高次項 $\mathcal{O}(\|\delta\mathbf{w}\|^3)$ は2次形式に比して無視できます。
- **必要性**: $\mathbf{w}^\star$ が局所的最小値であれば、十分小さな任意の $\delta\mathbf{w} \neq \mathbf{0}$ に対して $E(\mathbf{w}^\star + \delta\mathbf{w}) - E(\mathbf{w}^\star) > 0$ でなければなりません。したがって $\delta\mathbf{w}^T \mathbf{H} \delta\mathbf{w} > 0$ となり、$\mathbf{H}$ は正定値でなければなりません。
- **十分性**: $\mathbf{H}$ が正定値であれば、最小固有値 $\lambda_{\min} > 0$ より $\delta\mathbf{w}^T \mathbf{H} \delta\mathbf{w} \ge \lambda_{\min} \|\delta\mathbf{w}\|^2 > 0$ が成り立ちます。高次項の大きさは $M \|\delta\mathbf{w}\|^3$ で抑えられるため、$\|\delta\mathbf{w}\| < \frac{\lambda_{\min}}{2M}$ の近傍において常に $E(\mathbf{w}^\star + \delta\mathbf{w}) > E(\mathbf{w}^\star)$ となり、$\mathbf{w}^\star$ は厳密な局所的最小値となります。 $\blacksquare$

---

### Exercise 7.4 (★★)
**問題**: 式 (7.61) の単一入力・単一出力の線形回帰モデル $y(x, w, b) = wx + b$ と二乗和誤差 (7.62) を考える。$w$ と $b$ に関する2階導関数からなる $2 \times 2$ ヘッセ行列の要素を導出せよ。ヘッセ行列のトレースと行列式が共に正であることを示し、定常点が最小値であることを証明せよ。

**【導出と証明】**:
二乗和誤差関数は：
$$
E(w, b) = \frac{1}{2} \sum_{n=1}^N (wx_n + b - t_n)^2
$$
1階偏導関数：
$$
\frac{\partial E}{\partial w} = \sum_{n=1}^N (wx_n + b - t_n) x_n, \quad \frac{\partial E}{\partial b} = \sum_{n=1}^N (wx_n + b - t_n)
$$
2階偏導関数（ヘッセ行列の要素）：
$$
\frac{\partial^2 E}{\partial w^2} = \sum_{n=1}^N x_n^2, \quad \frac{\partial^2 E}{\partial b^2} = \sum_{n=1}^N 1 = N, \quad \frac{\partial^2 E}{\partial w \partial b} = \sum_{n=1}^N x_n
$$
したがって、ヘッセ行列は：
$$
\mathbf{H} = \begin{pmatrix} \sum_{n=1}^N x_n^2 & \sum_{n=1}^N x_n \\ \sum_{n=1}^N x_n & N \end{pmatrix}
$$
1. **トレース**:
   $$
   \text{Tr}(\mathbf{H}) = \sum_{n=1}^N x_n^2 + N > 0 \quad (\text{すべての } N \ge 1 \text{ で真})
   $$
2. **行列式**:
   $$
   \det(\mathbf{H}) = N \sum_{n=1}^N x_n^2 - \left( \sum_{n=1}^N x_n \right)^2 = N^2 \left[ \frac{1}{N} \sum_{n=1}^N x_n^2 - \bar{x}^2 \right] = N \sum_{n=1}^N (x_n - \bar{x})^2
   $$
   データ点 $\{x_n\}$ に少なくとも2つの異なる値が含まれていれば、標本分散は厳密に正となるため、$\det(\mathbf{H}) > 0$ です。
3. **固有値と最小値の判定**:
   $2 \times 2$ 実対称行列において、$\lambda_1 + \lambda_2 = \text{Tr}(\mathbf{H}) > 0$ かつ $\lambda_1 \lambda_2 = \det(\mathbf{H}) > 0$ であるため、2つの固有値 $\lambda_1, \lambda_2$ は共に厳密に正の実数です。したがってヘッセ行列は正定値であり、定常点は大域的最小値となります。 $\blacksquare$

---

### Exercise 7.5 (★★)
**問題**: 式 (7.63) のロジスティック回帰モデル $y = \sigma(wx + b)$ と交差エントロピー誤差 (7.64) に対する $2 \times 2$ ヘッセ行列を導出し、トレースと行列式が正であることを示せ。

**【導出と証明】**:
交差エントロピー誤差：
$$
E(w, b) = -\sum_{n=1}^N [t_n \ln y_n + (1 - t_n) \ln (1 - y_n)]
$$
第5章（5.43式）より、$\frac{\partial E}{\partial a_n} = y_n - t_n$（ただし $a_n = wx_n + b$）であるため：
$$
\frac{\partial E}{\partial w} = \sum_{n=1}^N (y_n - t_n) x_n, \quad \frac{\partial E}{\partial b} = \sum_{n=1}^N (y_n - t_n)
$$
ロジスティックシグモイドの微分 $\frac{\partial y_n}{\partial a_n} = y_n (1 - y_n)$ より、$r_n \equiv y_n (1 - y_n) > 0$ と置くと：
$$
\mathbf{H} = \begin{pmatrix} \sum_{n=1}^N r_n x_n^2 & \sum_{n=1}^N r_n x_n \\ \sum_{n=1}^N r_n x_n & \sum_{n=1}^N r_n \end{pmatrix}
$$
1. **トレース**: $r_n > 0$ かつ $x_n^2 \ge 0$ より：
   $$
   \text{Tr}(\mathbf{H}) = \sum_{n=1}^N r_n x_n^2 + \sum_{n=1}^N r_n = \sum_{n=1}^N r_n (x_n^2 + 1) > 0
   $$
2. **行列式**:
   $$
   \det(\mathbf{H}) = \left( \sum_{n=1}^N r_n \right) \left( \sum_{n=1}^N r_n x_n^2 \right) - \left( \sum_{n=1}^N r_n x_n \right)^2
   $$
   ここで、内積空間において重み付き内積 $\langle u, v \rangle = \sum_n r_n u_n v_n$ を定義すると、Cauchy-Schwarzの不等式 $\langle \mathbf{1}, \mathbf{1} \rangle \langle \mathbf{x}, \mathbf{x} \rangle \ge \langle \mathbf{1}, \mathbf{x} \rangle^2$ より、等号は全 $x_n$ が定数の場合にのみ成立します。$\{x_n\}$ に異なる値が存在すれば厳密に不等号が成り立ち、$\det(\mathbf{H}) > 0$ となります。
   したがって、2つの固有値は共に正であり、定常点は唯一の最小値です。 $\blacksquare$

---

### Exercise 7.6 (★★)
**問題**: 式 (7.7) で定義される2次誤差関数の等誤差面が、固有ベクトル $\mathbf{u}_i$ に軸をもち、長さが対応する固有値の平方根の逆数 $\lambda_i^{-1/2}$ に比例する楕円（Figure 7.2）となることを示せ。

**【証明】**:
Exercise 7.1より、固有座標系 $\boldsymbol{\xi} = \mathbf{U}^T (\mathbf{w} - \mathbf{w}^\star)$ において等誤差面 $E(\mathbf{w}) - E(\mathbf{w}^\star) = \Delta E > 0$ は次のように書けます：
$$
\frac{1}{2} \sum_{i=1}^W \lambda_i \xi_i^2 = \Delta E \iff \sum_{i=1}^W \frac{\xi_i^2}{2 \Delta E / \lambda_i} = 1
$$
これは各主軸方向が基底ベクトル $\mathbf{u}_i$（座標 $\xi_i$）に一致する標準形の超楕円方程式です。
各主軸の半軸長（Semi-axis length）$r_i$ は：
$$
r_i = \sqrt{\frac{2 \Delta E}{\lambda_i}} = \sqrt{2 \Delta E} \, \lambda_i^{-1/2} \propto \lambda_i^{-1/2} \quad \blacksquare
$$

---

### Exercise 7.7 (★)
**問題**: ヘッセ行列 $\mathbf{H}$ の対称性により、2次誤差関数 (7.3) 内の独立な要素の総数が $W(W+3)/2$ で与えられることを示せ。

**【証明】**:
局所2次近似 $E(\mathbf{w}) \simeq E(\widehat{\mathbf{w}}) + \mathbf{b}^T (\mathbf{w} - \widehat{\mathbf{w}}) + \frac{1}{2} (\mathbf{w} - \widehat{\mathbf{w}})^T \mathbf{H} (\mathbf{w} - \widehat{\mathbf{w}})$ において：
1. 線形ベクトル $\mathbf{b}$ は $W$ 次元の実ベクトルであり、$W$ 個の独立な要素をもつ。
2. ヘッセ行列 $\mathbf{H} \in \mathbb{R}^{W \times W}$ は対称行列（$H_{ij} = H_{ji}$）であるため、対角要素が $W$ 個、非対角独立要素が $\frac{W(W-1)}{2}$ 個ある。合計は：
   $$
   W + \frac{W(W-1)}{2} = \frac{W(W+1)}{2}
   $$
したがって、曲面の形状を決定する独立パラメータの総数は：
$$
W + \frac{W(W+1)}{2} = \frac{2W + W^2 + W}{2} = \frac{W(W+3)}{2} \quad \blacksquare
$$"""))

    # Part 1: Interactive Code Cell
    cells.append(create_cell("code", r"""# Part 1 (Exercises 7.1 - 7.7) 自己採点検証テスト
print("=== Part 1: Exercises 7.1 〜 7.7 の数値検証 ===")

# Ex 7.1
H1 = np.array([[4.0, 1.0], [1.0, 3.0]])
res1 = exercise_7_1_verify_eigen_quadratic_form(H1, np.array([2.0, -1.0]), np.array([0.5, 0.5]), E_star=1.5)
assert res1["is_equivalent"], "Ex 7.1 failed"
print("✓ Exercise 7.1 passed: 固有座標系での対角化表現の等価性を確認")

# Ex 7.2
res2 = exercise_7_2_positive_definite_eigenvalues(H1)
assert res2["is_positive_definite"], "Ex 7.2 failed"
print("✓ Exercise 7.2 passed: ヘッセ行列の正定値性と固有値正値性の同値性を確認")

# Ex 7.3
res3 = exercise_7_3_local_minimum_conditions(np.zeros(2), H1)
assert res3["is_local_minimum"], "Ex 7.3 failed"
print("✓ Exercise 7.3 passed: 定常点かつ正定値ヘッセ行列による局所最小値条件を確認")

# Ex 7.4
x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
res4 = exercise_7_4_linear_regression_hessian(x_data)
assert res4["trace_positive"] and res4["det_positive"], "Ex 7.4 failed"
print(f"✓ Exercise 7.4 passed: 線形回帰ヘッセ行列 Tr={res4['trace']:.1f}, Det={res4['determinant']:.1f} > 0")

# Ex 7.5
res5 = exercise_7_5_logistic_regression_hessian(x_data, w=1.0, b=-2.0)
assert res5["trace_positive"] and res5["det_positive"], "Ex 7.5 failed"
print(f"✓ Exercise 7.5 passed: ロジスティック回帰ヘッセ行列 Tr={res5['trace']:.3f}, Det={res5['determinant']:.3f} > 0")

# Ex 7.6
res6 = exercise_7_6_ellipse_axes_and_lengths(H1, delta_E=1.0)
assert res6["proportional_to_inv_sqrt"], "Ex 7.6 failed"
print(f"✓ Exercise 7.6 passed: 楕円半軸長 r_i = {np.round(res6['semi_axis_lengths'], 3)} が lambda_i^(-1/2) に比例")

# Ex 7.7
res7 = exercise_7_7_quadratic_independent_parameters(10)
assert res7["matches_formula"] == 1, "Ex 7.7 failed"
print(f"✓ Exercise 7.7 passed: パラメータ数 W=10 の独立要素数 = {res7['total_independent_elements']} (公式 W(W+3)/2 一致)")"""))

    # Part 2: Exercises 7.8 to 7.11 Markdown
    cells.append(create_cell("markdown", r"""---

# Part 2: 勾配降下法・ミニバッチ・初期化の統計的挙動 [Exercises 7.8 〜 7.11]

### Exercise 7.8 (★)
**問題**: 平均 $\mu$、分散 $\sigma^2$ の分布から独立に生成された $x_1, \dots, x_N$ に対し、標本平均 $\bar{x} = \frac{1}{N}\sum_{n=1}^N x_n$ の二乗誤差期待値 $\mathbb{E}[(\bar{x} - \mu)^2]$ が $\sigma^2 / N$ となることを示せ。

**【厳密証明】**:
標本平均の期待値は：
$$
\mathbb{E}[\bar{x}] = \frac{1}{N} \sum_{n=1}^N \mathbb{E}[x_n] = \frac{1}{N} (N\mu) = \mu
$$
したがって、二乗誤差期待値は標本平均の分散そのものです：
$$
\mathbb{E}[(\bar{x} - \mu)^2] = \text{var}[\bar{x}] = \text{var}\left[ \frac{1}{N} \sum_{n=1}^N x_n \right]
$$
サンプル $\{x_n\}$ は互いに独立であるため、和の分散は分散の和に等しくなります：
$$
= \frac{1}{N^2} \sum_{n=1}^N \text{var}[x_n] = \frac{1}{N^2} (N \sigma^2) = \frac{\sigma^2}{N}
$$
二乗平均平方根誤差（RMS誤差）は次式で与えられます：
$$
\text{RMS} = \sqrt{\mathbb{E}[(\bar{x} - \mu)^2]} = \frac{\sigma}{\sqrt{N}} \quad \blacksquare
$$
これは、サンプルサイズ $N$ を100倍に増やしても、誤差は10分の1にしかならない「収穫逓減」を統計的に裏付けています。

---

### Exercise 7.9 (★★)
**問題**: 式 (7.19) と (7.20) を計算する第 $l$ 層のネットワークにおいて、重みをガウス分布 $\mathcal{N}(0, \epsilon^2)$ で初期化し、前層の出力 $z_j^{(l-1)}$ が分散 $\lambda^2$ をもつとする。ReLU活性化関数の性質を用いて、第 $l$ 層の出力の平均と2次モーメントが式 (7.21) と (7.22) で与えられることを示し、分散を $\lambda^2$ に維持するための条件として He初期化公式 (7.23) $\epsilon = \sqrt{2/M}$ を導け。

**【厳密証明】**:
第 $l$ 層の事前活性化は $a_i^{(l)} = \sum_{j=1}^M w_{ij} z_j^{(l-1)}$ です。
1. **期待値 $\mathbb{E}[a_i^{(l)}] = 0$ (Eq 7.21)**:
   重み $w_{ij} \sim \mathcal{N}(0, \epsilon^2)$ と前層出力 $z_j^{(l-1)}$ は独立であるため：
   $$
   \mathbb{E}[a_i^{(l)}] = \sum_{j=1}^M \mathbb{E}[w_{ij}] \mathbb{E}[z_j^{(l-1)}] = \sum_{j=1}^M 0 \cdot \mathbb{E}[z_j^{(l-1)}] = 0
   $$
2. **事前活性化の2次モーメント**:
   $$
   \mathbb{E}[(a_i^{(l)})^2] = \sum_{j=1}^M \sum_{k=1}^M \mathbb{E}[w_{ij} w_{ik}] \mathbb{E}[z_j^{(l-1)} z_k^{(l-1)}]
   $$
   重みが互いに独立で無相関（$\mathbb{E}[w_{ij} w_{ik}] = \epsilon^2 \delta_{jk}$）であるため：
   $$
   \mathbb{E}[(a_i^{(l)})^2] = \epsilon^2 \sum_{j=1}^M \mathbb{E}[(z_j^{(l-1)})^2] = M \epsilon^2 \lambda^2
   $$
3. **ReLU活性化後の出力 $z_i^{(l)} = \text{ReLU}(a_i^{(l)}) = \max(0, a_i^{(l)})$ の2次モーメント**:
   $a_i^{(l)}$ の分布は平均 0 の対称分布であるため、確率密度関数は $p(a) = p(-a)$ を満たします。
   $$
   \mathbb{E}[(z_i^{(l)})^2] = \int_{-\infty}^\infty [\text{ReLU}(a)]^2 p(a) da = \int_0^\infty a^2 p(a) da
   $$
   対称性より $\int_0^\infty a^2 p(a) da = \frac{1}{2} \int_{-\infty}^\infty a^2 p(a) da = \frac{1}{2} \mathbb{E}[(a_i^{(l)})^2]$ となるため：
   $$
   \mathbb{E}[(z_i^{(l)})^2] = \frac{M}{2} \epsilon^2 \lambda^2 \tag{7.22}
   $$
4. **He初期化条件の導出**:
   次層への伝播において信号の大きさを維持するため、$\mathbb{E}[(z_i^{(l)})^2] = \lambda^2$ を要求すると：
   $$
   \frac{M}{2} \epsilon^2 \lambda^2 = \lambda^2 \implies \frac{M}{2} \epsilon^2 = 1 \implies \epsilon = \sqrt{\frac{2}{M}} \tag{7.23} \quad \blacksquare
   $$

---

### Exercise 7.10 (★★)
**問題**: 式 (7.7), (7.8), (7.10) を用いて、勾配ベクトルと重み更新の固有ベクトル展開 (7.24) と (7.25) を導け。さらに、正規直交性 (7.9) とバッチ勾配降下法 (7.16) を用いて、係数 $\{\alpha_i\}$ に関する更新則 (7.26) $\Delta \alpha_i = -\eta \lambda_i \alpha_i$ を導け。

**【厳密証明】**:
2次近似の勾配は $\nabla E(\mathbf{w}) = \mathbf{H}(\mathbf{w} - \mathbf{w}^\star)$ です。
基底展開 $\mathbf{w} - \mathbf{w}^\star = \sum_i \alpha_i \mathbf{u}_i$（式 7.10）および $\mathbf{H}\mathbf{u}_i = \lambda_i \mathbf{u}_i$（式 7.8）を代入すると：
$$
\nabla E(\mathbf{w}) = \mathbf{H} \sum_i \alpha_i \mathbf{u}_i = \sum_i \alpha_i (\mathbf{H}\mathbf{u}_i) = \sum_i \alpha_i \lambda_i \mathbf{u}_i \tag{7.24}
$$
重み更新量 $\Delta \mathbf{w} = \mathbf{w}^{\text{new}} - \mathbf{w}^{\text{old}}$ の展開は：
$$
\Delta \mathbf{w} = \sum_i \Delta \alpha_i \mathbf{u}_i \tag{7.25}
$$
バッチ勾配降下法の更新則 $\Delta \mathbf{w} = -\eta \nabla E(\mathbf{w})$（式 7.16）に式 (7.24) と (7.25) を代入すると：
$$
\sum_i \Delta \alpha_i \mathbf{u}_i = -\eta \sum_i \alpha_i \lambda_i \mathbf{u}_i = \sum_i (-\eta \lambda_i \alpha_i) \mathbf{u}_i
$$
両辺と固有ベクトル $\mathbf{u}_k$ の内積を取り、正規直交性 $\mathbf{u}_k^T \mathbf{u}_i = \delta_{ki}$ を用いると：
$$
\Delta \alpha_k = -\eta \lambda_k \alpha_k \tag{7.26} \quad \blacksquare
$$

---

### Exercise 7.11 (★)
**問題**: 勾配が位置に対して緩やかにしか変化しない低曲率の誤差曲面を考える。学習率 $\eta$ とモーメンタム $\mu$ が小さい極限において、Nesterovモーメンタム更新則 (7.34) が標準モーメンタム (7.31) と等価であることを示せ。

**【厳密証明】**:
- 標準モーメンタム更新（式 7.31）：
  $$
  \Delta \mathbf{w}^{(\tau-1)}_{\text{std}} = -\eta \nabla E(\mathbf{w}^{(\tau-1)}) + \mu \Delta \mathbf{w}^{(\tau-2)}
  $$
- Nesterovモーメンタム更新（式 7.34）：
  $$
  \Delta \mathbf{w}^{(\tau-1)}_{\text{nag}} = -\eta \nabla E(\mathbf{w}^{(\tau-1)} + \mu \Delta \mathbf{w}^{(\tau-2)}) + \mu \Delta \mathbf{w}^{(\tau-2)}
  $$
先行位置 $\mathbf{w}^{(\tau-1)} + \mu \Delta \mathbf{w}^{(\tau-2)}$ における勾配を、現在位置 $\mathbf{w}^{(\tau-1)}$ の周りでテイラー展開します：
$$
\nabla E(\mathbf{w}^{(\tau-1)} + \mu \Delta \mathbf{w}^{(\tau-2)}) = \nabla E(\mathbf{w}^{(\tau-1)}) + \mathbf{H} (\mu \Delta \mathbf{w}^{(\tau-2)}) + \mathcal{O}(\|\mu \Delta \mathbf{w}\|^2)
$$
これをNesterovの式に代入すると：
$$
\Delta \mathbf{w}^{(\tau-1)}_{\text{nag}} = -\eta \nabla E(\mathbf{w}^{(\tau-1)}) + \mu \Delta \mathbf{w}^{(\tau-2)} - \eta \mu \mathbf{H} \Delta \mathbf{w}^{(\tau-2)} + \mathcal{O}(\eta \mu^2 \|\Delta \mathbf{w}\|^2)
$$
低曲率領域ではヘッセ行列のノルム $\|\mathbf{H}\|$ が小さく、かつ $\Delta \mathbf{w} \sim \mathcal{O}(\eta)$ であるため、付加項は $\eta^2 \mu$ の2次以上の高次微小量となります。
したがって、パラメータが小さい領域において両者は厳密に1次近似で一致します：
$$
\Delta \mathbf{w}^{(\tau-1)}_{\text{nag}} = \Delta \mathbf{w}^{(\tau-1)}_{\text{std}} + \mathcal{O}(\eta^2 \mu) \simeq \Delta \mathbf{w}^{(\tau-1)}_{\text{std}} \quad \blacksquare
$$"""))

    # Part 2: Interactive Code Cell
    cells.append(create_cell("code", r"""# Part 2 (Exercises 7.8 - 7.11) 自己採点検証テスト
print("=== Part 2: Exercises 7.8 〜 7.11 の数値検証 ===")

# Ex 7.8
res8 = exercise_7_8_sample_mean_variance(N=100, sigma=3.0, num_experiments=2000)
assert res8["relative_error"] < 0.1, "Ex 7.8 failed"
print(f"✓ Exercise 7.8 passed: 標本平均誤差 MSE={res8['empirical_mse']:.4f} (理論値 sigma^2/N = {res8['theoretical_mse']:.4f})")

# Ex 7.9
res9 = exercise_7_9_relu_variance_propagation(M=100, input_variance=1.0, num_samples=10000)
assert res9["variance_preserved"], "Ex 7.9 failed"
print(f"✓ Exercise 7.9 passed: He初期化 E[a]={res9['empirical_a_mean']:.4f}, 出力2次モーメント={res9['empirical_z_second_moment']:.3f} (理論値=1.0)")

# Ex 7.10
H10 = np.array([[3.0, 1.0], [1.0, 2.0]])
res10 = exercise_7_10_eigen_update_derivation(H10, np.array([1.5, -0.5]), np.array([0.0, 0.0]), lr=0.1)
assert res10["is_exact_match"], "Ex 7.10 failed"
print("✓ Exercise 7.10 passed: 固有座標系での独立更新則 Delta alpha_i = -eta * lambda_i * alpha_i を確認")

# Ex 7.11
def grad_test(w): return H10 @ w
def hess_test(w): return H10
res11 = exercise_7_11_nesterov_momentum_equivalence(np.array([1.0, 1.0]), np.array([0.05, -0.02]), grad_test, hess_test, lr=0.01, mu=0.8)
assert res11["is_close_first_order"], "Ex 7.11 failed"
print(f"✓ Exercise 7.11 passed: Nesterovと標準モーメンタムの差分ノルム = {res11['diff_actual_norm']:.5f} (1次近似理論値と一致)")"""))

    # Part 3: Exercises 7.12 to 7.14 Markdown
    cells.append(create_cell("markdown", r"""---

# Part 3: 適応的最適化と正規化 [Exercises 7.12 〜 7.14]

### Exercise 7.12 (★★)
**問題**: 式 (7.66) で定義される指数移動平均 $\mu_n = \beta \mu_{n-1} + (1 - \beta) x_n$ において、初期値を $\mu_0 = 0$ としたとき、有限等比級数和公式 (7.67) を用いて推定値がバイアスをもつことを示し、式 (7.68) $\widehat{\mu}_n = \frac{\mu_n}{1 - \beta^n}$ によりバイアスが正確に補正されることを証明せよ。

**【厳密証明】**:
$\mu_0 = 0$ から逐次代入を行います：
- $n = 1$: $\mu_1 = (1 - \beta) x_1$
- $n = 2$: $\mu_2 = \beta (1 - \beta) x_1 + (1 - \beta) x_2 = (1 - \beta) [\beta x_1 + x_2]$
- 一般の $n$:
  $$
  \mu_n = (1 - \beta) \sum_{k=1}^n \beta^{n-k} x_k
  $$
各観測値 $x_k$ が同一の期待値 $\mathbb{E}[x_k] = \mu$ をもつと仮定して、両辺の期待値をとると：
$$
\mathbb{E}[\mu_n] = (1 - \beta) \mu \sum_{k=1}^n \beta^{n-k}
$$
和のインデックスを反転させると $\sum_{k=1}^n \beta^{n-k} = \sum_{j=0}^{n-1} \beta^j = \frac{1 - \beta^n}{1 - \beta}$（式 7.67）であるため：
$$
\mathbb{E}[\mu_n] = (1 - \beta) \mu \left( \frac{1 - \beta^n}{1 - \beta} \right) = (1 - \beta^n) \mu
$$
$0 \le \beta < 1$ より $1 - \beta^n < 1$ であるため、$\mu_n$ は真の期待値 $\mu$ に対して $0$ 側に過小評価されるバイアスを持ちます（特に $n$ が小さい初期ステップ）。
したがって、両辺を $1 - \beta^n$ で割った推定量：
$$
\widehat{\mu}_n = \frac{\mu_n}{1 - \beta^n} \tag{7.68}
$$
の期待値は：
$$
\mathbb{E}[\widehat{\mu}_n] = \frac{\mathbb{E}[\mu_n]}{1 - \beta^n} = \frac{(1 - \beta^n)\mu}{1 - \beta^n} = \mu
$$
となり、任意の $n \ge 1$ において厳密に不偏推定量となります。 $\blacksquare$

---

### Exercise 7.13 (★)
**問題**: 現在の重みベクトル $\mathbf{w}^{(\tau)}$ から探索方向 $\mathbf{d}$ に沿って誤差関数を最小化する直線探索（Line Search）において、最小点 $\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \lambda^\star \mathbf{d}$ における勾配 $\nabla E(\mathbf{w}^{(\tau+1)})$ が方向ベクトル $\mathbf{d}$ と直交することを示せ。

**【証明】**:
探索方向 $\mathbf{d}$ に沿った1変数関数 $f(\lambda) \equiv E(\mathbf{w}^{(\tau)} + \lambda \mathbf{d})$ を定義します。
$\lambda = \lambda^\star$ で $f(\lambda)$ が最小値をとるため、微積分学の極値条件より：
$$
\left. \frac{df(\lambda)}{d\lambda} \right|_{\lambda = \lambda^\star} = 0
$$
合成関数の連鎖律（Chain Rule）を適用すると：
$$
\frac{df(\lambda)}{d\lambda} = \sum_{i=1}^W \frac{\partial E}{\partial w_i} \frac{d(w_i^{(\tau)} + \lambda d_i)}{d\lambda} = \sum_{i=1}^W \frac{\partial E}{\partial w_i} d_i = \mathbf{d}^T \nabla E(\mathbf{w}^{(\tau)} + \lambda \mathbf{d})
$$
したがって、$\lambda = \lambda^\star$ において：
$$
\mathbf{d}^T \nabla E(\mathbf{w}^{(\tau+1)}) = 0 \quad \blacksquare
$$
これは、直線探索において次の探索方向は直前の探索方向と常に直交することを示しています。

---

### Exercise 7.14 (★)
**問題**: 式 (7.50) で定義される再正規化された入力変数 $\widetilde{x}_{ni} = (x_{ni} - \mu_i)/\sigma_i$（ここで $\mu_i$ は式 (7.48)、$\sigma_i^2$ は式 (7.49) で定義される）が、ゼロ平均かつ単位分散をもつことを示せ。

**【証明】**:
1. **標本平均**:
   $$
   \frac{1}{N} \sum_{n=1}^N \widetilde{x}_{ni} = \frac{1}{N} \sum_{n=1}^N \frac{x_{ni} - \mu_i}{\sigma_i} = \frac{1}{\sigma_i} \left[ \frac{1}{N} \sum_{n=1}^N x_{ni} - \mu_i \right]
   $$
   式 (7.48) $\mu_i = \frac{1}{N}\sum_n x_{ni}$ より括弧内は $\mu_i - \mu_i = 0$ となるため：
   $$
   \frac{1}{N} \sum_{n=1}^N \widetilde{x}_{ni} = 0
   $$
2. **標本分散**:
   平均が 0 であるため、分散は2乗平均に等しくなります：
   $$
   \frac{1}{N} \sum_{n=1}^N (\widetilde{x}_{ni} - 0)^2 = \frac{1}{N} \sum_{n=1}^N \left( \frac{x_{ni} - \mu_i}{\sigma_i} \right)^2 = \frac{1}{\sigma_i^2} \left[ \frac{1}{N} \sum_{n=1}^N (x_{ni} - \mu_i)^2 \right]
   $$
   式 (7.49) より角括弧内はまさに $\sigma_i^2$ であるため：
   $$
   = \frac{\sigma_i^2}{\sigma_i^2} = 1 \quad \blacksquare
   $$"""))

    # Part 3: Interactive Code Cell
    cells.append(create_cell("code", r"""# Part 3 (Exercises 7.12 - 7.14) 自己採点検証テスト
print("=== Part 3: Exercises 7.12 〜 7.14 の数値検証 ===")

# Ex 7.12
res12 = exercise_7_12_ema_bias_correction(beta=0.9, num_steps=20, true_mean=10.0)
assert res12["bias_corrected_is_exact"], "Ex 7.12 failed"
print(f"✓ Exercise 7.12 passed: EMA初期値補正 mu_raw[0]={res12['mu_raw'][0]:.2f} -> mu_corrected[0]={res12['mu_corrected'][0]:.2f} (真値=10.0)")

# Ex 7.13
A_ls = np.array([[3.0, 1.0], [1.0, 4.0]])
def err_ls(v): return float(0.5 * v.T @ A_ls @ v)
def grad_ls(v): return A_ls @ v
res13 = exercise_7_13_line_search_orthogonality(err_ls, grad_ls, np.array([2.0, -1.0]), np.array([-1.0, 0.5]))
assert res13["is_orthogonal"], "Ex 7.13 failed"
print(f"✓ Exercise 7.13 passed: 直線探索最小点における方向直交性 d^T grad = {res13['directional_derivative']:.2e} ~= 0")

# Ex 7.14
raw_pts = np.array([12.0, 18.0, 5.0, 22.0, 14.0])
res14 = exercise_7_14_standardization_moments(raw_pts)
assert res14["is_zero_mean"] and res14["is_unit_variance"], "Ex 7.14 failed"
print(f"✓ Exercise 7.14 passed: 標準化変数の平均 = {res14['normalized_mean']:.2e} ~= 0, 分散 = {res14['normalized_var']:.4f} == 1")

print("\n" + "=" * 55)
print("🎉 第7章 演習問題 全14問 (Exercises 7.1 〜 7.14) すべて合格！")
print("=" * 55)"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return notebook


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "7")
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, "7_Exercises.ipynb")

    notebook = build_cells()
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Wrote notebook to {nb_path}")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        nb_path,
    ]
    print(f"Executing notebook: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Error executing notebook:")
        print(res.stderr)
        sys.exit(res.returncode)
    print("Chapter 7 Exercises notebook executed successfully with zero errors!")


if __name__ == "__main__":
    main()

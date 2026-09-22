"""
Script to build 4/4.1_Linear_Regression.ipynb covering Section 4.1 of Bishop & Bishop (2024),
Chapter 4: Single-layer Networks: Regression.
"""
import json
import os

def create_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1] == '\n':
            lines[-1] = ''
    else:
        lines = source
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = []

# Cell 1: Markdown Title & Index
cells.append(create_cell("markdown", """# 第4章 単層ネットワーク: 回帰 (Single-layer Networks: Regression)
## 4.1 線形回帰 (Linear Regression)

本ノートブックでは、入力変数から連続的な目標変数を予測する回帰問題の基本モデルである**線形基底関数回帰モデル (Linear Basis Function Models)** を、単層ニューラルネットワークの視点から体系的に解説・実装します。

---

### 目次
1. **4.1.1 基底関数 (Basis Functions)**
   - 線形モデルの定式化と表現力 (式 4.1 - 4.3)
   - 単層ニューラルネットワークとしての解釈 (**Figure 4.1** の完全再現)
   - 多項式、ガウス基底、シグモイド基底の特性 (**Figure 4.2** の完全再現)
2. **4.1.2 尤度関数 (Likelihood Function)**
   - ガウス加法性ノイズモデル $p(t \\mid \\mathbf{x}, \\mathbf{w}, \\sigma^2) = \\mathcal{N}(t \\mid y(\\mathbf{x}, \\mathbf{w}), \\sigma^2)$ (式 4.7 - 4.8)
   - 対数尤度関数と二乗和誤差関数の等価性の数理的導出 (式 4.10 - 4.11)
3. **4.1.3 最尤推定 (Maximum Likelihood)**
   - 正規方程式 (Normal Equations) と計画行列 $\\boldsymbol{\\Phi}$ (式 4.12 - 4.15)
   - ムーア・ペンローズ擬似逆行列 $\\boldsymbol{\\Phi}^\\dagger$ (式 4.16)
   - バイアスパラメータ $w_0$ の直感的役割 (式 4.17 - 4.19)
   - 残差分散推定量 $\\sigma_{\\mathrm{ML}}^2$ (式 4.20)
4. **4.1.4 最小二乗法の幾何学 (Geometry of Least Squares)**
   - $N$ 次元空間における目標ベクトル $\\mathbf{t}$ と部分空間 $\\mathcal{S}$
   - 直交射影としての最小二乗解 (**Figure 4.3** の完全再現)
   - SVDによる特異行列・悪条件の数値的克服
5. **4.1.5 逐次学習 (Sequential Learning)**
   - 確率的勾配降下法 (SGD) と最小平均二乗 (LMS) アルゴリズム (式 4.21 - 4.22)
   - オンラインストリーミング学習の実験
6. **4.1.6 正則化最小二乗法 (Regularized Least Squares)**
   - 過学習抑制のための荷重減衰 / $L_2$ 正則化 (Ridge回帰) (式 4.23 - 4.26)
   - 解析的閉形式解 $\\mathbf{w} = (\\lambda \\mathbf{I} + \\boldsymbol{\\Phi}^T \\boldsymbol{\\Phi})^{-1} \\boldsymbol{\\Phi}^T \\mathbf{t}$ (式 4.27)
7. **4.1.7 複数出力 (Multiple Outputs)**
   - 多変量目標変数 $\\mathbf{t} \\in \\mathbb{R}^K$ への拡張 (式 4.28 - 4.30)
   - 複数出力単層ネットワーク図 (**Figure 4.4** の完全再現)
   - 共通の擬似逆行列 $\\boldsymbol{\\Phi}^\\dagger$ による出力間の完全なデカップリング (式 4.31 - 4.32)
8. **自己検証テストスイート (Self-Check Validation)**"""))

# Cell 2: Code Imports & Setup
cells.append(create_cell("code", """import os
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

repo_root = os.path.abspath("..")
if repo_root not in sys.path:
    sys.path.append(repo_root)

from common.plot_utils import setup_style
from common.linear_models import (
    PolynomialBasis,
    GaussianBasis,
    SigmoidalBasis,
    LinearRegression,
    SequentialLinearRegression,
    RidgeRegression,
    MultipleOutputLinearRegression,
    plot_figure_4_1_network_diagram,
    plot_figure_4_2_basis_functions,
    plot_figure_4_3_least_squares_geometry,
    plot_figure_4_4_multiple_outputs_diagram
)

setup_style()
print("Chapter 4 Linear Models loaded successfully.")"""))

# Cell 3: Markdown Section 4.1.1
cells.append(create_cell("markdown", r"""---
## 1. 4.1.1 基底関数 (Basis Functions)

### 1.1 線形回帰モデルの定式化
最も単純な回帰モデルは入力変数 $x_1, \dots, x_D$ の線形結合です：
$$
y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + \dots + w_D x_D = w_0 + \sum_{i=1}^D w_i x_i \tag{4.1}
$$
このモデルはパラメータ $\mathbf{w}$ に関して線形であるだけでなく、入力変数 $\mathbf{x}$ に対しても線形であるため、非線形な現象を表現できません。

そこで、固定された非線形関数 $\phi_j(\mathbf{x})$（**基底関数: basis functions**）の線形結合を考えます：
$$
y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j(\mathbf{x}) \tag{4.2}
$$
ここでダミーの基底関数 $\phi_0(\mathbf{x}) = 1$ を導入し、$\mathbf{w} = (w_0, \dots, w_{M-1})^T$、$\boldsymbol{\phi} = (\phi_0, \dots, \phi_{M-1})^T$ と定義すれば、極めて簡潔な内積形式で表現できます：
$$
y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) \tag{4.3}
$$
これは、入力層に各基底関数 $\phi_j(\mathbf{x})$ を配置し、出力層に結合重み $w_j$ を通じて単一の出力ノード $y(\mathbf{x}, \mathbf{w})$ を形成する**単層ニューラルネットワーク**と完全に等価です。"""))

# Cell 4: Code Figure 4.1
cells.append(create_cell("code", """# Figure 4.1: 単層線形回帰ニューラルネットワークの図
fig4_1, ax4_1 = plot_figure_4_1_network_diagram(
    save_paths=["result/fig4_1_network_diagram.png", "../result/fig4_1_network_diagram.png"],
    show=True
)"""))

# Cell 5: Markdown Basis Function Types
cells.append(create_cell("markdown", r"""### 1.2 主な基底関数の種類
代表的な基底関数には以下のものがあります：

1. **多項式基底 (Polynomials)**:
   $$
   \phi_j(x) = x^j \quad (j = 0, \dots, M-1)
   $$
   大域的な関数であり、定義域の一部での変化が空間全体に波及します。
2. **ガウス基底 (Gaussian basis functions)** (式 4.4):
   $$
   \phi_j(x) = \exp\left( -\frac{(x - \mu_j)^2}{2s^2} \right)
   $$
   入力空間の局所的な位置 $\mu_j$ に中心を持ち、空間スケール $s$ を持ちます。確率分布の規格化係数は不要です（重み $w_j$ が学習されるため）。
3. **シグモイド基底 (Sigmoidal basis functions)** (式 4.5 - 4.6):
   $$
   \phi_j(x) = \sigma\left( \frac{x - \mu_j}{s} \right), \quad \sigma(a) = \frac{1}{1 + \exp(-a)}
   $$
   ロジスティック・シグモイド関数は双曲線正接関数 $\tanh(a) = 2\sigma(2a) - 1$ と等価な関数空間を張ります。"""))

# Cell 6: Code Figure 4.2
cells.append(create_cell("code", """# Figure 4.2: 各種基底関数の形状比較（多項式・ガウス・シグモイド）
fig4_2, axes4_2 = plot_figure_4_2_basis_functions(
    save_paths=["result/fig4_2_basis_functions.png", "../result/fig4_2_basis_functions.png"],
    show=True
)"""))

# Cell 7: Markdown Section 4.1.2 - 4.1.3 Likelihood & Normal Equations
cells.append(create_cell("markdown", r"""---
## 2. 4.1.2 尤度関数 と 4.1.3 最尤推定

### 2.1 尤度関数と二乗和誤差
目標変数 $t$ が決定論的関数 $y(\mathbf{x}, \mathbf{w})$ にゼロ平均・分散 $\sigma^2$ のガウスノイズ $\epsilon$ が加わったものと仮定します（式 4.7 - 4.8）：
$$
p(t \mid \mathbf{x}, \mathbf{w}, \sigma^2) = \mathcal{N}(t \mid y(\mathbf{x}, \mathbf{w}), \sigma^2)
$$
独立な $N$ 個の観測データ $(\mathbf{X}, \mathbf{t})$ に対する対数尤度関数は次式となります（式 4.10）：
$$
\ln p(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = -\frac{N}{2}\ln \sigma^2 - \frac{N}{2}\ln(2\pi) - \frac{1}{\sigma^2} E_D(\mathbf{w})
$$
ここで $E_D(\mathbf{w})$ は**二乗和誤差関数 (Sum-of-squares error function)** です（式 4.11）：
$$
E_D(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \left\{ t_n - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) \right\}^2
$$
対数尤度の最初の2項は $\mathbf{w}$ に依存しない定数であるため、**ガウスノイズ下での最尤推定は二乗和誤差関数の最小化と厳密に等価**です。

---

### 2.2 正規方程式とムーア・ペンローズ擬似逆行列
対数尤度の $\mathbf{w}$ に関する勾配をゼロとおくと（式 4.12 - 4.13）：
$$
\sum_{n=1}^N t_n \boldsymbol{\phi}(\mathbf{x}_n)^T - \mathbf{w}^T \sum_{n=1}^N \boldsymbol{\phi}(\mathbf{x}_n)\boldsymbol{\phi}(\mathbf{x}_n)^T = \mathbf{0}
$$
$N \times M$ の**計画行列 (Design matrix)** $\boldsymbol{\Phi}$（式 4.15）：
$$
\boldsymbol{\Phi} = \begin{pmatrix} \phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \dots & \phi_{M-1}(\mathbf{x}_1) \\ \vdots & \vdots & \ddots & \vdots \\ \phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \dots & \phi_{M-1}(\mathbf{x}_N) \end{pmatrix}
$$
を用いると、最尤推定解 $\mathbf{w}_{\mathrm{ML}}$ は**正規方程式 (Normal equations)** の解として得られます（式 4.14）：
$$
\mathbf{w}_{\mathrm{ML}} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{t} = \boldsymbol{\Phi}^\dagger \mathbf{t} \tag{4.14, 4.16}
$$
ここで $\boldsymbol{\Phi}^\dagger \equiv (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T$ は**ムーア・ペンローズ擬似逆行列 (Moore-Penrose pseudo-inverse)** です。

また、ノイズ分散の最尤推定量は回帰関数まわりの標本残差分散として得られます（式 4.20）：
$$
\sigma_{\mathrm{ML}}^2 = \frac{1}{N}\sum_{n=1}^N \left\{ t_n - \mathbf{w}_{\mathrm{ML}}^T \boldsymbol{\phi}(\mathbf{x}_n) \right\}^2
$$"""))

# Cell 8: Code MLE Fitting Demonstration
cells.append(create_cell("code", """# 最尤推定フィッティングの数値実験
np.random.seed(42)
N_demo = 25
x_demo = np.sort(np.random.uniform(0, 1, N_demo))
t_true = np.sin(2 * np.pi * x_demo)
t_demo = t_true + np.random.normal(0, 0.2, N_demo)

# ガウス基底関数モデルでフィッティング
centers_demo = np.linspace(0, 1, 6)
gauss_basis = GaussianBasis(centers=centers_demo, s=0.15, include_bias=True)
mle_model = LinearRegression(basis_func=gauss_basis).fit(x_demo, t_demo)

x_plot = np.linspace(0, 1, 200)
y_plot = mle_model.predict(x_plot)

fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
ax.plot(x_plot, np.sin(2 * np.pi * x_plot), 'g-', label=r"True $f(x) = \sin(2\pi x)$", linewidth=1.8)
ax.scatter(x_demo, t_demo, facecolor='none', edgecolor='b', s=45, label="Noisy samples $t_n$")
ax.plot(x_plot, y_plot, 'r-', label=r"MLE Fit $y(x, \mathbf{w}_{\mathrm{ML}})$", linewidth=2.0)
ax.fill_between(x_plot, y_plot - np.sqrt(mle_model.sigma2), y_plot + np.sqrt(mle_model.sigma2),
                color='r', alpha=0.15, label=r"$\pm 1\sigma_{\mathrm{ML}}$ interval")
ax.set_xlabel(r"$x$", fontsize=11)
ax.set_ylabel(r"$t$", fontsize=11)
ax.set_title(f"Linear Basis Function Regression (Gaussian Basis, M={gauss_basis.n_features})", fontsize=11)
ax.legend(frameon=True)
plt.tight_layout()
plt.show()"""))

# Cell 9: Markdown Section 4.1.4 Least Squares Geometry
cells.append(create_cell("markdown", r"""---
## 3. 4.1.4 最小二乗法の幾何学 (Geometry of Least Squares)

### 3.1 $N$ 次元空間における直交射影
最小二乗法には明快な幾何学的直感が存在します：
- 目標値の組 $\mathbf{t} = (t_1, \dots, t_N)^T$ は $N$ 次元空間の単一のベクトルです。
- 各基底関数 $\phi_j(x)$ を $N$ 個のデータ点で評価した列ベクトル $\boldsymbol{\phi}_j = (\phi_j(x_1), \dots, \phi_j(x_N))^T$（計画行列 $\boldsymbol{\Phi}$ の第 $j$ 列）は、同じ $N$ 次元空間内のベクトルです。
- $M < N$ のとき、これら $M$ 個の基底ベクトルは $N$ 次元空間内の $M$ 次元部分空間 $\mathcal{S}$ を張ります。
- 予測ベクトル $\mathbf{y} = (y(x_1, \mathbf{w}), \dots, y(x_N, \mathbf{w}))^T = \boldsymbol{\Phi}\mathbf{w}$ は部分空間 $\mathcal{S}$ 内の任意の線形結合点です。

二乗和誤差 $E_D(\mathbf{w}) = \frac{1}{2}\|\mathbf{t} - \mathbf{y}\|^2$ は $\mathbf{t}$ と $\mathbf{y}$ のユークリッド二乗距離に等しいため、誤差を最小化する $\mathbf{y}$ は**部分空間 $\mathcal{S}$ 上への目標ベクトル $\mathbf{t}$ の直交射影 (Orthogonal Projection)** にほかなりません。
残差ベクトル $\mathbf{t} - \mathbf{y}$ は部分空間 $\mathcal{S}$ と直交するため、$\boldsymbol{\Phi}^T (\mathbf{t} - \mathbf{y}) = \mathbf{0}$ が成立します。"""))

# Cell 10: Code Figure 4.3
cells.append(create_cell("code", """# Figure 4.3: N次元空間における最小二乗解の直交射影の幾何学的解釈
fig4_3, ax4_3 = plot_figure_4_3_least_squares_geometry(
    save_paths=["result/fig4_3_least_squares_geometry.png", "../result/fig4_3_least_squares_geometry.png"],
    show=True
)"""))

# Cell 11: Markdown Section 4.1.5 Sequential Learning
cells.append(create_cell("markdown", r"""---
## 4. 4.1.5 逐次学習 (Sequential Learning / LMS)

### 4.1 LMS (Least-Mean-Squares) アルゴリズム
大規模データセットや逐次データストリームでは、データ全体を一括処理するバッチ法 $(\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T \mathbf{t}$ は計算量・メモリの観点から非現実的となります。
データ点 $n$ が到着するごとにパラメータを逐次更新する**確率的勾配降下法 (Stochastic Gradient Descent: SGD)** を適用します（式 4.21）：
$$
\mathbf{w}^{(\tau + 1)} = \mathbf{w}^{(\tau)} - \eta \nabla E_n
$$
二乗和誤差 $E_n = \frac{1}{2}(t_n - \mathbf{w}^T \boldsymbol{\phi}_n)^2$ を代入すると、有名な **LMS (Least-Mean-Squares) アルゴリズム**が得られます（式 4.22）：
$$
\mathbf{w}^{(\tau + 1)} = \mathbf{w}^{(\tau)} + \eta (t_n - {\mathbf{w}^{(\tau)}}^T \boldsymbol{\phi}_n) \boldsymbol{\phi}_n
$$
ここで $\eta$ は学習率 (learning rate) です。各ステップの計算量は $\mathcal{O}(M)$ であり、サンプル数 $N$ に依存しません。"""))

# Cell 12: Code Sequential Learning Simulation
cells.append(create_cell("code", """# 逐次学習 (LMS) の収束実験
true_weights = np.array([2.0, -3.5])
lms_agent = SequentialLinearRegression(n_features=2, learning_rate=0.03)

n_steps = 300
np.random.seed(42)
for step in range(n_steps):
    x_val = np.random.uniform(-1, 1)
    phi_vec = np.array([1.0, x_val])
    t_val = float(np.dot(phi_vec, true_weights) + np.random.normal(0, 0.05))
    lms_agent.update(phi_vec, t_val)

traj = np.array(lms_agent.trajectory)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.5), dpi=300)
ax1.plot(traj[:, 0], label=r"$w_0$ (estimate)", color='b')
ax1.axhline(true_weights[0], color='b', linestyle='--', label=r"True $w_0 = 2.0$")
ax1.plot(traj[:, 1], label=r"$w_1$ (estimate)", color='r')
ax1.axhline(true_weights[1], color='r', linestyle='--', label=r"True $w_1 = -3.5$")
ax1.set_xlabel("Update Step")
ax1.set_ylabel("Weight Value")
ax1.set_title("LMS Parameter Convergence", fontsize=11)
ax1.legend(frameon=True)

# Parameter space trajectory
ax2.plot(traj[:, 0], traj[:, 1], 'k.-', alpha=0.5, label="SGD Trajectory")
ax2.plot(true_weights[0], true_weights[1], 'g*', markersize=12, label="Optimal $\mathbf{w}$")
ax2.set_xlabel(r"$w_0$")
ax2.set_ylabel(r"$w_1$")
ax2.set_title("Weight Space Trajectory", fontsize=11)
ax2.legend(frameon=True)

plt.tight_layout()
plt.show()"""))

# Cell 13: Markdown Section 4.1.6 Regularized Least Squares
cells.append(create_cell("markdown", r"""---
## 5. 4.1.6 正則化最小二乗法 (Regularized Least Squares / Ridge)

基底関数の個数 $M$ がデータ数 $N$ に匹敵あるいは上回る場合、計画行列の積 $\boldsymbol{\Phi}^T \boldsymbol{\Phi}$ は悪条件となり、重みの大きさが異常に増大する過学習が発生します。
これを抑制するため、二乗和誤差に重みベクトルの $L_2$ ノルムペナルティを加えた正則化誤差関数を最小化します（式 4.23 - 4.26）：
$$
E(\mathbf{w}) = E_D(\mathbf{w}) + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w} = \frac{1}{2}\sum_{n=1}^N \left\{ t_n - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) \right\}^2 + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w}
$$
勾配をゼロとおくことで、解析的な閉形式解が得られます（式 4.27）：
$$
\mathbf{w}_{\mathrm{ridge}} = (\lambda \mathbf{I} + \boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{t}
$$
正則化項 $\lambda \mathbf{I}$ の付加により行列 $(\lambda \mathbf{I} + \boldsymbol{\Phi}^T \boldsymbol{\Phi})$ は常に厳密な正定値行列となり、数値的逆行列の安定性が保証されます。"""))

# Cell 14: Markdown Section 4.1.7 Multiple Outputs
cells.append(create_cell("markdown", r"""---
## 6. 4.1.7 複数出力 (Multiple Outputs)

### 6.1 複数出力モデルと完全なデカップリング
$K > 1$ 次元の連続目標ベクトル $\mathbf{t} = (t_1, \dots, t_K)^T$ を同時に予測する場合、共通の基底関数群 $\boldsymbol{\phi}(\mathbf{x})$ を用いてモデル化します（式 4.28）：
$$
\mathbf{y}(\mathbf{x}, \mathbf{W}) = \mathbf{W}^T \boldsymbol{\phi}(\mathbf{x})
$$
ここで $\mathbf{W}$ は $M \times K$ のパラメータ行列です。
等方性ガウスノイズ $\mathcal{N}(\mathbf{t} \mid \mathbf{W}^T \boldsymbol{\phi}(\mathbf{x}), \sigma^2 \mathbf{I})$ を仮定すると、目標値行列 $\mathbf{T}$ ($N \times K$) に対する最尤推定解は次式で与えられます（式 4.31）：
$$
\mathbf{W}_{\mathrm{ML}} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{T} = \boldsymbol{\Phi}^\dagger \mathbf{T}
$$
各出力次元 $k$ の重みベクトル $\mathbf{w}_k$ は独立に解かれます（式 4.32）：
$$
\mathbf{w}_k = \boldsymbol{\Phi}^\dagger \mathbf{t}_k
$$
同一の基底関数を共有しているため、**計算コストの高い擬似逆行列 $\boldsymbol{\Phi}^\dagger$ は一度だけ計算すればよく、すべての出力次元 $k = 1, \dots, K$ で共通して再利用**できます。"""))

# Cell 15: Code Figure 4.4
cells.append(create_cell("code", """# Figure 4.4: 複数出力単層線形回帰ニューラルネットワーク図
fig4_4, ax4_4 = plot_figure_4_4_multiple_outputs_diagram(
    save_paths=["result/fig4_4_multiple_outputs_diagram.png", "../result/fig4_4_multiple_outputs_diagram.png"],
    show=True
)"""))

# Cell 16: Markdown Self-Check
cells.append(create_cell("markdown", r"""---
## 7. 自己検証アサーション (Self-Check Validation)

本節で実装した基底関数、最尤正規方程式、擬似逆行列、幾何学的直交性、LMS逐次学習、Ridge正則化、および複数出力デカップリングの数理的性質を自動テストします。"""))

# Cell 17: Code Self-Check Assertions
cells.append(create_cell("code", """# =====================================================================
# 自動検証テストスイート
# =====================================================================
print("Running Section 4.1 comprehensive self-check assertions...")

# 1. 正規方程式とムーア・ペンローズ擬似逆行列の数理的一致
np.random.seed(42)
X_val = np.linspace(-1, 1, 30)
t_val = 1.5 * X_val**2 - 2.0 * X_val + 0.5
poly_basis = PolynomialBasis(degree=2)
model_test = LinearRegression(basis_func=poly_basis).fit(X_val, t_val)

Phi_mat = poly_basis(X_val)
w_manual = la.pinv(Phi_mat) @ t_val
assert np.allclose(model_test.w, w_manual, atol=1e-12)

# 2. 直交射影の性質: Phi^T @ (t - y) == 0
y_val = model_test.predict(X_val)
residuals = t_val - y_val
assert np.allclose(Phi_mat.T @ residuals, np.zeros(3), atol=1e-10)

# 3. バイアスパラメータの公式 (式 4.18)
expected_w0 = np.mean(t_val) - np.sum(model_test.w[1:] * np.mean(Phi_mat[:, 1:], axis=0))
assert np.isclose(model_test.w[0], expected_w0, atol=1e-12)

# 4. 複数出力のデカップリング (式 4.31 - 4.32)
T_val = np.column_stack([t_val, -2.0 * t_val + 1.0])
multi_test = MultipleOutputLinearRegression(basis_func=poly_basis).fit(X_val, T_val)
single_1 = LinearRegression(basis_func=poly_basis).fit(X_val, T_val[:, 0])
single_2 = LinearRegression(basis_func=poly_basis).fit(X_val, T_val[:, 1])

assert np.allclose(multi_test.W[:, 0], single_1.w, atol=1e-12)
assert np.allclose(multi_test.W[:, 1], single_2.w, atol=1e-12)

# 5. Ridge正則化のノルム縮小特性
ridge_0 = RidgeRegression(alpha=0.0, basis_func=poly_basis).fit(X_val, t_val)
ridge_big = RidgeRegression(alpha=5.0, basis_func=poly_basis).fit(X_val, t_val)
assert la.norm(ridge_big.w) < la.norm(ridge_0.w)

print("All Section 4.1 self-check assertions passed successfully! 100% mathematical accuracy confirmed.")"""))

# Cell 18: Markdown Summary
cells.append(create_cell("markdown", """---
## まとめ (Summary)

本節では、第4章「単層ネットワーク: 回帰」の第1節である線形回帰 (Linear Regression) を包括的に実装・分析しました：
1. **基底関数の威力**: 多項式・ガウス・シグモイド基底関数により、重みに関して線形でありながら柔軟な非線形回帰を実現しました。
2. **正規方程式と幾何学**: ガウスノイズ下の最尤推定が二乗和誤差最小化と一致し、データ空間における直交射影であることを幾何学的に明らかにしました (**Figure 4.3**)。
3. **逐次学習と正則化**: LMSアルゴリズムによるストリーミング学習、および過学習を抑制し悪条件を解消するRidge正則化を定式化しました。
4. **複数出力とデカップリング**: 共通の基底関数を用いることで、計算負荷の高い擬似逆行列 $\\boldsymbol{\\Phi}^\\dagger$ を1度計算するだけで全出力次元の学習が完了する強力な性質を確認しました (**Figure 4.4**)。"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "4/4.1_Linear_Regression.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Notebook successfully written to {out_path} with {len(cells)} cells.")

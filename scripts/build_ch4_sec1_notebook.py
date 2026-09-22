import nbformat as nbf
import os
import numpy as np

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.12"
        }
    }

    # Cell 0: Markdown - Title and Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# 第4章 単層ネットワーク: 回帰 (Single-layer Networks: Regression)
## 4.1 線形回帰 (Linear Regression)

### 本節の目的と概要
本ノートブックでは、Christopher M. Bishop & Hugh Bishop による『*Deep Learning: Foundations and Concepts*』(2024年刊) の **Chapter 4: Single-layer Networks: Regression** のうち、**Section 4.1: Linear Regression** を完全かつ厳密に解説・実装・検証します。

線形回帰は機械学習と深層学習の基礎であり、**学習可能パラメータを1層だけ持つ最も単純なニューラルネットワーク**として自然に定式化されます。本節では以下の7つの小節を網羅します：

1. **4.1.1 基底関数 (Basis functions)**: 入力の固定非線形変換 $\\boldsymbol{\\phi}(\\mathbf{x})$ によるモデル拡張、多項式・ガウス・シグモイド基底、ニューラルネットワーク表現 (Figure 4.1, 4.2)
2. **4.1.2 尤度関数 (Likelihood function)**: ガウスノイズモデル $p(t|\\mathbf{x}, \\mathbf{w}, \\sigma^2) = \\mathcal{N}(t|\\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}), \\sigma^2)$ と二乗和誤差関数の等価性
3. **4.1.3 最尤推定 (Maximum likelihood)**: 正規方程式 (Normal equations)、ムーア・ペンローズ擬似逆行列 $\\boldsymbol{\\Phi}^{\\dagger}$、バイアスパラメータ $w_0$ の役割、残差分散 $\\sigma_{\\mathrm{ML}}^2$
4. **4.1.4 最小二乗法の幾何学 (Geometry of least squares)**: $N$次元目的変数空間における部分空間 $\\mathcal{S}$ への直交射影、射影行列 $\\mathbf{P}$ の性質 (Figure 4.3)
5. **4.1.5 逐次学習 (Sequential learning)**: 確率的勾配降下法 (SGD) と最小平均二乗法 (LMS / Widrow-Hoff アルゴリズム)
6. **4.1.6 正則化最小二乗法 (Regularized least squares)**: リッジ回帰 ($L_2$ 正則化)、パラメータ縮小効果 (Shrinkage)、数値的安定性の保証
7. **4.1.7 複数出力 (Multiple outputs)**: 多変量目的変数 $\\mathbf{t} \\in \\mathbb{R}^K$、共有基底関数による多出力ネットワーク図 (Figure 4.4)、各出力への完全な分離性 (Decoupling)"""))

    # Cell 1: Code - Imports and Setup
    nb.cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from IPython.display import Image, display

# プロジェクトルートをパスに追加
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
os.makedirs('result', exist_ok=True)
os.makedirs('../result', exist_ok=True)
print("環境セットアップ完了: common.linear_models を読み込みました。")"""))

    # Cell 2: Markdown - 4.1.1
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4.1.1 基底関数 (Basis Functions)

回帰問題の最も単純な定式化は、入力変数の線形結合です：
$$
y(\\mathbf{x}, \\mathbf{w}) = w_0 + w_1 x_1 + \\dots + w_D x_D = w_0 + \\mathbf{w}_{1:D}^{\\mathrm{T}}\\mathbf{x} \\tag{4.1}
$$
このモデルはパラメータ $\\mathbf{w}$ に関して線形であるだけでなく、入力変数 $\\mathbf{x}$ に対しても線形であるため、表現力に強い制限があります。

### 基底関数によるモデルの拡張
この制限を克服するため、入力変数 $\\mathbf{x}$ の固定された非線形変換 $\\{\\phi_j(\\mathbf{x})\\}$ の線形結合を考えます：
$$
y(\\mathbf{x}, \\mathbf{w}) = w_0 + \\sum_{j=1}^{M-1} w_j \\phi_j(\\mathbf{x}) \\tag{4.2}
$$
ここで $\\phi_j(\\mathbf{x})$ は**基底関数 (basis function)** と呼ばれ、パラメータの総数は $M$ 個です。
ダミー基底関数 $\\phi_0(\\mathbf{x}) = 1$ を導入し、$\\mathbf{w} = (w_0, \\dots, w_{M-1})^{\\mathrm{T}}$、$\\boldsymbol{\\phi} = (\\phi_0, \\dots, \\phi_{M-1})^{\\mathrm{T}}$ と定義すれば、内積形式で簡潔に表せます：
$$
y(\\mathbf{x}, \\mathbf{w}) = \\sum_{j=0}^{M-1} w_j \\phi_j(\\mathbf{x}) = \\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}) \\tag{4.3}
$$

### 単層ニューラルネットワークとしての解釈 (Figure 4.1)
式 (4.3) は、入力層に各基底関数 $\\phi_j(\\mathbf{x})$ のノード（黒塗りノードはバイアス $\\phi_0=1$）を持ち、出力層にノード $y(\\mathbf{x}, \\mathbf{w})$ を持ち、両者が結合重み $w_j$ で結ばれた**単層のニューラルネットワーク**とみなすことができます。

### 代表的な基底関数 (Figure 4.2)
1. **多項式基底 (Polynomial basis)**: $\\phi_j(x) = x^j$ （大域的な性質を持ち、一部の変化が全体に影響）
2. **ガウス基底 (Gaussian basis)**:
   $$
   \\phi_j(x) = \\exp\\left\\{ -\\frac{(x - \\mu_j)^2}{2s^2} \\right\\} \\tag{4.4}
   $$
   中心 $\\mu_j$ と空間スケール $s$ を持つ局所的基底関数（確率密度関数としての正規化係数は不要）。
3. **シグモイド基底 (Sigmoidal basis)**:
   $$
   \\phi_j(x) = \\sigma\\left(\\frac{x - \\mu_j}{s}\\right), \\quad \\sigma(a) = \\frac{1}{1 + e^{-a}} \\tag{4.5, 4.6}
   $$
   なお、双曲線正接関数 $\\tanh(a) = 2\\sigma(2a) - 1$ も線形結合においてシグモイド関数と等価な表現能力を持ちます。"""))

    # Cell 3: Code - Plot Figure 4.1 and 4.2
    nb.cells.append(nbf.v4.new_code_cell("""# Figure 4.1: 単層線形回帰ニューラルネットワーク図の再現
fig4_1, ax4_1 = plot_figure_4_1_network_diagram(
    save_paths=['result/fig4_1_network_diagram.png', '../result/fig4_1_network_diagram.png']
)
plt.show()

# Figure 4.2: 代表的な基底関数 (多項式、ガウス基底、シグモイド基底) のプロット
fig4_2, axes4_2 = plot_figure_4_2_basis_functions(
    save_paths=['result/fig4_2_basis_functions.png', '../result/fig4_2_basis_functions.png']
)
plt.show()

# 単体テスト的検証: 基底関数の評価値の厳密性
poly = PolynomialBasis(degree=3)
gauss = GaussianBasis(centers=[-0.5, 0.5], s=0.25, include_bias=True)
sig = SigmoidalBasis(centers=[-0.5, 0.5], s=0.25, include_bias=True)

x_test = np.array([0.0])
assert np.allclose(poly(x_test), [[1.0, 0.0, 0.0, 0.0]])
assert np.isclose(gauss(x_test)[0, 0], 1.0) # bias
assert np.isclose(gauss(x_test)[0, 1], np.exp(-0.5 * (-0.5 / 0.25)**2))
assert np.isclose(sig(x_test)[0, 0], 1.0)   # bias
assert np.isclose(sig(x_test)[0, 1], 1.0 / (1.0 + np.exp(- (0.0 - (-0.5)) / 0.25)))
print("Assertion Passed: 基底関数の計算が教科書定義と完全一致しています。")"""))

    # Cell 4: Markdown - 4.1.2
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4.1.2 尤度関数 (Likelihood Function)

目的変数 $t$ が、決定論的関数 $y(\\mathbf{x}, \\mathbf{w})$ に平均 0、分散 $\\sigma^2$ の加法的ガウスノイズ $\\epsilon$ が加わったものとして生成されると仮定します：
$$
t = y(\\mathbf{x}, \\mathbf{w}) + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\sigma^2) \\tag{4.7}
$$
したがって、入力 $\\mathbf{x}$ が与えられたときの目標値 $t$ の条件付き確率分布は以下となります：
$$
p(t|\\mathbf{x}, \\mathbf{w}, \\sigma^2) = \\mathcal{N}(t | y(\\mathbf{x}, \\mathbf{w}), \\sigma^2) \\tag{4.8}
$$

### データセットに対する尤度関数
独立同分布 (i.i.d.) に生成された $N$ 個の観測データ $\\mathbf{X} = \\{\\mathbf{x}_1, \\dots, \\mathbf{x}_N\\}$ および目標値ベクトル $\\mathbf{t} = (t_1, \\dots, t_N)^{\\mathrm{T}}$ に対する尤度関数は積として表されます：
$$
p(\\mathbf{t}|\\mathbf{X}, \\mathbf{w}, \\sigma^2) = \\prod_{n=1}^N \\mathcal{N}(t_n | \\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n), \\sigma^2) \\tag{4.9}
$$
対数尤度関数を取ると、1変量ガウス分布の公式から：
$$
\\ln p(\\mathbf{t}|\\mathbf{X}, \\mathbf{w}, \\sigma^2) = -\\frac{N}{2}\\ln\\sigma^2 - \\frac{N}{2}\\ln(2\\pi) - \\frac{1}{\\sigma^2} E_D(\\mathbf{w}) \\tag{4.10}
$$
ここで、二乗和誤差関数 $E_D(\\mathbf{w})$ は次のように定義されます：
$$
E_D(\\mathbf{w}) = \\frac{1}{2}\\sum_{n=1}^N \\{ t_n - \\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n) \\}^2 \\tag{4.11}
$$
**重要な結論**:
式 (4.10) の初めの2項は $\\mathbf{w}$ に依存しないため、**ガウスノイズを仮定した対数尤度関数の最大化は、二乗和誤差関数 $E_D(\\mathbf{w})$ の最小化と完全に等価**です。"""))

    # Cell 5: Code - Verification of Likelihood
    nb.cells.append(nbf.v4.new_code_cell("""# 尤度関数と二乗和誤差の等価性の数値的確認
np.random.seed(42)
N_sample = 25
X_synth = np.sort(np.random.uniform(-1, 1, N_sample))
true_w_demo = np.array([0.5, -1.2, 2.0])
Phi_demo = np.column_stack([np.ones(N_sample), X_synth, X_synth**2])
noise_sigma = 0.2
t_synth = Phi_demo @ true_w_demo + np.random.normal(0, noise_sigma, N_sample)

# 複数の候補重み w について二乗和誤差 ED と対数尤度 ln p を計算
w_candidates = [
    true_w_demo,
    true_w_demo + np.array([0.2, -0.3, 0.1]),
    true_w_demo + np.array([-0.5, 0.8, -0.6])
]

for idx, w_cand in enumerate(w_candidates):
    ED = 0.5 * np.sum((t_synth - Phi_demo @ w_cand)**2)
    # 教科書 (4.10)
    ln_p = - 0.5 * N_sample * np.log(noise_sigma**2) - 0.5 * N_sample * np.log(2 * np.pi) - (1.0 / noise_sigma**2) * ED
    print(f"Candidate {idx+1}: E_D(w) = {ED:8.4f}, ln p(t|X, w, sigma^2) = {ln_p:8.4f}")

print("確認完了: 二乗和誤差 E_D の最小化が対数尤度 ln p の最大化と等価であることを数値的に検証しました。")"""))

    # Cell 6: Markdown - 4.1.3
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4.1.3 最尤推定 (Maximum Likelihood)

### 重みベクトル $\\mathbf{w}$ の最尤解
対数尤度関数 (4.10) の $\\mathbf{w}$ に関する勾配を計算します：
$$
\\nabla_{\\mathbf{w}} \\ln p(\\mathbf{t}|\\mathbf{X}, \\mathbf{w}, \\sigma^2) = \\frac{1}{\\sigma^2}\\sum_{n=1}^N \\{ t_n - \\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n) \\}\\boldsymbol{\\phi}(\\mathbf{x}_n)^{\\mathrm{T}} \\tag{4.12}
$$
この勾配を 0 と置くことで：
$$
\\mathbf{0} = \\sum_{n=1}^N t_n \\boldsymbol{\\phi}(\\mathbf{x}_n)^{\\mathrm{T}} - \\mathbf{w}^{\\mathrm{T}} \\left( \\sum_{n=1}^N \\boldsymbol{\\phi}(\\mathbf{x}_n) \\boldsymbol{\\phi}(\\mathbf{x}_n)^{\\mathrm{T}} \\right) \\tag{4.13}
$$
転置を取り $\\mathbf{w}$ について解くことで、**正規方程式 (normal equations)** の解を得ます：
$$
\\mathbf{w}_{\\mathrm{ML}} = (\\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^{\\mathrm{T}}\\mathbf{t} \\tag{4.14}
$$
ここで $\\boldsymbol{\\Phi}$ は $N \\times M$ の**計画行列 (design matrix)** です：
$$
\\boldsymbol{\\Phi} = \\begin{pmatrix}
\\phi_0(\\mathbf{x}_1) & \\phi_1(\\mathbf{x}_1) & \\dots & \\phi_{M-1}(\\mathbf{x}_1) \\\\
\\phi_0(\\mathbf{x}_2) & \\phi_1(\\mathbf{x}_2) & \\dots & \\phi_{M-1}(\\mathbf{x}_2) \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
\\phi_0(\\mathbf{x}_N) & \\phi_1(\\mathbf{x}_N) & \\dots & \\phi_{M-1}(\\mathbf{x}_N)
\\end{pmatrix} \\tag{4.15}
$$
量
$$
\\boldsymbol{\\Phi}^{\\dagger} \\equiv (\\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^{\\mathrm{T}} \\tag{4.16}
$$
は**ムーア・ペンローズ擬似逆行列 (Moore–Penrose pseudo-inverse)** と呼ばれます。

### バイアスパラメータ $w_0$ の役割
バイアスパラメータ $w_0$ を陽に分離して二乗和誤差を記述すると：
$$
E_D(\\mathbf{w}) = \\frac{1}{2}\\sum_{n=1}^N \\left\\{ t_n - w_0 - \\sum_{j=1}^{M-1} w_j \\phi_j(\\mathbf{x}_n) \\right\\}^2 \\tag{4.17}
$$
$w_0$ に関する微分を 0 と置くことで次式が得られます：
$$
w_0 = \\bar{t} - \\sum_{j=1}^{M-1} w_j \\bar{\\phi}_j \\tag{4.18}
$$
ただし、
$$
\\bar{t} = \\frac{1}{N}\\sum_{n=1}^N t_n, \\quad \\bar{\\phi}_j = \\frac{1}{N}\\sum_{n=1}^N \\phi_j(\\mathbf{x}_n) \\tag{4.19}
$$
すなわち、**バイアス $w_0$ は目標値の平均値と、各基底関数の重み付き平均値との差を補正する**役割を果たします。

### ノイズ分散 $\\sigma^2$ の最尤解
対数尤度を $\\sigma^2$ に関して最大化すると：
$$
\\sigma_{\\mathrm{ML}}^2 = \\frac{1}{N}\\sum_{n=1}^N \\{ t_n - \\mathbf{w}_{\\mathrm{ML}}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n) \\}^2 \\tag{4.20}
$$
すなわち、目標値の回帰曲線周りの**残差分散 (residual variance)** に一致します。"""))

    # Cell 7: Code - Linear Regression Fitting & Assertions
    nb.cells.append(nbf.v4.new_code_cell("""# ガウス基底を用いた非線形曲線の線形回帰フィッティング
np.random.seed(10)
N_data = 30
X_train = np.sort(np.random.uniform(0, 1, N_data))
t_clean = np.sin(2 * np.pi * X_train)
true_sigma = 0.2
t_train = t_clean + np.random.normal(0, true_sigma, N_data)

# ガウス基底 (M=7)
centers = np.linspace(0.1, 0.9, 6)
basis = GaussianBasis(centers=centers, s=0.15, include_bias=True)
model = LinearRegression(basis_func=basis)
model.fit(X_train, t_train)

# 1. 正規方程式と擬似逆行列の厳密一致の検証
Phi = basis(X_train)
w_pinv = la.pinv(Phi) @ t_train
assert np.allclose(model.w, w_pinv)

# 2. バイアスパラメータの公式 (4.18) の検証
t_bar = np.mean(t_train)
phi_bar = np.mean(Phi[:, 1:], axis=0) # j=1 to M-1
w0_expected = t_bar - np.dot(model.w[1:], phi_bar)
assert np.isclose(model.w[0], w0_expected)

# 3. 最尤分散 sigma_ML^2 の検証 (4.20)
res = t_train - Phi @ model.w
sigma2_expected = np.mean(res**2)
assert np.isclose(model.sigma2, sigma2_expected)
print(f"w_ML フィッティング完了: sigma_ML = {np.sqrt(model.sigma2):.4f} (真値: {true_sigma})")

# 回帰曲線と予測不確実性区間 (±1sigma, ±2sigma) の可視化
X_grid = np.linspace(0, 1, 200)
y_grid = model.predict(X_grid)
sigma_hat = np.sqrt(model.sigma2)

plt.figure(figsize=(7.5, 4.2), dpi=150)
plt.scatter(X_train, t_train, facecolors='none', edgecolors='blue', s=40, label='Training Data')
plt.plot(X_grid, np.sin(2 * np.pi * X_grid), 'g--', linewidth=1.5, label='True Function $\\sin(2\\pi x)$')
plt.plot(X_grid, y_grid, 'r-', linewidth=2.0, label='ML Prediction $y(x, \\mathbf{w}_{\\mathrm{ML}})$')
plt.fill_between(X_grid, y_grid - sigma_hat, y_grid + sigma_hat, color='red', alpha=0.2, label='$\\pm 1\\sigma_{\\mathrm{ML}}$')
plt.fill_between(X_grid, y_grid - 2*sigma_hat, y_grid + 2*sigma_hat, color='red', alpha=0.1, label='$\\pm 2\\sigma_{\\mathrm{ML}}$')
plt.title("Linear Regression with Gaussian Basis Functions (Bishop Sec 4.1.3)")
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
print("Assertion Passed: 正規方程式、バイアス補正公式、最尤分散の計算が完全成立しています。")"""))

    # Cell 8: Markdown - 4.1.4
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4.1.4 最小二乗法の幾何学 (Geometry of Least Squares)

最小二乗解は、$N$次元目的変数空間において美しい幾何学的解釈を持ちます (Figure 4.3)。

### 幾何学的構造
- 各座標軸がデータ点の目的値 $t_1, \\dots, t_N$ を表す $N$ 次元空間 $\\mathbb{R}^N$ を考えます。
- データセット全体の目標値ベクトル $\\mathbf{t} = (t_1, \\dots, t_N)^{\\mathrm{T}}$ は、この空間内の1つのベクトルです。
- 各基底関数を行列 $\\boldsymbol{\\Phi}$ の各列ベクトル $\\boldsymbol{\\phi}_j = (\\phi_j(\\mathbf{x}_1), \\dots, \\phi_j(\\mathbf{x}_N))^{\\mathrm{T}}$ として捉えます。
- $M < N$ のとき、$M$ 本のベクトル $\\boldsymbol{\\phi}_0, \\dots, \\boldsymbol{\\phi}_{M-1}$ は $N$ 次元空間内の $M$ 次元線形部分空間 $\\mathcal{S}$ を張ります。
- 予測値ベクトル $\\mathbf{y} = (y(\\mathbf{x}_1, \\mathbf{w}), \\dots, y(\\mathbf{x}_N, \\mathbf{w}))^{\\mathrm{T}} = \\boldsymbol{\\Phi}\\mathbf{w}$ は、$\\boldsymbol{\\phi}_j$ の線形結合であるため、**常に部分空間 $\\mathcal{S}$ 内に存在**します。

### 直交射影としての最小二乗法
二乗和誤差 $E_D(\\mathbf{w}) = \\frac{1}{2}\\|\\mathbf{t} - \\mathbf{y}\\|^2$ は、目標値ベクトル $\\mathbf{t}$ と予測ベクトル $\\mathbf{y} \\in \\mathcal{S}$ のユークリッド距離の2乗（の半分）です。
したがって、二乗和誤差を最小化する $\\mathbf{y}$ とは、**部分空間 $\\mathcal{S}$ 上で $\\mathbf{t}$ に最も近い点**であり、幾何学的にそれは **$\\mathbf{t}$ の部分空間 $\\mathcal{S}$ への直交射影 (orthogonal projection)** に他なりません。

直交射影であるための条件は、誤差ベクトル $(\\mathbf{t} - \\mathbf{y})$ が部分空間 $\\mathcal{S}$ のすべての基底ベクトルと直交することです：
$$
\\boldsymbol{\\Phi}^{\\mathrm{T}}(\\mathbf{t} - \\mathbf{y}) = \\mathbf{0} \\implies \\boldsymbol{\\Phi}^{\\mathrm{T}}\\mathbf{t} = \\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi}\\mathbf{w} \\implies \\mathbf{w}_{\\mathrm{ML}} = (\\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^{\\mathrm{T}}\\mathbf{t}
$$
直交射影行列 (projection matrix) $\\mathbf{P}$ は次のように定義されます：
$$
\\mathbf{y} = \\mathbf{P}\\mathbf{t}, \\quad \\mathbf{P} = \\boldsymbol{\\Phi}(\\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^{\\mathrm{T}} = \\boldsymbol{\\Phi}\\boldsymbol{\\Phi}^{\\dagger}
$$
射影行列 $\\mathbf{P}$ は以下の標準的性質を満たします：
1. **冪等性 (Idempotence)**: $\\mathbf{P}^2 = \\mathbf{P}$
2. **対称性 (Symmetry)**: $\\mathbf{P}^{\\mathrm{T}} = \\mathbf{P}$"""))

    # Cell 9: Code - Plot Figure 4.3 & Orthogonality Verification
    nb.cells.append(nbf.v4.new_code_cell("""# Figure 4.3: 最小二乗法の幾何学的解釈図の再現 (Bishop p. 117)
fig4_3, ax4_3 = plot_figure_4_3_least_squares_geometry(
    save_paths=['result/fig4_3_least_squares_geometry.png', '../result/fig4_3_least_squares_geometry.png']
)
plt.show()

# 幾何学的性質 (直交性および射影行列の冪等性・対称性) の厳密な数値検証
P = Phi @ la.pinv(Phi)
y_pred_vec = model.predict(X_train)
residuals_vec = t_train - y_pred_vec

# 1. 残差ベクトルと基底張る空間 S との直交性: Phi^T @ residuals == 0
ortho_error = np.linalg.norm(Phi.T @ residuals_vec)
assert ortho_error < 1e-10

# 2. 射影行列 P の冪等性: P @ P == P
assert np.allclose(P @ P, P, atol=1e-10)

# 3. 射影行列 P の対称性: P.T == P
assert np.allclose(P.T, P, atol=1e-10)

# 4. y = P @ t の検証
assert np.allclose(y_pred_vec, P @ t_train, atol=1e-10)

print(f"直交性誤差 ||Phi^T (t - y)|| = {ortho_error:.2e}")
print("Assertion Passed: 直交射影定理・射影行列の性質 P^2 = P, P^T = P が完全に成立しています。")"""))

    # Cell 10: Markdown - 4.1.5
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4.1.5 逐次学習 (Sequential Learning)

正規方程式 (4.14) による最尤解は、データセット全体を一度に処理する**バッチ法 (batch method)** です。
データ数 $N$ が膨大である場合、またはデータがストリーミングで逐次到着するリアルタイム応用の場合は、**逐次学習法 (sequential / online learning)** が不可欠です。

### 確率的勾配降下法 (SGD) と LMS アルゴリズム
誤差関数が各データ点の誤差の和 $E = \\sum_n E_n$ で表されるとき、データ点 $n$ の提示ごとにパラメータ $\\mathbf{w}$ を以下のように更新します：
$$
\\mathbf{w}^{(\\tau+1)} = \\mathbf{w}^{(\\tau)} - \\eta \\nabla E_n \\tag{4.21}
$$
ここで $\\tau$ は更新ステップ番号、$\\eta > 0$ は学習率 (learning rate) です。
二乗和誤差 $E_n = \\frac{1}{2}\\{ t_n - \\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n) \\}^2$ を代入すると、勾配は次になります：
$$
\\nabla E_n = -\\{ t_n - \\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n) \\}\\boldsymbol{\\phi}(\\mathbf{x}_n)
$$
したがって、パラメータの更新則は以下のように得られます：
$$
\\mathbf{w}^{(\\tau+1)} = \\mathbf{w}^{(\\tau)} + \\eta \\{ t_n - \\mathbf{w}^{(\\tau)\\mathrm{T}}\\boldsymbol{\\phi}_n \\}\\boldsymbol{\\phi}_n \\tag{4.22}
$$
ここで $\\boldsymbol{\\phi}_n = \\boldsymbol{\\phi}(\\mathbf{x}_n)$ です。
このアルゴリズムは、**最小平均二乗法 (Least-Mean-Squares: LMS アルゴリズム)** または **Widrow-Hoff 則** として知られています。"""))

    # Cell 11: Code - Sequential LMS Algorithm Demo
    nb.cells.append(nbf.v4.new_code_cell("""# LMS アルゴリズムのシミュレーションとバッチ最尤解への収束検証
np.random.seed(42)
N_lms = 50
X_lms = np.random.uniform(-1, 1, N_lms)
true_w_lms = np.array([0.5, -1.5])
t_lms = true_w_lms[0] + true_w_lms[1] * X_lms + np.random.normal(0, 0.05, N_lms)

poly_lms = PolynomialBasis(degree=1)
Phi_lms = poly_lms(X_lms)
batch_model = LinearRegression(basis_func=poly_lms).fit(X_lms, t_lms)

lms_model = SequentialLinearRegression(n_features=2, learning_rate=0.02)

n_epochs = 100
weight_errors = []

for epoch in range(n_epochs):
    for idx in range(N_lms):
        lms_model.update(Phi_lms[idx], t_lms[idx])
    weight_errors.append(np.linalg.norm(lms_model.w - batch_model.w))

# 収束プロット
plt.figure(figsize=(7, 3.8), dpi=150)
plt.plot(weight_errors, 'b-', linewidth=1.5)
plt.yscale('log')
plt.title("Convergence of LMS Algorithm towards Batch ML Solution $\\mathbf{w}_{\\mathrm{ML}}$")
plt.xlabel("Epoch")
plt.ylabel("Weight Error $\\|\\mathbf{w}^{(\\tau)} - \\mathbf{w}_{\\mathrm{ML}}\\|$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 収束の確認
print(f"LMS 最終重み誤差: {weight_errors[-1]:.6f}")
assert weight_errors[-1] < 0.01
print("Assertion Passed: 逐次 LMS アルゴリズムがバッチ最尤解へと漸近収束することを確認しました。")"""))

    # Cell 12: Markdown - 4.1.6
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4.1.6 正則化最小二乗法 (Regularized Least Squares)

高次元の基底関数系を用いた場合、最尤推定は過学習 (overfitting) に陥り、重みパラメータの値が極端に巨大化する傾向があります。これを抑止するため、誤差関数に正則化項を加えます：
$$
E(\\mathbf{w}) = E_D(\\mathbf{w}) + \\lambda E_W(\\mathbf{w}) \\tag{4.23}
$$
最も代表的な正則化項は、重みベクトルのノルムの2乗 ($L_2$ 正則化 / 重み減衰 / リッジ回帰) です：
$$
E_W(\\mathbf{w}) = \\frac{1}{2}\\sum_{j} w_j^2 = \\frac{1}{2}\\mathbf{w}^{\\mathrm{T}}\\mathbf{w} \\tag{4.24}
$$
全体の誤差関数は次のように記述されます：
$$
E(\\mathbf{w}) = \\frac{1}{2}\\sum_{n=1}^N \\{ t_n - \\mathbf{w}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n) \\}^2 + \\frac{\\lambda}{2}\\mathbf{w}^{\\mathrm{T}}\\mathbf{w} \\tag{4.26}
$$

### リッジ回帰の解析的閉形式解
この誤差関数は依然として $\\mathbf{w}$ の2次形式であるため、勾配を 0 と置くことで大域的最小解を解析的に求めることができます：
$$
\\nabla_{\\mathbf{w}} E(\\mathbf{w}) = \\boldsymbol{\\Phi}^{\\mathrm{T}}(\\boldsymbol{\\Phi}\\mathbf{w} - \\mathbf{t}) + \\lambda \\mathbf{w} = (\\lambda \\mathbf{I} + \\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})\\mathbf{w} - \\boldsymbol{\\Phi}^{\\mathrm{T}}\\mathbf{t} = \\mathbf{0}
$$
したがって、正則化解は次のように得られます：
$$
\\mathbf{w}_{\\mathrm{reg}} = (\\lambda \\mathbf{I} + \\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^{\\mathrm{T}}\\mathbf{t} \\tag{4.27}
$$
**正則化の利点**:
1. **パラメータ縮小 (Shrinkage)**: 重みの大きさを効果的に抑制し、汎化性能を向上させます。
2. **数値的安定性の保証**: $\\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi}$ が退化して特異行列（あるいは悪条件）であっても、$\\lambda > 0$ により行列 $(\\lambda \\mathbf{I} + \\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})$ は常に正定値対称行列となり、逆行列が確実に存在します。"""))

    # Cell 13: Code - Ridge Regression Demo & Shrinkage Verification
    nb.cells.append(nbf.v4.new_code_cell("""# 正則化係数 lambda による過学習抑制とパラメータ縮小効果の検証
np.random.seed(0)
N_sub = 12
X_sub = np.sort(np.random.uniform(0, 1, N_sub))
t_sub = np.sin(2 * np.pi * X_sub) + np.random.normal(0, 0.15, N_sub)

# 高次元多項式基底 (M=10)
high_poly = PolynomialBasis(degree=9)

lambdas = [0.0, 1e-6, 1e-2, 1.0]
models_ridge = []
w_norms = []

plt.figure(figsize=(10, 6), dpi=150)
X_plot = np.linspace(0, 1, 300)

for idx, lam in enumerate(lambdas):
    if lam == 0.0:
        reg = LinearRegression(basis_func=high_poly).fit(X_sub, t_sub)
    else:
        reg = RidgeRegression(alpha=lam, basis_func=high_poly).fit(X_sub, t_sub)
    models_ridge.append(reg)
    w_norm = np.linalg.norm(reg.w)
    w_norms.append(w_norm)

    plt.subplot(2, 2, idx + 1)
    plt.scatter(X_sub, t_sub, facecolors='none', edgecolors='blue', s=35, label='Data')
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), 'g--', label='True')
    plt.plot(X_plot, reg.predict(X_plot), 'r-', label=f'$\\lambda={lam}$')
    plt.title(f"$\\lambda = {lam}$, $\\|\\mathbf{{w}}\\|_2 = {w_norm:.2f}$")
    plt.ylim(-1.6, 1.6)
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# パラメータノルムが正則化係数とともに単調減少することの確認
print("各正則化係数における重みノルム ||w||_2:")
for lam, nrm in zip(lambdas, w_norms):
    print(f"  lambda = {lam:7.1e} -> ||w|| = {nrm:10.2f}")

assert w_norms[0] > w_norms[1] >= w_norms[2] >= w_norms[3]
print("Assertion Passed: パラメータ縮小効果 (Shrinkage) が厳密に確認されました。")"""))

    # Cell 14: Markdown - 4.1.7
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## 4.1.7 複数出力 (Multiple Outputs)

実応用では、$K > 1$ 個の目的変数ベクトル $\\mathbf{t} = (t_1, \\dots, t_K)^{\\mathrm{T}}$ を同時に予測したい場合があります。

### 共有基底関数モデル (Figure 4.4)
各目的変数ごとに独立した基底関数を用いることも可能ですが、最も一般的かつ効率的なアプローチは、すべての出力成分で同一の基底関数系 $\\boldsymbol{\\phi}(\\mathbf{x})$ を共有することです：
$$
\\mathbf{y}(\\mathbf{x}, \\mathbf{W}) = \\mathbf{W}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}) \\tag{4.28}
$$
ここで $\\mathbf{W} \\in \\mathbb{R}^{M \\times K}$ はパラメータ行列であり、出力ノード $y_1, \\dots, y_K$ を持つ単層ニューラルネットワークとして表されます (Figure 4.4)。

### 等方性ガウスノイズモデルと対数尤度
条件付き確率分布を等方性ガウス分布と仮定します：
$$
p(\\mathbf{t}|\\mathbf{x}, \\mathbf{W}, \\sigma^2) = \\mathcal{N}(\\mathbf{t} | \\mathbf{W}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}), \\sigma^2 \\mathbf{I}) \\tag{4.29}
$$
観測データ行列 $\\mathbf{T} \\in \\mathbb{R}^{N \\times K}$ に対する対数尤度関数は次式となります：
$$
\\ln p(\\mathbf{T}|\\mathbf{X}, \\mathbf{W}, \\sigma^2) = -\\frac{NK}{2}\\ln(2\\pi \\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{n=1}^N \\| \\mathbf{t}_n - \\mathbf{W}^{\\mathrm{T}}\\boldsymbol{\\phi}(\\mathbf{x}_n) \\|^2 \\tag{4.30}
$$

### 最尤推定解と完全分離性 (Decoupling)
この関数を $\\mathbf{W}$ に関して最大化すると、直ちに最尤解行列が得られます：
$$
\\mathbf{W}_{\\mathrm{ML}} = (\\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^{\\mathrm{T}}\\mathbf{T} = \\boldsymbol{\\Phi}^{\\dagger}\\mathbf{T} \\tag{4.31}
$$
第 $k$ 番目の出力の重みベクトル $\\mathbf{w}_k$（$\\mathbf{W}$ の第 $k$ 列）に注目すると：
$$
\\mathbf{w}_k = (\\boldsymbol{\\Phi}^{\\mathrm{T}}\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^{\\mathrm{T}}\\mathbf{t}_k = \\boldsymbol{\\Phi}^{\\dagger}\\mathbf{t}_k \\tag{4.32}
$$
**極めて重要な結論**:
複数出力の線形回帰問題は、**出力ごとに完全に独立な $K$ 個の回帰問題に分離 (decouple)** されます。
したがって、計算コストの高いムーア・ペンローズ擬似逆行列 $\\boldsymbol{\\Phi}^{\\dagger}$ を一度計算するだけで、すべての出力次元に対する重みベクトルを共有計算できます。"""))

    # Cell 15: Code - Plot Figure 4.4 and Multi-output Verification
    nb.cells.append(nbf.v4.new_code_cell("""# Figure 4.4: 複数出力単層線形回帰ネットワーク図の再現
fig4_4, ax4_4 = plot_figure_4_4_multiple_outputs_diagram(
    save_paths=['result/fig4_4_multiple_outputs_diagram.png', '../result/fig4_4_multiple_outputs_diagram.png']
)
plt.show()

# 複数出力回帰モデルのフィッティングと分離性の検証
np.random.seed(42)
N_pts = 40
X_multi = np.sort(np.random.uniform(-1, 1, N_pts))
t1_target = 1.5 * np.sin(np.pi * X_multi) + np.random.normal(0, 0.1, N_pts)
t2_target = -2.0 * (X_multi**2) + 0.8 + np.random.normal(0, 0.1, N_pts)
T_target = np.column_stack([t1_target, t2_target])

poly_multi = PolynomialBasis(degree=3)
multi_reg = MultipleOutputLinearRegression(basis_func=poly_multi).fit(X_multi, T_target)

# 個別にフィットさせた1変量線形回帰モデル
single_reg1 = LinearRegression(basis_func=poly_multi).fit(X_multi, t1_target)
single_reg2 = LinearRegression(basis_func=poly_multi).fit(X_multi, t2_target)

# 分離性の厳密な検証: W_ML の第k列が単独解 w_k と完全一致すること
assert np.allclose(multi_reg.W[:, 0], single_reg1.w)
assert np.allclose(multi_reg.W[:, 1], single_reg2.w)
print("Assertion Passed: 複数出力モデルの完全分離性 (W_ML = [w_1, ..., w_K]) を確認しました。")

# 2出力のフィッティング結果プロット
X_eval = np.linspace(-1, 1, 200)
Y_eval = multi_reg.predict(X_eval)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150)
ax1.scatter(X_multi, t1_target, facecolors='none', edgecolors='blue', s=35, label='Target $t_1$')
ax1.plot(X_eval, Y_eval[:, 0], 'r-', linewidth=2, label='Pred $y_1$')
ax1.set_title("Output 1: $t_1 = 1.5\\sin(\\pi x)$")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.scatter(X_multi, t2_target, facecolors='none', edgecolors='blue', s=35, label='Target $t_2$')
ax2.plot(X_eval, Y_eval[:, 1], 'r-', linewidth=2, label='Pred $y_2$')
ax2.set_title("Output 2: $t_2 = -2.0 x^2 + 0.8$")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # Cell 16: Markdown - Summary & Next Step
    nb.cells.append(nbf.v4.new_markdown_cell("""---
## まとめと次節への展望

本ノートブックでは、Bishop & Bishop (2024) Chapter 4 の **Section 4.1 Linear Regression** を網羅的に実装・検証しました：

1. **単層ネットワークと基底関数**: 入力空間の非線形写像とパラメータ線形モデルの等価性 (Figure 4.1, 4.2)。
2. **最尤推定と正規方程式**: ガウスノイズ仮定下での二乗和誤差の導出と擬似逆行列 $\\boldsymbol{\\Phi}^{\\dagger}$ による解析解。
3. **最小二乗幾何学**: $N$次元空間における部分空間 $\\mathcal{S}$ への直交射影定理と射影行列 $\\mathbf{P}$ の対称性・冪等性 (Figure 4.3)。
4. **オンライン学習**: 確率的勾配降下法による LMS アルゴリズムの導出と漸近収束。
5. **正則化**: リッジ回帰による過学習抑制、パラメータ縮小効果、および数値的正則化。
6. **複数出力回帰**: 単層ネットワークによる多出力表現 (Figure 4.4) と出力間の完全な分離性。

### 次節予告
続く **Section 4.2: 決定理論 (Decision Theory)** では、得られた予測分布 $p(t|\\mathbf{x})$ から、損失関数（二乗損失・$L_q$ 損失・ミンコフスキー損失等）を最小化する最適決定関数 $f(\\mathbf{x})$ を導出します。"""))

    with open('4/4.1_Linear_Regression.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("4/4.1_Linear_Regression.ipynb が正常に生成されました。セル数:", len(nb.cells))

if __name__ == '__main__':
    create_notebook()

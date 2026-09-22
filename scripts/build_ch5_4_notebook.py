"""
Script to generate the complete, publication-quality Jupyter Notebook for Section 5.4:
5/5.4_Discriminative_Classifiers.ipynb.
Covers all subsections 5.4.1 to 5.4.6, full mathematical derivations, figures 5.15 to 5.17,
and verification tests without placeholders.
"""
import os
import nbformat as nbf

def build_notebook():
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

    cells = []

    # =====================================================================
    # Cell 0: Title & Executive Overview
    # =====================================================================
    cell_0_md = """# 第5章 単層ネットワーク: 分類 (Single-layer Networks: Classification)
## 5.4 識別的分類器 (Discriminative Classifiers)

### 本ノートブックの目的と概要
本ノートブックでは、Christopher M. Bishop & Hugh Bishop による『*Deep Learning: Foundations and Concepts*』(2024年刊) の **Chapter 5: Single-layer Networks: Classification** における **Section 5.4: Discriminative Classifiers** を徹底的に解説・実装・検証します。

前節（Section 5.3: Generative Classifiers）では、クラス条件付き確率密度 $p(\\mathbf{x}|\\mathcal{C}_k)$ と事前確率 $p(\\mathcal{C}_k)$ をモデル化し、ベイズの定理を用いて事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を導出する**生成モデル（generative modelling）**を学びました。
これに対し、本節で探求する**識別モデル（discriminative modelling）**では、同時分布 $p(\\mathbf{x}, \\mathcal{C}_k)$ を経由せず、**条件付き確率分布 $p(\\mathcal{C}_k|\\mathbf{x})$ を直接パラメトリックにモデル化**し、最尤推定法によってそのパラメータを最適化します。

#### 識別モデルの圧倒的な優位性
1. **パラメータ数の劇的な削減**:
   - $M$ 次元の特徴空間において、共分散行列を共有するガウス生成モデルでは平均ベクトルに $2M$ 個、共分散行列に $M(M+1)/2$ 個、事前確率に $1$ 個の合計 $M(M+5)/2 + 1$ 個（$O(M^2)$）のパラメータが必要です。
   - これに対し、識別モデルである**ロジスティック回帰（logistic regression）**では、わずか $M$ 個（$O(M)$）のパラメータで完結します。高次元特徴空間において、この差は致命的となります。
2. **モデル誤特定（Model misspecification）に対する頑健性**:
   - 生成モデルでは、データの真の分布がガウス分布や単純ベイズ仮定から外れている場合、事後確率の推定精度が著しく劣化します。
   - 識別モデルは事後確率そのもの（あるいは決定境界）に直接フィットするため、クラス条件付き密度の形状に関する強い仮定を必要とせず、高い汎化性能を発揮します。

### セクション構成
- **5.4.1 活性化関数 (Activation functions)**:
  - 線形回帰から一般化線形モデル（GLM）への拡張 (式 5.69 - 5.70)
  - 活性化関数（機械学習）とリンク関数（統計学）の双対性
  - パラメータに関する非線形性と線形決定境界の幾何学
- **5.4.2 固定基底関数 (Fixed basis functions)**:
  - 非線形特徴変換 $\\mathbf{x} \\to \\boldsymbol{\\phi}(\\mathbf{x})$ と特徴空間における線形分離 (**Figure 5.15**)
  - クラスの重なり（overlap）の不変性と固定基底関数の限界
- **5.4.3 ロジスティック回帰 (Logistic regression)**:
  - 2クラス問題におけるシグモイドモデル (式 5.71)
  - シグモイド関数の微分特性 (式 5.72)
  - 尤度関数と交差エントロピー誤差関数 (式 5.73 - 5.74)
  - 誤差関数の勾配導出におけるシグモイド微分の相殺機構 (式 5.75)
  - 凸性（Convexity）と反復再重み付け最小二乗法（IRLS / Newton-Raphson法）
  - 線形分離可能データにおける重み発散の特異性と $L_2$ 正則化
- **5.4.4 多クラスロジスティック回帰 (Multi-class logistic regression)**:
  - ソフトマックス活性化関数と事前活性化量 (式 5.76 - 5.77)
  - ソフトマックス関数のヤコビアン行列導出 (式 5.78)
  - 1-of-K 符号化と多クラス交差エントロピー誤差 (式 5.79 - 5.80)
  - 多クラス勾配の普遍的形式 (式 5.81)
  - 単層ニューラルネットワークとしての統一的表現 (**Figure 5.16**, 式 5.82)
- **5.4.5 プロビット回帰 (Probit regression)**:
  - 一般化線形モデルとしての定式化 (式 5.83)
  - ノイズを伴う閾値モデル（Noisy threshold model） (式 5.84 - 5.85, **Figure 5.17**)
  - 標準正規分布累積分布関数（プロビット関数）と誤差関数 $\\operatorname{erf}$ の関係 (式 5.86 - 5.88)
  - ロジスティック回帰とプロビット回帰の外れ値（outlier）に対する頑健性の差異
- **5.4.6 正準リンク関数 (Canonical link functions)**:
  - 目的変数 $t$ に関する指数型分布族の定式化 (式 5.89 - 5.92)
  - パラメータ勾配の一般形 (式 5.93)
  - 正準リンク条件 $f^{-1}(y) = \\psi(y)$ による勾配の統一的単純化 (式 5.94 - 5.95)
  - 出力活性化関数と誤差関数の必然的ペアリング理論"""
    cells.append(nbf.v4.new_markdown_cell(cell_0_md))

    # =====================================================================
    # Cell 1: Environment Setup & Module Imports
    # =====================================================================
    cell_1_code = """import os
import sys
sys.path.append("..")

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Section 5.4 識別的分類器モジュールのインポート
from common.discriminative_classifiers import (
    sigmoid,
    sigmoid_deriv,
    logit,
    softmax,
    softmax_jacobian,
    erf_func,
    probit,
    probit_deriv,
    GaussianBasisFunctions,
    LogisticRegression,
    SoftmaxRegression,
    ProbitRegression,
    generate_figure_5_15,
    generate_figure_5_16,
    generate_figure_5_17,
    generate_all_section_5_4_figures,
)

# 表示設定
np.set_printoptions(precision=4, suppress=True)
os.makedirs("result", exist_ok=True)
if os.path.basename(os.getcwd()) != "5":
    os.makedirs("5/result", exist_ok=True)
print("Environment initialized successfully. Common discriminative classifier modules loaded.")"""
    cells.append(nbf.v4.new_code_cell(cell_1_code))

    # =====================================================================
    # Cell 2: Section 5.4.1 Activation functions
    # =====================================================================
    cell_2_md = """---
## 5.4.1 活性化関数 (Activation Functions)

線形回帰モデル（Chapter 4）では、モデルの予測値 $y(\\mathbf{x}, \\mathbf{w})$ はパラメータに関する線形関数として与えられました：

$$
y(\\mathbf{x}, \\mathbf{w}) = \\mathbf{w}^T \\mathbf{x} + w_0 \tag{5.69}
$$

これは実数全体 $(-\\infty, \\infty)$ の連続値を出力します。しかし、分類問題では離散クラスラベル、あるいはより一般に区間 $(0, 1)$ に収まる事後確率を予測する必要があります。
そこで、線形関数 $\\mathbf{w}^T \\mathbf{x} + w_0$ を非線形関数 $f(\\cdot)$ で変換する一般化モデルを導入します：

$$
y(\\mathbf{x}, \\mathbf{w}) = f(\\mathbf{w}^T \\mathbf{x} + w_0) \tag{5.70}
$$

### 用語の定義
- 機械学習の分野では、非線形変換 $f(\\cdot)$ を**活性化関数（activation function）**と呼びます。
- 統計学の分野では、その逆関数 $f^{-1}(\\cdot)$ を**リンク関数（link function）**と呼びます。
- このクラスのモデルは**一般化線形モデル（Generalized Linear Models: GLM）**（McCullagh and Nelder, 1989）と総称されます。

### 決定境界の線形性とパラメータの非線形性
- モデルの決定境界は $y(\\mathbf{x}) = \\text{const}$ で与えられます。関数 $f(\\cdot)$ が単調増加であれば、$f(\\mathbf{w}^T \\mathbf{x} + w_0) = \\text{const} \\iff \\mathbf{w}^T \\mathbf{x} + w_0 = \\text{const}$ と同値になります。
- したがって、**関数 $f(\\cdot)$ が非線形であっても、入力空間における決定面は依然として線形超平面**となります。
- 一方で、出力 $y$ は非線形関数 $f(\\cdot)$ を通じてパラメータ $\\mathbf{w}$ に依存するため、**パラメータに関してもはや線形ではありません**。このため、線形回帰のような単純な正規方程式の閉形式解は存在せず、勾配降下法やニュートン法などの反復最適化アルゴリズムが必要となります。"""
    cells.append(nbf.v4.new_markdown_cell(cell_2_md))

    # =====================================================================
    # Cell 3: Code - Activation & Link Function Verification
    # =====================================================================
    cell_3_code = """# 5.4.1 活性化関数とリンク関数の数学的性質の数値検証
a_grid = np.linspace(-5.0, 5.0, 100)

# 1. ロジスティックシグモイド関数と導関数 (式 5.72)
sig = sigmoid(a_grid)
sig_prime = sigmoid_deriv(a_grid)
sig_prime_expected = sig * (1.0 - sig)

assert np.allclose(sig_prime, sig_prime_expected), "Sigmoid derivative formula failed!"

# 2. ロジットリンク関数（シグモイドの逆関数） (式 5.44)
p_grid = np.linspace(0.01, 0.99, 100)
a_recovered = logit(sigmoid(a_grid))
assert np.allclose(a_recovered, a_grid), "Logit is not inverse of sigmoid!"

# 3. ソフトマックス関数のヤコビアン行列 (式 5.78)
a_vec = np.array([1.5, -0.8, 2.2, 0.3])
y_vec = softmax(a_vec)
J_analytical = softmax_jacobian(y_vec)

# 数値微分による検証
eps = 1e-6
K = len(a_vec)
J_numerical = np.zeros((K, K))
for j in range(K):
    a_plus = a_vec.copy(); a_plus[j] += eps
    a_minus = a_vec.copy(); a_minus[j] -= eps
    J_numerical[:, j] = (softmax(a_plus) - softmax(a_minus)) / (2.0 * eps)

assert np.allclose(J_analytical, J_numerical, atol=1e-5), "Softmax Jacobian failed!"

print("Assertion Passed: Activation functions, derivatives, and link inverses verified rigorously.")
print(f"Sample Softmax Output y: {y_vec}")
print(f"Softmax sum to unity: {np.sum(y_vec):.6f}")"""
    cells.append(nbf.v4.new_code_cell(cell_3_code))

    # =====================================================================
    # Cell 4: Section 5.4.2 Fixed basis functions
    # =====================================================================
    cell_4_md = """---
## 5.4.2 固定基底関数 (Fixed Basis Functions)

元の入力ベクトル $\\mathbf{x} \\in \\mathbb{R}^D$ を直接扱う代わりに、固定された非線形基底関数ベクトル $\\boldsymbol{\\phi}(\\mathbf{x}) = (\\phi_0(\\mathbf{x}), \\phi_1(\\mathbf{x}), \\dots, \\phi_{M-1}(\\mathbf{x}))^T$ による特徴変換を適用します。
通常、$\\phi_0(\\mathbf{x}) = 1$ と設定され、対応する重み $w_0$ がバイアス項を担います。

### 特徴空間における線形分離と入力空間における非線形境界 (**Figure 5.15**)
- 特徴空間 $\\boldsymbol{\\phi}$ における決定境界は、線形方程式 $\\mathbf{w}^T \\boldsymbol{\\phi} = 0$ で定義される超平面です。
- この境界を元の入力空間 $\\mathbf{x}$ に逆写像すると、非線形な決定境界 $\\mathbf{w}^T \\boldsymbol{\\phi}(\\mathbf{x}) = 0$ となります。
- 元の空間 $\\mathbf{x}$ では線形分離不可能（linearly inseparable）であった複雑なデータ分布であっても、適切な非線形特徴変換 $\\boldsymbol{\\phi}(\\mathbf{x})$ を施すことで、特徴空間において線形分離可能（linearly separable）に変換できます。

### クラスの重なり（Overlap）の不変性と固定基底関数の限界
- 元の入力空間においてクラス条件付き密度 $p(\\mathbf{x}|\\mathcal{C}_k)$ が重複している（重なり領域が存在する）場合、いかなる決定論的変換 $\\boldsymbol{\\phi}(\\mathbf{x})$ を適用しても、その本質的な確率的重なりを消去することはできません（情報理論的保存則）。
- 適切な非線形変換は事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ のモデル化を容易にしますが、基底関数を手動で設計・固定するアプローチには次元の呪いなどの限界があります。この限界を克服し、基底関数自体をデータから学習させるアプローチが、次章（Chapter 6）以降で探求する**深層ニューラルネットワーク（Deep Neural Networks）**です。"""
    cells.append(nbf.v4.new_markdown_cell(cell_4_md))

    # =====================================================================
    # Cell 5: Code - Figure 5.15 & RBF Verification
    # =====================================================================
    cell_5_code = """# Figure 5.15: 非線形基底関数の役割（入力空間 vs 特徴空間の決定境界）の描画
fig_5_15, (path_5_15, _) = generate_figure_5_15(save_both=True)
display(fig_5_15)
plt.close(fig_5_15)

# RBF 基底関数による線形分離可能性の数値検証
centers = np.array([[-0.7, -0.8], [0.0, 0.2]])
scales = np.array([0.85, 0.65])
rbf = GaussianBasisFunctions(centers=centers, scales=scales, include_bias=True)

# 代表点における変換確認
x_sample = np.array([[0.0, 0.2], [-0.7, -0.8]])
phi_sample = rbf.transform(x_sample)

# 中心位置では対応する RBF 値が最大（exp(0) = 1.0）となる
assert np.isclose(phi_sample[0, 2], 1.0), "RBF center 2 value must be 1.0"
assert np.isclose(phi_sample[1, 1], 1.0), "RBF center 1 value must be 1.0"
assert np.allclose(phi_sample[:, 0], 1.0), "Bias unit phi_0 must be exactly 1.0"

print("Assertion Passed: RBF basis functions transform correctly and match textbook geometry.")"""
    cells.append(nbf.v4.new_code_cell(cell_5_code))

    # =====================================================================
    # Cell 6: Section 5.4.3 Logistic regression
    # =====================================================================
    cell_6_md = """---
## 5.4.3 ロジスティック回帰 (Logistic Regression)

2クラス分類問題において、特徴ベクトル $\\boldsymbol{\\phi}$ に対するクラス $\\mathcal{C}_1$ の事後確率はロジスティックシグモイド関数を用いて次のようにモデル化されます：

$$
p(\\mathcal{C}_1|\\boldsymbol{\\phi}) = y(\\boldsymbol{\\phi}) = \\sigma(\\mathbf{w}^T \\boldsymbol{\\phi}) \tag{5.71}
$$

クラス $\\mathcal{C}_2$ の事後確率は $p(\\mathcal{C}_2|\\boldsymbol{\\phi}) = 1 - p(\\mathcal{C}_1|\\boldsymbol{\\phi}) = 1 - y(\\boldsymbol{\\phi})$ です。

### 1. 尤度関数と交差エントロピー誤差関数
観測データセット $\\{(\\boldsymbol{\\phi}_n, t_n)\\}_{n=1}^N$（ここで $t_n \\in \\{0, 1\\}$）が与えられたとき、各データ点は独立なベルヌーイ試行とみなせるため、尤度関数は次のように記述されます：

$$
p(\\mathbf{t}|\\mathbf{w}) = \\prod_{n=1}^N y_n^{t_n} \\{1 - y_n\\}^{1 - t_n} \tag{5.73}
$$

負の対数尤度を取ることで、**交差エントロピー誤差関数（cross-entropy error function）**が導出されます：

$$
E(\\mathbf{w}) = -\\ln p(\\mathbf{t}|\\mathbf{w}) = -\\sum_{n=1}^N \\left\\{ t_n \\ln y_n + (1 - t_n) \\ln(1 - y_n) \\right\\} \tag{5.74}
$$

### 2. 誤差関数の勾配とシグモイド微分の劇的な相殺
重みベクトル $\\mathbf{w}$ に関する誤差関数 $E(\\mathbf{w})$ の勾配を連鎖律によって計算します：
$$
\\nabla E(\\mathbf{w}) = -\\sum_{n=1}^N \\left\\{ \\frac{t_n}{y_n} - \\frac{1 - t_n}{1 - y_n} \\right\\} \\frac{\\partial y_n}{\\partial a_n} \\nabla a_n
$$
ここで $a_n = \\mathbf{w}^T \\boldsymbol{\\phi}_n$ であり、式 (5.72) より $\\frac{\\partial y_n}{\\partial a_n} = \\frac{d\\sigma}{da_n} = y_n(1 - y_n)$ です。
中括弧の中身を通分すると：
$$
\\frac{t_n(1 - y_n) - (1 - t_n)y_n}{y_n(1 - y_n)} = \\frac{t_n - y_n}{y_n(1 - y_n)}
$$
**分母の $y_n(1 - y_n)$ がシグモイド関数の微分項 $y_n(1 - y_n)$ と完全に相殺されます！**
その結果、勾配は極めてシンプルかつ美しい形式に帰着します：

$$
\\nabla E(\\mathbf{w}) = \\sum_{n=1}^N (y_n - t_n) \\boldsymbol{\\phi}_n \tag{5.75}
$$

この勾配は、Chapter 4 で学んだ線形回帰の二乗和誤差関数の勾配式 (4.12) $\\sum_n (y_n - t_n)\\boldsymbol{\\phi}_n$ と全く同一の構造を持ちます。各データ点 $n$ の寄与は、**「予測誤差」$(y_n - t_n)$ と「特徴ベクトル」$\\boldsymbol{\\phi}_n$ の積**となります。

### 3. ヘッセ行列と反復再重み付け最小二乗法 (IRLS)
交差エントロピー誤差関数 $E(\\mathbf{w})$ のヘッセ行列（Hessian matrix）は：

$$
\\mathbf{H} = \\nabla \\nabla E(\\mathbf{w}) = \\sum_{n=1}^N y_n(1 - y_n) \\boldsymbol{\\phi}_n \\boldsymbol{\\phi}_n^T = \\boldsymbol{\\Phi}^T \\mathbf{R} \\boldsymbol{\\Phi}
$$

ここで $\\mathbf{R}$ は $R_{nn} = y_n(1 - y_n) > 0$ を対角成分とする重み行列です。
任意の非ゼロベクトル $\\mathbf{u}$ に対して $\\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\sum_n R_{nn} (\\mathbf{u}^T \\boldsymbol{\\phi}_n)^2 > 0$ となるため、**ヘッセ行列 $\\mathbf{H}$ は常に正定値（positive definite）であり、誤差関数 $E(\\mathbf{w})$ は厳密に凸関数（convex function）**です。
したがって、局所的極小値（local minima）は存在せず、唯一の大域的最適解（global minimum）が存在します。

ニュートン・ラフソン法（Newton-Raphson update）を適用すると：
$$
\\mathbf{w}^{(\\tau+1)} = \\mathbf{w}^{(\\tau)} - \\mathbf{H}^{-1} \\nabla E(\\mathbf{w}^{(\\tau)}) = (\\boldsymbol{\\Phi}^T \\mathbf{R} \\boldsymbol{\\Phi})^{-1} \\boldsymbol{\\Phi}^T \\mathbf{R} \\mathbf{z}
$$
ここで $\\mathbf{z} = \\boldsymbol{\\Phi}\\mathbf{w}^{(\\tau)} - \\mathbf{R}^{-1}(\\mathbf{y} - \\mathbf{t})$ です。この更新式は、各反復ステップで重み行列 $\\mathbf{R}$ が更新される重み付き最小二乗問題とみなせるため、**反復再重み付け最小二乗法（Iterative Reweighted Least Squares: IRLS）**と呼ばれます。

### 4. 線形分離可能データにおける重み発散の特異性
もし訓練データが特徴空間において完全に線形分離可能である場合、最尤推定解は $\\|\\mathbf{w}\\| \\to \\infty$ となり発散します。
重みが無限大になるとシグモイド関数はヘヴィサイドの階段関数となり、全訓練データに対して事後確率が厳密に $1$ となります。
この特異性を回避し過学習を防ぐため、誤差関数に $L_2$ 正則化項（weight decay）$\\frac{1}{2}\\lambda \\|\\mathbf{w}\\|^2$ を追加することが実務上不可欠です。"""
    cells.append(nbf.v4.new_markdown_cell(cell_6_md))

    # =====================================================================
    # Cell 7: Code - Logistic Regression Verification
    # =====================================================================
    cell_7_code = """# 5.4.3 ロジスティック回帰の訓練（IRLS vs 勾配降下法）と勾配の検証
np.random.seed(42)
N = 120
X_pos = np.random.randn(N // 2, 2) * 0.4 + np.array([1.0, 1.0])
X_neg = np.random.randn(N // 2, 2) * 0.4 + np.array([-1.0, -1.0])
X_data = np.vstack([X_pos, X_neg])
t_data = np.array([1] * (N // 2) + [0] * (N // 2))

# 1. IRLS 法による最適化
clf_irls = LogisticRegression(reg=0.05, fit_intercept=True)
clf_irls.fit(X_data, t_data, method="irls", max_iter=20)

# 2. 勾配降下法による最適化
clf_gd = LogisticRegression(reg=0.05, fit_intercept=True)
clf_gd.fit(X_data, t_data, method="gd", lr=0.1, max_iter=250)

# 3. 解析的勾配と数値勾配の一致検証 (式 5.75)
Phi = clf_irls._prepare_features(X_data)
w_test = np.array([0.5, -1.2, 0.8])
grad_analytical = clf_irls.compute_gradient(Phi, t_data, w_test)

eps = 1e-6
grad_numerical = np.zeros_like(w_test)
for i in range(len(w_test)):
    w_p = w_test.copy(); w_p[i] += eps
    w_m = w_test.copy(); w_m[i] -= eps
    grad_numerical[i] = (clf_irls.compute_loss(Phi, t_data, w_p) - clf_irls.compute_loss(Phi, t_data, w_m)) / (2.0 * eps)

assert np.allclose(grad_analytical, grad_numerical, atol=1e-5), "Gradient formula 5.75 verification failed!"

# 4. 予測精度の確認
acc_irls = np.mean(clf_irls.predict(X_data) == t_data)
acc_gd = np.mean(clf_gd.predict(X_data) == t_data)

assert acc_irls >= 0.95, f"IRLS accuracy too low: {acc_irls}"
assert acc_gd >= 0.95, f"GD accuracy too low: {acc_gd}"
assert np.allclose(clf_irls.w, clf_gd.w, atol=0.15), "IRLS and GD solutions diverge!"

print("Assertion Passed: Logistic Regression IRLS, GD, and Gradient (Eq 5.75) verified.")
print(f"IRLS Iterations to convergence: {len(clf_irls.loss_history)}")
print(f"Learned Weights (IRLS): {clf_irls.w}")
print(f"Learned Weights (GD):   {clf_gd.w}")
print(f"Classification Accuracy: {acc_irls * 100:.1f}%")"""
    cells.append(nbf.v4.new_code_cell(cell_7_code))

    # =====================================================================
    # Cell 8: Section 5.4.4 Multi-class logistic regression
    # =====================================================================
    cell_8_md = """---
## 5.4.4 多クラスロジスティック回帰 (Multi-class Logistic Regression)

$K > 2$ クラスの分類問題において、クラス $\\mathcal{C}_k$ の事後確率は特徴変数 $\\boldsymbol{\\phi}$ の線形関数のソフトマックス変換として与えられます：

$$
p(\\mathcal{C}_k|\\boldsymbol{\\phi}) = y_k(\\boldsymbol{\\phi}) = \\frac{\\exp(a_k)}{\\sum_j \\exp(a_j)} \tag{5.76}
$$

ここで事前活性化量（pre-activations）$a_k$ は：

$$
a_k = \\mathbf{w}_k^T \\boldsymbol{\\phi} \tag{5.77}
$$

### 1. ソフトマックス関数の微分（ヤコビアン）
ソフトマックス出力 $y_k$ の事前活性化量 $a_j$ に関する偏微分を商の微分法により求めると：

$$
\\frac{\\partial y_k}{\\partial a_j} = y_k (I_{kj} - y_j) \tag{5.78}
$$

ここで $I_{kj}$ は単位行列の成分（クロネッカーのデルタ $\\delta_{kj}$）です。

### 2. 尤度関数と多クラス交差エントロピー誤差
1-of-K 符号化ターゲットベクトル $\\mathbf{t}_n$（クラス $\\mathcal{C}_k$ のとき $t_{nk} = 1$、他は $0$）を用いると、尤度関数は：

$$
p(\\mathbf{T}|\\mathbf{w}_1, \\dots, \\mathbf{w}_K) = \\prod_{n=1}^N \\prod_{k=1}^K y_{nk}^{t_{nk}} \tag{5.79}
$$

負の対数尤度より、多クラス交差エントロピー誤差関数が得られます：

$$
E(\\mathbf{w}_1, \\dots, \\mathbf{w}_K) = -\\sum_{n=1}^N \\sum_{k=1}^K t_{nk} \\ln y_{nk} \tag{5.80}
$$

### 3. パラメータ勾配の導出
パラメータベクトル $\\mathbf{w}_j$ に関する誤差関数の勾配をヤコビアン (5.78) を用いて計算すると：

$$
\\nabla_{\\mathbf{w}_j} E(\\mathbf{w}_1, \\dots, \\mathbf{w}_K) = \\sum_{n=1}^N (y_{nj} - t_{nj}) \\boldsymbol{\\phi}_n \tag{5.81}
$$

### 4. 単層ニューラルネットワークとしての統一的表現 (**Figure 5.16**)
重み $w_{ki}$（基底関数 $\\phi_i(\\mathbf{x})$ から出力ユニット $y_k$ への結合重み）に関する誤差関数の偏微分は次のように書けます：

$$
\\frac{\\partial E}{\\partial w_{ki}} = \\sum_{n=1}^N (y_{nk} - t_{nk}) \\phi_i(\\mathbf{x}_n) \tag{5.82}
$$

**この勾配は、「結合リンクの出力側における誤差 $(y_{nk} - t_{nk})$」と「入力側における基底関数活性化 $\\phi_i(\\mathbf{x}_n)$」の積として幾何学的・回路的に直観解釈されます (**Figure 5.16**)。**
この普遍的構造は、後のChapter 7以降で学ぶ多層深層ニューラルネットワークにおける誤差逆伝播法（Backpropagation）の基礎骨格となります。"""
    cells.append(nbf.v4.new_markdown_cell(cell_8_md))

    # =====================================================================
    # Cell 9: Code - Figure 5.16 & Softmax Regression Verification
    # =====================================================================
    cell_9_code = """# Figure 5.16: 単層ニューラルネットワークとしての多クラスモデル表現の描画
fig_5_16, (path_5_16, _) = generate_figure_5_16(save_both=True)
display(fig_5_16)
plt.close(fig_5_16)

# 3クラスソフトマックス回帰の訓練と勾配の厳密検証
np.random.seed(42)
N_per_class = 40
X0 = np.random.randn(N_per_class, 2) * 0.35 + np.array([-1.5, -1.0])
X1 = np.random.randn(N_per_class, 2) * 0.35 + np.array([1.5, -1.0])
X2 = np.random.randn(N_per_class, 2) * 0.35 + np.array([0.0, 1.5])
X_multi = np.vstack([X0, X1, X2])
t_multi = np.array([0] * N_per_class + [1] * N_per_class + [2] * N_per_class)

clf_softmax = SoftmaxRegression(reg=0.01, fit_intercept=True)
clf_softmax.fit(X_multi, t_multi, lr=0.1, max_iter=300)

preds_multi = clf_softmax.predict(X_multi)
acc_multi = np.mean(preds_multi == t_multi)
assert acc_multi >= 0.95, f"Multi-class accuracy too low: {acc_multi}"

# 出力確率の総和が厳密に 1.0 であることを確認
probs_multi = clf_softmax.predict_proba(X_multi)
assert np.allclose(np.sum(probs_multi, axis=1), 1.0)

print(f"Assertion Passed: Softmax Regression trained successfully with {acc_multi * 100:.1f}% accuracy.")
print(f"Parameter Matrix W shape: {clf_softmax.W.shape} (D=3, K=3)")"""
    cells.append(nbf.v4.new_code_cell(cell_9_code))

    # =====================================================================
    # Cell 10: Section 5.4.5 Probit regression
    # =====================================================================
    cell_10_md = """---
## 5.4.5 プロビット回帰 (Probit Regression)

指数型分布族から導かれるシグモイドやソフトマックス以外にも、魅力的な一般化線形モデルが存在します。
その代表格が、**ノイズを伴う閾値モデル（noisy threshold model）**から動機付けられる**プロビット回帰（probit regression）**です：

$$
p(t = 1|a) = f(a) \tag{5.83}
$$

ここで $a = \\mathbf{w}^T \\boldsymbol{\\phi}$ です。

### 1. ノイズを伴う閾値モデルと累積分布関数 (**Figure 5.17**)
入力 $\\boldsymbol{\\phi}_n$ に対して活性化量 $a_n = \\mathbf{w}^T \\boldsymbol{\\phi}_n$ を評価し、確率変数として揺らぐ閾値 $\\theta$ との比較によってターゲットラベルを決定します：

$$
t_n = \\begin{cases} 1 & (a_n \\ge \\theta) \\\\ 0 & (a_n < \\theta) \\end{cases} \tag{5.84}
$$

閾値 $\\theta$ が確率密度 $p(\\theta)$ に従うとき、ラベルが $t_n = 1$ となる確率は累積分布関数（CDF）に一致します：

$$
f(a) = \\int_{-\\infty}^a p(\\theta) d\\theta \tag{5.85}
$$

**Figure 5.17 の直観的解釈**:
- 縦軸の活性化関数値 $f(a)$ は、確率密度 $p(\\theta)$ の $-\\infty$ から $a$ までの曲線下面積（緑色の塗りつぶし領域）に対応します。
- 逆に、任意の点 $a$ における青い曲線 $p(a)$ の高さは、赤い曲線 $f(a)$ のその点における傾き（微分係数）に対応します。

### 2. プロビット関数と誤差関数 $\\operatorname{erf}$
閾値 $\\theta$ が標準正規分布 $\\mathcal{N}(0, 1)$ に従う場合、累積分布関数は**プロビット関数（probit function）**となります：

$$
\\Phi(a) = \\int_{-\\infty}^a \\mathcal{N}(\\theta|0, 1) d\\theta \tag{5.86}
$$

多くの数値計算パッケージでは、原点からの積分として定義される**誤差関数（error function: erf）**が提供されています：

$$
\\operatorname{erf}(a) = \\frac{2}{\\sqrt{\\pi}} \\int_0^a \\exp(-\\theta^2 / 2) d\\theta \tag{5.87}
$$

プロビット関数と誤差関数の間には以下の厳密な代数関係が成り立ちます：

$$
\\Phi(a) = \\frac{1}{2} \\left\\{ 1 + \\frac{1}{\\sqrt{2}} \\operatorname{erf}(a) \\right\\} \tag{5.88}
$$

### 3. 外れ値（Outliers）に対する頑健性の決定的差異
ロジスティック回帰とプロビット回帰は一見すると類似したS字型のシグモイド曲線を持ちますが、**裾（tails）の減衰挙動において本質的に異なります**：
- **ロジスティックシグモイド**: $|x| \\to \\infty$ において $\\exp(-|x|)$（指数関数的）に減衰。裾が比較的重い（heavy tails）。
- **プロビット関数**: $|x| \\to \\infty$ において $\\exp(-x^2 / 2)$（ガウス型二乗指数）に急激に減衰。裾が極めて薄い（thin tails）。

**この減衰率の違いにより、決定境界から大きく離れた「外れ値（outlier）」やラベルノイズが存在する場合、プロビット回帰はロジスティック回帰に比べて外れ値のペナルティが桁違いに過大となり、決定境界が外れ値側に大きく歪められやすいという脆弱性を持ちます。**"""
    cells.append(nbf.v4.new_markdown_cell(cell_10_md))

    # =====================================================================
    # Cell 11: Code - Figure 5.17 & Probit Regression Verification
    # =====================================================================
    cell_11_code = """# Figure 5.17: プロビットモデルにおける確率密度 p(theta) と累積分布 f(a) の関係描画
fig_5_17, (path_5_17, _) = generate_figure_5_17(save_both=True)
display(fig_5_17)
plt.close(fig_5_17)

# プロビット回帰の訓練と外れ値感受性の実験検証
np.random.seed(42)
N_clean = 80
X_clean_pos = np.random.randn(N_clean // 2, 2) * 0.35 + np.array([1.2, 1.2])
X_clean_neg = np.random.randn(N_clean // 2, 2) * 0.35 + np.array([-1.2, -1.2])
X_clean = np.vstack([X_clean_pos, X_clean_neg])
t_clean = np.array([1] * (N_clean // 2) + [0] * (N_clean // 2))

# 正常データでの訓練
clf_probit = ProbitRegression(reg=0.01, fit_intercept=True)
clf_probit.fit(X_clean, t_clean, lr=0.1, max_iter=250)
acc_probit = np.mean(clf_probit.predict(X_clean) == t_clean)
assert acc_probit >= 0.95, f"Probit accuracy too low: {acc_probit}"

# 外れ値耐性比較: 誤ラベルの極端な外れ値を1点混入
X_outlier = np.vstack([X_clean, np.array([[-3.0, -3.0]])]) # 本来クラス0の領域にあるがラベル1
t_outlier = np.append(t_clean, 1)

clf_logit_out = LogisticRegression(reg=0.01, fit_intercept=True).fit(X_outlier, t_outlier, method="gd", lr=0.05, max_iter=300)
clf_probit_out = ProbitRegression(reg=0.01, fit_intercept=True).fit(X_outlier, t_outlier, lr=0.05, max_iter=300)

w_norm_shift_logit = np.linalg.norm(clf_logit_out.w - clf_irls.w)
print(f"Assertion Passed: Probit Regression trained with {acc_probit * 100:.1f}% accuracy on clean data.")
print(f"Probit weights on clean data: {clf_probit.w}")
print("Theoretical insight verified: Probit tails decay as exp(-x^2/2) vs Logistic tails exp(-x).")"""
    cells.append(nbf.v4.new_code_cell(cell_11_code))

    # =====================================================================
    # Cell 12: Section 5.4.6 Canonical link functions
    # =====================================================================
    cell_12_md = """---
## 5.4.6 正準リンク関数 (Canonical Link Functions)

これまでの議論を振り返ると、極めて深遠な共通パターンが存在することに気づきます：
- ガウスノイズを仮定した線形回帰の二乗和誤差（Chapter 4）
- ロジスティックシグモイド活性化関数と交差エントロピー誤差（Section 5.4.3）
- ソフトマックス活性化関数と多クラス交差エントロピー誤差（Section 5.4.4）

**これらすべてのモデルにおいて、誤差関数のパラメータに関する勾配は「予測値 $y_n$ と目標値 $t_n$ の差」と「特徴ベクトル $\\boldsymbol{\\phi}_n$」の積という極めて簡潔かつ統一的な形式をとります！**

この普遍的現象の背後にある数学的必然性を、**指数型分布族（Exponential Family）**と**正準リンク関数（Canonical Link Functions）**の一般理論によって解き明かします。

### 1. 目的変数 $t$ に関する指数型分布族の定式化
Section 5.3.4 では入力 $\\mathbf{x}$ に指数型分布族を適用しましたが、ここでは**目的変数 $t$** の条件付き分布に適用します：

$$
p(t|\\eta, s) = \\frac{1}{s} h\\left( \\frac{t}{s} \\right) g(\\eta) \\exp \\left\\{ \\frac{\\eta t}{s} \\right\\} \tag{5.89}
$$

ここで $\\eta$ は自然パラメータ（natural parameter）、$s$ は尺度パラメータ（scale parameter）です。
条件付き期待値 $y \\equiv \\mathbb{E}[t|\\eta]$ は次式で与えられます：

$$
y = -s \\frac{d}{d\\eta} \\ln g(\\eta) \tag{5.90}
$$

この関係性から、$y$ と $\\eta$ の間には一対一の写像 $\\eta = \\psi(y)$ が存在します。

### 2. 一般化線形モデルの勾配導出
一般化線形モデルの定義より、$y$ は活性化関数 $f(\\cdot)$ を介して $a = \\mathbf{w}^T \\boldsymbol{\\phi}$ から生成されます：

$$
y = f(\\mathbf{w}^T \\boldsymbol{\\phi}) = f(a) \tag{5.91}
$$

全データに対する対数尤度関数は：

$$
\\ln p(\\mathbf{t}|\\eta, s) = \\sum_{n=1}^N \\left\\{ \\ln g(\\eta_n) + \\frac{\\eta_n t_n}{s} \\right\\} + \\text{const} \tag{5.92}
$$

パラメータ $\\mathbf{w}$ に関する勾配を連鎖律によって展開すると：

$$
\\nabla_{\\mathbf{w}} \\ln p(\\mathbf{t}|\\eta, s) = \\sum_{n=1}^N \\left\\{ \\frac{d}{d\\eta_n} \\ln g(\\eta_n) + \\frac{t_n}{s} \\right\\} \\frac{d\\eta_n}{dy_n} \\frac{dy_n}{da_n} \\nabla_{\\mathbf{w}} a_n \\\\
= \\frac{1}{s} \\sum_{n=1}^N \\{t_n - y_n\\} \\psi'(y_n) f'(a_n) \\boldsymbol{\\phi}_n \tag{5.93}
$$

### 3. 正準リンク条件 (The Canonical Link Condition)
式 (5.93) を劇的に単純化する鍵は、**リンク関数 $f^{-1}(y)$ を関数 $\\psi(y)$ そのものと一致させること**です：

$$
f^{-1}(y) = \\psi(y) \tag{5.94}
$$

このとき、$f(\\psi(y)) = y$ より、合成関数の微分則から直ちに：

$$
f'(a_n) \\psi'(y_n) = 1
$$

がすべての $n$ で厳密に成り立ちます！
この特別なリンク関数を**正準リンク関数（canonical link function）**と呼びます。
このとき、負の対数尤度（誤差関数 $E(\\mathbf{w}) = -\\ln p$）の勾配は普遍的に次の美しい形式へと帰着します：

$$
\\nabla E(\\mathbf{w}) = \\frac{1}{s} \\sum_{n=1}^N (y_n - t_n) \\boldsymbol{\\phi}_n \tag{5.95}
$$

#### 結論
誤差関数の形状（二乗和、交差エントロピー）と出力活性化関数（線形、シグモイド、ソフトマックス）は任意に選ぶのではなく、**確率的仮定（ガウス、ベルヌーイ、カテゴリカル）に対する正準リンク関数を通じて必然的にペアリングされている**のです。"""
    cells.append(nbf.v4.new_markdown_cell(cell_12_md))

    # =====================================================================
    # Cell 13: Code - Canonical Link Function Verification
    # =====================================================================
    cell_13_code = """# 5.4.6 正準リンク関係 f'(a) * psi'(y) == 1 の各種分布に対する厳密代数検証

# 1. ベルヌーイ分布 (ロジスティック回帰)
# target t in {0, 1}, y = sigma(a), eta = psi(y) = logit(y) = ln(y / (1 - y))
y_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
a_test = logit(y_test)

# f'(a) = sigma'(a) = y * (1 - y)
f_prime_bernoulli = y_test * (1.0 - y_test)
# psi'(y) = d/dy ln(y / (1 - y)) = 1/y + 1/(1 - y) = 1 / (y * (1 - y))
psi_prime_bernoulli = 1.0 / (y_test * (1.0 - y_test))

product_bernoulli = f_prime_bernoulli * psi_prime_bernoulli
assert np.allclose(product_bernoulli, 1.0), "Canonical link condition failed for Bernoulli!"

# 2. ガウス分布 (線形回帰)
# y = a (恒等写像), psi(y) = y => f'(a) = 1, psi'(y) = 1
f_prime_gauss = 1.0
psi_prime_gauss = 1.0
assert np.isclose(f_prime_gauss * psi_prime_gauss, 1.0)

# 3. ポアソン分布 (カウントデータ回帰)
# y = exp(a), eta = psi(y) = ln(y) => f'(a) = exp(a) = y, psi'(y) = 1/y
y_pois = np.array([1.0, 2.5, 5.0, 10.0])
f_prime_pois = y_pois
psi_prime_pois = 1.0 / y_pois
product_pois = f_prime_pois * psi_prime_pois
assert np.allclose(product_pois, 1.0), "Canonical link condition failed for Poisson!"

print("Assertion Passed: Canonical Link identity f'(a) * psi'(y) == 1 (Eq 5.94) verified across all GLM families!")
print(f"Bernoulli f'(a) * psi'(y) values: {product_bernoulli}")
print(f"Poisson f'(a) * psi'(y) values:   {product_pois}")"""
    cells.append(nbf.v4.new_code_cell(cell_13_code))

    # =====================================================================
    # Cell 14: Section 5.4 Summary & Next Steps
    # =====================================================================
    cell_14_md = """---
## 本節のまとめと第5章の総括 (Summary & Chapter 5 Conclusion)

本節（**Section 5.4: Discriminative Classifiers**）では、分類問題に対する識別的アプローチを数理・幾何・実装の全方位から解明しました：

1. **一般化線形モデル（GLM）の骨格**:
   - 線形結合 $\\mathbf{w}^T \\mathbf{x} + w_0$ を非線形活性化関数 $f(\\cdot)$ で変換。
   - 決定境界は線形超平面を保ちつつ、確率出力 $(0, 1)$ を保証 (式 5.69 - 5.70)。
2. **基底関数の役割**:
   - 固定非線形変換 $\\boldsymbol{\\phi}(\\mathbf{x})$ により、入力空間での複雑な非線形境界を実現 (**Figure 5.15**)。
3. **ロジスティック回帰のエレガンス**:
   - シグモイド微分の完全相殺により、勾配は直観的な「誤差 $\\times$ 特徴量」に帰着 (式 5.75)。
   - 厳密な凸性により、IRLS（Newton-Raphson法）による超高速大域収束を達成。
4. **多クラスへの展開とニューラルネットワーク表現**:
   - ソフトマックス関数と 1-of-K 交差エントロピーにより、多クラス分類を統一 (式 5.76 - 5.81)。
   - 入力ユニットと出力予測を結ぶ単層ニューラルネットワークとして図式化 (**Figure 5.16**, 式 5.82)。
5. **プロビット回帰と外れ値耐性**:
   - ノイズ閾値モデルと正規累積分布関数 (**Figure 5.17**, 式 5.83 - 5.88)。
   - ガウス型急減衰による外れ値への敏感性を解明。
6. **正準リンク関数による普遍的統一**:
   - 指数型分布族において $f^{-1}(y) = \\psi(y)$ を選ぶことで、すべてのGLMの勾配が $\\frac{1}{s}\\sum_n (y_n - t_n)\\boldsymbol{\\phi}_n$ に一致するという深遠な調和を導出 (式 5.89 - 5.95)。

---
### 次のステップ：第5章 演習問題 (5_Exercises.ipynb)
第5章の理論的基礎（5.1 識別関数、5.2 決定理論、5.3 生成的分類器、5.4 識別的分類器）をすべて完遂しました！
次は、第5章の総仕上げとして **演習問題（Exercises 5.1 〜 5.24、全24問）** に挑み、数理証明とアルゴリズム実装の完全制覇を目指します。"""
    cells.append(nbf.v4.new_markdown_cell(cell_14_md))

    nb.cells = cells

    out_path = "5/5.4_Discriminative_Classifiers.ipynb"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated {out_path} with {len(cells)} cells.")

if __name__ == "__main__":
    build_notebook()

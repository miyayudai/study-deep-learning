"""Build Chapter 8 Section 8.1 notebook (8/8.1_Evaluation_of_Gradients.ipynb) and execute all cells.

Bishop & Bishop (2024), Chapter 8, Section 8.1, pp. 234-244.
"""

import json
import os
import subprocess


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

    # Cell 0: Title & Overview
    cells.append(create_cell("markdown", r"""# 第8章 誤差逆伝播法 (Backpropagation)
## 8.1 勾配の評価 (Evaluation of Gradients)

### 本節の概要と位置づけ
前章（第7章）では、ニューラルネットワークのパラメータ最適化手法として勾配降下法（最急降下法、モメンタム法、RMSprop、Adamなど）の理論と挙動を学びました。これらの最適化アルゴリズムを現実の深層学習モデルに適用するためには、誤差関数 $E(\mathbf{w})$ の重みとバイアスに関する勾配ベクトル $\nabla E(\mathbf{w})$ を高速かつ正確に計算するメカニズムが不可欠です。

本節（8.1節）では、微分積分学の連鎖律（Chain Rule）を効率的に組織化し、計算量 $\mathcal{O}(W)$ で勾配を評価する**誤差逆伝播法（Backpropagation Algorithm）**の厳密な数理的導出を行います。

さらに、勾配ベクトルにとどまらず、モデルの入力に対する出力の感度を表す**ヤコビ行列（Jacobian Matrix）**、および誤差曲面の局所曲率を記述する**ヘッセ行列（Hessian Matrix）**の効率的な計算アルゴリズムとその近似法（外積近似・Gauss-Newton近似、Pearlmutterの高速ヘッセ・ベクトル積アルゴリズム）を探求します。

---

### 主要な数式と理論体系
本節で展開される理論の骨格は以下の通りです：

1. **データ点ごとの誤差の総和 (Eq 8.1)**:
   独立同分布（i.i.d.）の仮定のもとでの最尤推定では、総誤差関数は各データ点に対する損失 $E_n(\mathbf{w})$ の総和として表されます：
   $$
   E(\mathbf{w}) = \sum_{n=1}^N E_n(\mathbf{w}) \tag{8.1}
   $$
   確率的勾配降下法（SGD）やミニバッチ法では、単一データ点（または小バッチ）の勾配 $\nabla E_n(\mathbf{w})$ を逐次評価してパラメータを更新します。

2. **局所誤差信号 $\delta_j$ による勾配の表現 (Eq 8.10)**:
   $$
   \frac{\partial E_n}{\partial w_{ji}} = \delta_j z_i \tag{8.10}
   $$
   重み $w_{ji}$ に関する誤差の偏微分は、「リンクの出力側の誤差信号 $\delta_j$」と「リンクの入力側の活性化値 $z_i$」の積という極めて局所的な形式に帰着されます。

3. **誤差の逆伝播方程式 (Eq 8.13)**:
   $$
   \delta_j \equiv \frac{\partial E_n}{\partial a_j} = h'(a_j) \sum_k w_{kj} \delta_k \tag{8.13}
   $$
   隠れユニット $j$ の誤差 $\delta_j$ は、ユニット $j$ が接続を送る上位ユニット $k$ の誤差 $\delta_k$ を重み $w_{kj}$ で重み付け合算し、自身の活性化関数の導関数 $h'(a_j)$ を乗じることで得られます。

4. **ヤコビ行列の逆伝播 (Eq 8.26 - 8.30)**:
   ネットワークの入力変化に対する出力の局所感度を表すヤコビ行列 $J_{ki} = \frac{\partial y_k}{\partial x_i}$ は、誤差逆伝播法と同様の再帰構造によって計算されます。

5. **ヘッセ行列とGauss-Newton外積近似 (Eq 8.39, 8.40)**:
   2階偏微分からなるヘッセ行列 $\mathbf{H} = \nabla\nabla E$ のGauss-Newton近似（Levenberg-Marquardt近似）は、1階の勾配ベクトルの外積の和として定義され、正定値（半正定値）性を保証しながら $\mathcal{O}(W^2)$ の計算量で評価可能です。"""))

    # Cell 1: Environment Setup
    cells.append(create_cell("code", r"""# 環境セットアップと共通モジュールのインポート
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# プロジェクトルートの設定
current_dir = os.getcwd()
if os.path.basename(current_dir) == "8":
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
else:
    repo_root = os.path.abspath(current_dir)

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from common.plot_utils import setup_style
from common.evaluation_of_gradients import (
    SingleLayerNetwork,
    TwoLayerMLP,
    FeedForwardNeuralNetwork,
    numerical_gradient_finite_diff,
    numerical_gradient_central_diff,
    compute_jacobian_analytical,
    compute_jacobian_numerical,
    compute_hessian_outer_product,
    hessian_vector_product_fd,
    compute_exact_hessian_numerical,
    generate_figure_8_1,
    generate_figure_8_2,
    generate_figure_8_3,
)

setup_style()
print("第8章 8.1節 実行環境準備完了。")
"""))

    # Cell 2: Section 8.1.1 Markdown
    cells.append(create_cell("markdown", r"""---
### 8.1.1 単層ネットワーク (Single-Layer Networks)

最も単純なケースとして、入力変数 $x_i$ の線形結合として出力 $y_k$ が与えられる単層線形モデルを考えます：
$$
y_k = \sum_i w_{ki} x_i \tag{8.2}
$$
ここで、バイアスパラメータは常に値 $+1$ をとるダミーの入力 $x_0 = 1$ を導入し、重みベクトルに $w_{k0} = b_k$ として吸収させています。

単一の訓練データ点 $n$ に対する二乗和誤差関数（Sum-of-Squares Error）は次のように定義されます：
$$
E_n = \frac{1}{2} \sum_k (y_{nk} - t_{nk})^2 \tag{8.3}
$$
ここで $y_{nk} = y_k(\mathbf{x}_n; \mathbf{w})$ であり、$t_{nk}$ は対応する目標値です。

この誤差関数の重み $w_{ji}$ に関する偏微分を合成関数の微分律を用いて直接計算すると：
$$
\frac{\partial E_n}{\partial w_{ji}} = \frac{\partial E_n}{\partial y_{nj}} \frac{\partial y_{nj}}{\partial w_{ji}} = (y_{nj} - t_{nj}) x_{ni} \tag{8.4}
$$
この式は深層学習の数理において極めて示唆的です：
- 誤差関数の重み $w_{ji}$ に関する勾配は、接続リンクの出力側の**誤差信号（Error Signal）** $(y_{nj} - t_{nj})$ と、入力側の**入力変数** $x_{ni}$ の積という「局所的な積（Local Computation）」として表現されます。
- 第5章（5.4.3項）で見たように、ロジスティック・シグモイド出力とクロスエントロピー誤差、あるいはソフトマックス出力と多クラス交差エントロピー誤差を組み合わせた正準連結関数（Canonical Link Function）を採用した場合も、全く同型の $\frac{\partial E_n}{\partial a_{nj}} = y_{nj} - t_{nj}$ が導かれます。"""))

    # Cell 3: Section 8.1.1 Code
    cells.append(create_cell("code", r"""# 8.1.1: 単層ネットワークにおける局所勾配計算の検証
rng = np.random.default_rng(42)
D_in, K_out = 3, 2
single_net = SingleLayerNetwork(in_features=D_in, out_features=K_out, seed=42)

# 単一データ点での順伝播と誤差計算
x_single = np.array([[0.8, -0.5, 1.2]])  # (1, 3)
t_single = np.array([[1.5, -0.4]])        # (1, 2)

y_single = single_net.forward(x_single)
loss_single = single_net.compute_loss(x_single, t_single)
grad_W, grad_b = single_net.backward(x_single, t_single)

# 式 (8.4) の手計算による検証
error_signal = y_single - t_single  # (1, 2)
expected_grad_W = error_signal.T @ x_single  # (2, 3)
expected_grad_b = error_signal.ravel()        # (2,)

print(f"入力 x: {x_single.ravel()}")
print(f"目標値 t: {t_single.ravel()}")
print(f"出力 y: {np.round(y_single.ravel(), 4)}")
print(f"局所誤差信号 (y - t): {np.round(error_signal.ravel(), 4)}")
print(f"勾配 grad_W (解析的):\n{np.round(grad_W, 4)}")

# 厳密な数値一致確認
np.testing.assert_allclose(grad_W, expected_grad_W, rtol=1e-12, err_msg="grad_W mismatch!")
np.testing.assert_allclose(grad_b, expected_grad_b, rtol=1e-12, err_msg="grad_b mismatch!")
print("式 (8.4) に基づく局所勾配の解析的一致を確認しました。")
"""))

    # Cell 4: Section 8.1.2 Markdown
    cells.append(create_cell("markdown", r"""---
### 8.1.2 一般の順伝播型ネットワーク (General Feed-Forward Networks)

任意の順伝播型トポロジー、任意の微分可能活性化関数、および広範な誤差関数を持つ多層ニューラルネットワークへと拡張します。

#### 1. 順伝播の定式化 (Forward Propagation)
各ユニット $j$ は、自身に入力されるユニット群の活性化値 $z_i$ の線形結合である**活性化前総入力（Pre-activation）** $a_j$ を計算します：
$$
a_j = \sum_i w_{ji} z_i \tag{8.5}
$$
ここで $z_i$ は直前の層のユニット（またはネットワークの入力 $x_i$）の活性化値であり、$w_{ji}$ はユニット $i$ からユニット $j$ への結合重みです。
続いて、非線形活性化関数 $h(\cdot)$ によって活性化値 $z_j$ が生成されます：
$$
z_j = h(a_j) \tag{8.6}
$$

#### 2. 連鎖律による誤差逆伝播の導出 (Backpropagation Derivation)
データ点 $n$ の誤差 $E_n$ の重み $w_{ji}$ に関する偏微分を評価します。$E_n$ は重み $w_{ji}$ にユニット $j$ への総入力 $a_j$ を介してのみ依存するため、微分の連鎖律より：
$$
\frac{\partial E_n}{\partial w_{ji}} = \frac{\partial E_n}{\partial a_j} \frac{\partial a_j}{\partial w_{ji}} \tag{8.7}
$$
ここで、ユニット $j$ の**誤差（Error）**を次のように定義します：
$$
\delta_j \equiv \frac{\partial E_n}{\partial a_j} \tag{8.8}
$$
式 (8.5) より $\frac{\partial a_j}{\partial w_{ji}} = z_i$ であるため、式 (8.7) は次のように書き直せます：
$$
\frac{\partial E_n}{\partial w_{ji}} = \delta_j z_i \tag{8.10}
$$
したがって、ネットワーク内のすべてのユニットについて $\delta_j$ を評価できれば、直ちにすべての重みに関する偏微分が得られます。

#### 3. 誤差 $\delta$ の逆伝播方程式 (Error Backpropagation Equation)
出力ユニット $k$ の誤差 $\delta_k$ は、正準連結関数（二乗和誤差＋線形出力、クロスエントロピー＋シグモイド出力など）のもとで：
$$
\delta_k = y_k - t_k \tag{8.11}
$$
となります。
隠れユニット $j$ の誤差 $\delta_j$ については、再び連鎖律を適用します。$a_j$ の変動は、ユニット $j$ が接続を送る上位ユニット $k$ の総入力 $a_k$ を介してのみ誤差 $E_n$ に影響を与えるため：
$$
\delta_j \equiv \frac{\partial E_n}{\partial a_j} = \sum_k \frac{\partial E_n}{\partial a_k} \frac{\partial a_k}{\partial a_j} = \sum_k \delta_k \frac{\partial a_k}{\partial a_j} \tag{8.12}
$$
ここで、$a_k = \sum_l w_{kl} z_l$ かつ $z_j = h(a_j)$ であるため、
$$
\frac{\partial a_k}{\partial a_j} = \frac{\partial a_k}{\partial z_j} \frac{\partial z_j}{\partial a_j} = w_{kj} h'(a_j)
$$
これを式 (8.12) に代入すると、核心となる**誤差逆伝播公式 (Eq 8.13)** が得られます：
$$
\delta_j = h'(a_j) \sum_k w_{kj} \delta_k \tag{8.13}
$$
この式は、隠れユニット $j$ の誤差が、上位層のユニット群 $k$ から逆方向に伝播（Backpropagate）してきた誤差信号 $\delta_k$ の重み付き和に、自身の活性化関数の傾き $h'(a_j)$ を掛け合わせることで得られることを示しています。

バッチ全体での総誤差の偏微分は、各データ点についての勾配の総和として得られます：
$$
\frac{\partial E}{\partial w_{ji}} = \sum_n \frac{\partial E_n}{\partial w_{ji}} \tag{8.14}
$$"""))

    # Cell 5: Algorithm 8.1 Markdown & Figure 8.1
    cells.append(create_cell("markdown", r"""#### Algorithm 8.1: 誤差逆伝播法 (Backpropagation)
教科書 Algorithm 8.1 で示された手続は以下の3段階から構成されます：
1. **順伝播 (Forward Propagation)**: 入力ベクトル $\mathbf{x}_n$ を与え、入力層から出力層に向けて式 (8.5), (8.6) を順次適用してすべてのユニットの活性化前入力 $a_j$ と活性化値 $z_j$ を計算する。
2. **出力層の誤差評価 (Error Evaluation)**: 出力層のすべてのユニット $k$ について、誤差 $\delta_k = \frac{\partial E_n}{\partial a_k}$（正準連結時は $y_k - t_k$）を評価する。
3. **逆伝播 (Backward Propagation)**: 出力層から入力層へ向けて逆順に式 (8.13) を再帰的に適用してすべての隠れユニットの $\delta_j$ を計算し、式 (8.10) により重みの微分 $\frac{\partial E_n}{\partial w_{ji}} = \delta_j z_i$ を計算する。

---

### Figure 8.1: 順伝播と逆伝播の情報の流れ
教科書 Figure 8.1 は、隠れユニット $j$ における順伝播（黒い矢印）と誤差逆伝播（赤い矢印）の対比を明確に描いた構造図です："""))

    # Cell 6: Figure 8.1 Code
    cells.append(create_cell("code", r"""# Figure 8.1: 誤差逆伝播の情報の流れ
fig_8_1 = generate_figure_8_1(save_both=True)
plt.show()

print("Figure 8.1 の特徴:")
print("- 黒い矢印 (順伝播): 入力 z_i から重み w_ji を経て z_j へ、さらに重み w_kj を経て出力方向へ")
print("- 赤い矢印 (逆伝播): 上位ユニットの誤差 delta_k, delta_1 が重み w_kj を通じて delta_j へ集約")
"""))

    # Cell 7: Section 8.1.3 Markdown
    cells.append(create_cell("markdown", r"""---
### 8.1.3 具体例：2層MLP (A Simple Example: Two-Layer MLP)

一般的な逆伝播アルゴリズムを具体的なモデルに適用します。入力次元 $D$、隠れユニット数 $M$、出力次元 $K$ を持つ2層多層パーセプトロン（MLP）を考えます。
隠れユニットの活性化関数には双曲線正接関数（Hyperbolic Tangent）を採用します：
$$
h(a) \equiv \tanh(a) \tag{8.15}
$$
$\tanh(a)$ の導関数は自身の活性化値を用いて極めて簡潔に計算できます：
$$
h'(a) = 1 - h(a)^2 \tag{8.16}
$$
出力ユニットは線形活性化関数（$y_k = a_k$）を持ち、誤差関数として二乗和誤差を採用します：
$$
E_n = \frac{1}{2} \sum_{k=1}^K (y_k - t_k)^2 \tag{8.17}
$$

#### 1. 順伝播計算 (Eq 8.18 - 8.20)
各データ点について：
$$
a_j = \sum_{i=0}^D w_{ji}^{(1)} x_i \tag{8.18}
$$
$$
z_j = \tanh(a_j) \tag{8.19}
$$
$$
y_k = \sum_{j=0}^M w_{kj}^{(2)} z_j \tag{8.20}
$$
ここで $x_0 = 1$、$z_0 = 1$ はバイアス項に対応します。

#### 2. 誤差の逆伝播計算 (Eq 8.21 - 8.23)
出力ユニットの誤差信号は：
$$
\delta_k = y_k - t_k \tag{8.21}
$$
隠れユニットの誤差信号は、式 (8.13) と式 (8.16) より：
$$
\delta_j = (1 - z_j^2) \sum_{k=1}^K w_{kj}^{(2)} \delta_k \tag{8.22}
$$
第1層および第2層の重みに関する誤差の導関数は：
$$
\frac{\partial E_n}{\partial w_{ji}^{(1)}} = \delta_j x_i, \quad \frac{\partial E_n}{\partial w_{kj}^{(2)}} = \delta_k z_j \tag{8.23}
$$"""))

    # Cell 8: Section 8.1.3 Code
    cells.append(create_cell("code", r"""# 8.1.3: TwoLayerMLP による順伝播と解析的逆伝播の実装と動作確認
D, M, K = 4, 5, 2
mlp = TwoLayerMLP(D=D, M=M, K=K, seed=42)

# 合成バッチデータ (N=8) の生成
N = 8
rng = np.random.default_rng(42)
X_batch = rng.normal(0.0, 1.0, (N, D))
T_batch = rng.normal(0.0, 1.0, (N, K))

# 順伝播
A1, Z, A2, Y = mlp.forward(X_batch)
error_val = mlp.compute_error(X_batch, T_batch)

# 逆伝播
g_W1, g_b1, g_W2, g_b2 = mlp.backward(X_batch, T_batch)
grad_vec = mlp.get_gradient_vector(X_batch, T_batch)

print(f"ネットワーク構造: 入力 {D} -> 隠れ {M} (tanh) -> 出力 {K} (linear)")
print(f"総パラメータ数 W: {mlp.num_parameters} (W1:{mlp.W1.size}, b1:{mlp.b1.size}, W2:{mlp.W2.size}, b2:{mlp.b2.size})")
print(f"バッチ二乗和誤差 E: {error_val:.6f}")
print(f"勾配ベクトルのノルム ||grad||: {np.linalg.norm(grad_vec):.6f}")
print(f"g_W1 形状: {g_W1.shape}, g_W2 形状: {g_W2.shape}")
"""))

    # Cell 9: Section 8.1.4 Markdown
    cells.append(create_cell("markdown", r"""---
### 8.1.4 数値微分 (Numerical Differentiation)

誤差逆伝播法の最も重要な特性は、その**計算効率（Computational Efficiency）**にあります。ネットワーク全体の重みとバイアスの総数を $W$ としたとき、計算回数がどのようにスケールするかを検証します。

1. **誤差逆伝播法の計算量 $\mathcal{O}(W)$**:
   1回の順伝播において、重み付き和の計算コストは結合数に比例するため $\mathcal{O}(W)$ です。逆伝播においても式 (8.13) の和の計算コストは同様に $\mathcal{O}(W)$ です。したがって、**全 $W$ 個のパラメータに対する勾配をわずか $\mathcal{O}(W)$ の計算量で一度に評価**できます。

2. **片側有限差分近似 (Forward Finite Differences, Eq 8.24)**:
   誤差逆伝播法を用いずに、各パラメータ $w_{ji}$ を微小量 $\epsilon$ だけ変化させて誤差の変化を測定する数値微分法：
   $$
   \frac{\partial E_n}{\partial w_{ji}} = \frac{E_n(w_{ji} + \epsilon) - E_n(w_{ji})}{\epsilon} + \mathcal{O}(\epsilon) \tag{8.24}
   $$
   この方法では、各パラメータごとに1回の順伝播が必要となるため、全体として **$\mathcal{O}(W^2)$ の計算量**を要します。

3. **対称中心差分近似 (Symmetrical Central Differences, Eq 8.25)**:
   対称的な中心差分を用いると、テイラー展開の1次の補正項が相殺し、打ち切り誤差が2次に向上します：
   $$
   \frac{\partial E_n}{\partial w_{ji}} = \frac{E_n(w_{ji} + \epsilon) - E_n(w_{ji} - \epsilon)}{2\epsilon} + \mathcal{O}(\epsilon^2) \tag{8.25}
   $$
   計算量は片側差分の約2倍になりますが、精度が劇的に改善します。

#### 打ち切り誤差と丸め誤差のトレードオフ (Truncation vs Round-off Error)
Figure 8.2 は、ステップサイズ $\epsilon$ に対する数値微分の誤差（解析解との差）を両対数プロットで示したものです：
- **大きな $\epsilon$ 側**: 理論的な打ち切り誤差が支配的です。両対数軸において、有限差分（赤線）は傾き $1$（$\mathcal{O}(\epsilon)$）、中心差分（青線）は傾き $2$（$\mathcal{O}(\epsilon^2)$）の直線となります。
- **小さな $\epsilon$ 側**: コンピュータの浮動小数点数の桁落ち・丸め誤差（Round-off Error $\approx \eta / \epsilon$）が急激に増大し、ノイズの多い V字型の谷を形成します。
- **実用上の意義**: 数値微分は訓練の実行には計算量的に不適ですが、自作の誤差逆伝播や自動微分の実装が正しいかを検証する**勾配チェック（Gradient Checking）**の黄金律として不可欠です。"""))

    # Cell 10: Figure 8.2 Code & Gradient Checking
    cells.append(create_cell("code", r"""# Figure 8.2: 数値微分のステップサイズ epsilon と誤差の依存性
fig_8_2 = generate_figure_8_2(save_both=True)
plt.show()

# 実際の TwoLayerMLP における勾配チェック (Gradient Checking) の実演
eps = 1e-5
grad_ana = mlp.get_gradient_vector(X_batch, T_batch)
grad_num_fd = numerical_gradient_finite_diff(mlp, X_batch, T_batch, epsilon=eps)
grad_num_cd = numerical_gradient_central_diff(mlp, X_batch, T_batch, epsilon=eps)

rel_err_fd = np.linalg.norm(grad_num_fd - grad_ana) / (np.linalg.norm(grad_num_fd) + np.linalg.norm(grad_ana))
rel_err_cd = np.linalg.norm(grad_num_cd - grad_ana) / (np.linalg.norm(grad_num_cd) + np.linalg.norm(grad_ana))

print("=== 勾配チェック (Gradient Checking) の結果 ===")
print(f"ステップサイズ epsilon: {eps}")
print(f"片側有限差分の相対誤差 (O(eps)):   {rel_err_fd:.3e}")
print(f"中心差分の相対誤差   (O(eps^2)): {rel_err_cd:.3e}")
assert rel_err_cd < 1e-8, "中心差分勾配チェックに失敗しました！"
print("中心差分による相対誤差が 10^-8 未満であり、解析的逆伝播の実装の正確性が完全に検証されました。")
"""))

    # Cell 11: Section 8.1.5 Markdown
    cells.append(create_cell("markdown", r"""---
### 8.1.5 ヤコビ行列 (The Jacobian Matrix)

誤差逆伝播法の技法は、誤差関数のパラメータに関する微分だけでなく、ネットワークの**入力に対する出力の感度を表すヤコビ行列（Jacobian Matrix）**の評価にも応用できます。
ヤコビ行列の各要素は、他の入力を固定したときの入力 $x_i$ に対する出力 $y_k$ の偏微分として定義されます：
$$
J_{ki} \equiv \frac{\partial y_k}{\partial x_i} \tag{8.26}
$$

#### 1. モジュール型深層学習アーキテクチャにおける役割 (Eq 8.27)
Figure 8.3 に示すように、微分可能な複数のモジュールを接続して構築された深層学習システムを考えます。あるモジュールの内部パラメータ $w$ に関してシステム全体の誤差 $E$ を最小化する場合、合成関数の微分律より：
$$
\frac{\partial E}{\partial w} = \sum_{k, j} \frac{\partial E}{\partial y_k} \frac{\partial y_k}{\partial z_j} \frac{\partial z_j}{\partial w} \tag{8.27}
$$
右辺の中央に現れる $\frac{\partial y_k}{\partial z_j}$ こそが、中間モジュールのヤコビ行列です。ヤコビ行列を介して、出力側の誤差信号 $\frac{\partial E}{\partial y_k}$ が手前のモジュールへと逆伝播されます。

#### 2. 入力誤差の伝播と局所感度解析 (Eq 8.28)
入力変数に微小な測定ノイズや誤差 $\Delta x_i$ が含まれている場合、出力に生じる変動 $\Delta y_k$ はヤコビ行列を通じて1次近似されます：
$$
\Delta y_k \simeq \sum_i \frac{\partial y_k}{\partial x_i} \Delta x_i \tag{8.28}
$$

#### 3. ヤコビ行列の逆伝播計算法 (Eq 8.29 - 8.34)
ヤコビ行列の要素 $J_{ki}$ は、連鎖律を用いて次のように展開されます：
$$
J_{ki} = \frac{\partial y_k}{\partial x_i} = \sum_j \frac{\partial y_k}{\partial a_j} \frac{\partial a_j}{\partial x_i} = \sum_j w_{ji} \frac{\partial y_k}{\partial a_j} \tag{8.29}
$$
ここで $\frac{\partial y_k}{\partial a_j}$ は、上位ユニット $l$ からの逆伝播漸化式を満たします：
$$
\frac{\partial y_k}{\partial a_j} = h'(a_j) \sum_l w_{lj} \frac{\partial y_k}{\partial a_l} \tag{8.30}
$$
境界条件として、出力ユニットが線形であれば $\frac{\partial y_k}{\partial a_l} = I_{kl}$（単位行列）、シグモイドであれば $I_{kl}\sigma'(a_l)$、ソフトマックスであれば $I_{kl} y_k - y_k y_l$ となります。

特に2層MLP（$\tanh$ 隠れ層＋線形出力層）の場合、ヤコビ行列を行列形式で表すと極めて明快になります：
$$
\mathbf{J} = \mathbf{W}^{(2)} \operatorname{diag}(1 - \mathbf{z}^2) \mathbf{W}^{(1)}
$$"""))

    # Cell 12: Figure 8.3 & Jacobian Code
    cells.append(create_cell("code", r"""# Figure 8.3: モジュール型アーキテクチャにおける誤差伝播とヤコビ行列
fig_8_3 = generate_figure_8_3(save_both=True)
plt.show()

# 2層MLP におけるヤコビ行列の解析解と数値解 (Eq 8.35) の比較
x_test = np.array([0.5, -0.2, 0.8, -0.4])  # D=4
J_analytical = compute_jacobian_analytical(mlp, x_test)
J_numerical = compute_jacobian_numerical(mlp, x_test, epsilon=1e-5)

print(f"入力 x: {x_test}")
print(f"解析的ヤコビ行列 J (形状 {J_analytical.shape}):\n{np.round(J_analytical, 5)}")
print(f"数値的ヤコビ行列 J_num:\n{np.round(J_numerical, 5)}")

diff_J = np.linalg.norm(J_analytical - J_numerical)
print(f"ヤコビ行列の絶対誤差ノルム: {diff_J:.3e}")
assert diff_J < 1e-6, "ヤコビ行列の数値解と解析解が一致しません！"

# 感度解析の検証 (Eq 8.28: Delta y approx J Delta x)
delta_x = np.array([1e-4, -2e-4, 1.5e-4, -0.5e-4])
expected_delta_y = J_analytical @ delta_x

y_base = mlp.predict(x_test)[0]
y_perturbed = mlp.predict(x_test + delta_x)[0]
actual_delta_y = y_perturbed - y_base

print(f"\n摂動 Delta x: {delta_x}")
print(f"ヤコビ行列による予測 Delta y: {np.round(expected_delta_y, 8)}")
print(f"実際の出力変化 Delta y:       {np.round(actual_delta_y, 8)}")
np.testing.assert_allclose(actual_delta_y, expected_delta_y, rtol=1e-3)
print("1次テイラー展開 Delta y ≈ J Delta x の成立を確認しました。")
"""))

    # Cell 13: Section 8.1.6 Markdown
    cells.append(create_cell("markdown", r"""---
### 8.1.6 ヘッセ行列 (The Hessian Matrix)

1階導関数である勾配ベクトル $\nabla E$ に加え、2階偏微分からなる**ヘッセ行列（Hessian Matrix）** $\mathbf{H}$ を評価することも重要です：
$$
H_{ij} \equiv \frac{\partial^2 E}{\partial w_i \partial w_j} \tag{8.37}
$$
パラメータ数を $W$ とすると、ヘッセ行列のサイズは $W \times W$ です。
ヘッセ行列は、Newton-Raphson法などの2次最適化手法、ベイズ的ニューラルネットワークにおけるラプラス近似（MacKay 1992, Bishop 2006）、および大規模言語モデル（LLM）の重み量子化・刈り込み（Shen et al. 2019）など多岐にわたる応用を持ちます。

#### 1. 計算量と近似の必要性
- 厳密なヘッセ行列の要素数は $W^2$ 個であり、その計算量は $\mathcal{O}(W^2)$ です。
- ニュートン法で必要となるヘッセ逆行列の計算は $\mathcal{O}(W^3)$ を要します。数百万〜数千億パラメータを持つ現代の深層学習において、完全なヘッセ行列の保持や逆行列計算は不可能です。
- **対角近似 (Diagonal Approximation)**: 非対角成分を無視して対角成分のみを保持・反転する手法ですが、実際のヘッセ行列には強い非対角相関が存在するため限界があります。

#### 2. 外積近似 / Gauss-Newton近似 (Outer Product Approximation, Eq 8.39, 8.40)
二乗和誤差 $E = \frac{1}{2} \sum_{n=1}^N (y_n - t_n)^2$ の2階微分を厳密に計算すると：
$$
\mathbf{H} = \nabla\nabla E = \sum_{n=1}^N \nabla y_n (\nabla y_n)^T + \sum_{n=1}^N (y_n - t_n) \nabla\nabla y_n \tag{8.39}
$$
ネットワークが十分に訓練されて出力 $y_n$ が目標値 $t_n$ に近づいた場合、第2項は無視できます。また、第4章（4.2節）で見たように最適関数が条件付平均であるため、残差 $(y_n - t_n)$ は平均ゼロの確率変数とみなせ、第2階微分項と無相関であれば総和において第2項は打ち消し合います。

したがって、第2項を省略することで**外積近似（Levenberg-Marquardt近似 / Gauss-Newton近似）**が得られます：
$$
\mathbf{H} \simeq \sum_{n=1}^N \nabla a_n (\nabla a_n)^T \tag{8.40}
$$
二値分類の交差エントロピー誤差に対しては：
$$
\mathbf{H} \simeq \sum_{n=1}^N y_n (1 - y_n) \nabla a_n (\nabla a_n)^T \tag{8.41}
$$
**Gauss-Newton近似の絶大な利点**:
1. 1階の勾配情報 $\nabla y_n$ のみを用いて構築可能（通常の逆伝播で各点 $\mathcal{O}(W)$ で計算可能）。
2. ベクトルの外積の和であるため、**常に半正定値（Positive Semi-Definite）**であり、負の固有値（鞍点や最大値への発散）を原理的に回避できる。

#### 3. 高速ヘッセ・ベクトル積 (Fast Hessian-Vector Products, Pearlmutter 1994)
完全なヘッセ行列そのものは必要とせず、任意のベクトル $\mathbf{v}$ に対する積 $\mathbf{H}\mathbf{v}$ のみが求められる場面（共役勾配法やトロン（TRON）法など）では、有限差分方向微分を利用して**ヘッセ行列を陽に構築することなく $\mathcal{O}(W)$ の計算量**で計算できます：
$$
\mathbf{H}\mathbf{v} \simeq \frac{\nabla E(\mathbf{w} + \epsilon \mathbf{v}) - \nabla E(\mathbf{w} - \epsilon \mathbf{v})}{2\epsilon}
$$"""))

    # Cell 14: Section 8.1.6 Code
    cells.append(create_cell("code", r"""# 8.1.6: ヘッセ行列の厳密解、Gauss-Newton外積近似、および高速 Hv 積の実装検証
# 1. 厳密な数値ヘッセ行列の計算
H_exact = compute_exact_hessian_numerical(mlp, X_batch, T_batch, epsilon=1e-5)

# 2. Gauss-Newton 外積近似ヘッセ行列の計算 (Eq 8.40)
H_gn = compute_hessian_outer_product(mlp, X_batch)

# 3. 性質検証: 対称性と半正定値性
sym_err_exact = np.linalg.norm(H_exact - H_exact.T)
sym_err_gn = np.linalg.norm(H_gn - H_gn.T)

eigvals_exact = np.linalg.eigvalsh(H_exact)
eigvals_gn = np.linalg.eigvalsh(H_gn)

print(f"ヘッセ行列の次元: {H_gn.shape[0]} x {H_gn.shape[1]}")
print(f"Gauss-Newtonヘッセ行列の対称性誤差: {sym_err_gn:.2e}")
print(f"Gauss-Newtonヘッセ行列の最小固有値: {np.min(eigvals_gn):.6e}")
assert np.min(eigvals_gn) >= -1e-10, "Gauss-Newtonヘッセ行列は半正定値でなければなりません！"
print("Gauss-Newtonヘッセ行列が半正定値 (すべての固有値 >= 0) であることを確認しました。")

# 4. Pearlmutter 高速 Hv 積の検証
v = rng.normal(0.0, 1.0, mlp.num_parameters)
v = v / np.linalg.norm(v)

Hv_fast = hessian_vector_product_fd(mlp, X_batch, T_batch, v, epsilon=1e-5)
Hv_from_exact = H_exact @ v

diff_Hv = np.linalg.norm(Hv_fast - Hv_from_exact) / np.linalg.norm(Hv_from_exact)
print(f"高速 Hv 積 (O(W)) と明示的 H@v (O(W^2)) の相対誤差: {diff_Hv:.3e}")
assert diff_Hv < 1e-4, "高速 Hv 積と厳密なヘッセ行列との積が一致しません！"
print("ヘッセ行列を陽に保持することなく O(W) で Hv 積が正確に得られることを実証しました。")
"""))

    # Cell 15: Summary & Next Chapter Connection
    cells.append(create_cell("markdown", r"""---
## 本節のまとめと次節（8.2節）への接続

### 1. 本節で学んだ核心事項
1. **誤差逆伝播法の計算量優位性**:
   - 順伝播で中間状態を保存し、出力層から入力層へ連鎖律を適用して誤差信号 $\delta$ を逆伝播させることで、パラメータ数 $W$ に対して線形な計算量 $\mathcal{O}(W)$ で全勾配を評価できる。
   - 数値微分（有限差分・中心差分）は $\mathcal{O}(W^2)$ を要するが、勾配チェックという実装検証において極めて強力なツールとなる。
2. **ヤコビ行列とモジュール構造**:
   - ヤコビ行列 $J_{ki} = \frac{\partial y_k}{\partial x_i}$ も逆伝播形式により効率的に計算可能であり、モジュール間の誤差伝播や局所感度解析の基礎を与える。
3. **ヘッセ行列と外積近似**:
   - 2階微分の完全な評価は $\mathcal{O}(W^2)$、逆行列は $\mathcal{O}(W^3)$ を要するため、1階勾配の外積からなる半正定値なGauss-Newton近似や、メモリ消費なしに $\mathcal{O}(W)$ で計算できる高速ヘッセ・ベクトル積（Pearlmutterの手法）が実用上決定的な役割を果たす。

---

### 2. 次節（8.2 自動微分 Automatic Differentiation）への展開
本節で導出した誤差逆伝播法は、手計算で数式を導出し、順伝播コードと逆伝播コードを個別に記述する伝統的なアプローチでした。しかし、この手法はモデルの構造変更のたびに数式の再導出と実装が必要となり、ヒューマンエラーを誘発します。
次節（8.2節）では、現代の深層学習フレームワーク（PyTorch, JAX, TensorFlow）の心臓部である**自動微分（Automatic Differentiation: AD）**の理論体系、すなわち**前方向モード（Forward Mode: 双対数と接線トレース）**と**逆方向モード（Reverse Mode: 計算グラフと随伴変数）**の数理的探求へと進みます。"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    return notebook


def main():
    root = _get_project_root() if "_get_project_root" in globals() else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "8")
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, "8.1_Evaluation_of_Gradients.ipynb")

    print(f"ノートブック生成開始: {nb_path}")
    nb = build_cells()

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"ノートブック書き込み完了 ({len(nb['cells'])} セル)")

    # Execute notebook using nbconvert
    print("ノートブックの実行開始 (jupyter nbconvert --execute)...")
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        nb_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("実行時エラー発生:")
        print(res.stderr)
        raise RuntimeError(f"Notebook execution failed with code {res.returncode}")
    else:
        print("すべてのセルが正常に実行完了しました (エラーゼロ)。")


if __name__ == "__main__":
    main()

"""
scripts/build_ch10_exercises_notebook.py
========================================
Builds and executes 10/10_Exercises.ipynb.
Bishop & Bishop (2024), Chapter 10: Convolutional Networks, Exercises 10.1 - 10.13 (all 13 exercises).
"""

import os
import sys
import subprocess
from pathlib import Path
import nbformat as nbf


def create_cell(cell_type: str, source: str):
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(source)
    elif cell_type == "code":
        return nbf.v4.new_code_cell(source)
    raise ValueError(f"Unknown cell type: {cell_type}")


def build_cells():
    cells = []

    # Title & Overview
    title_md = r"""# 第10章 畳み込みネットワーク (Convolutional Networks)
## 章末演習問題 (Exercises 10.1 〜 10.13)

本ノートブックは、Christopher M. Bishop & Hugh Bishop (2024)『深層学習：基礎と概念 (Deep Learning: Foundations and Concepts)』第10章「畳み込みネットワーク」の**章末演習問題 全13問 (Exercises 10.1 〜 10.13)** に対する完全解答・数学的厳密導出・自己検証コードです。

---

### 演習問題 目次
1. **Exercise 10.1 (★)**: 単位ノルム制約下での線形射影最大化 ($\mathbf{x}^* = \mathbf{w} / \|\mathbf{w}\|$)
2. **Exercise 10.2 (★★)**: 1次元畳み込みのテプリッツ (Toeplitz) 行列積表現 ($\mathbf{y} = \mathbf{A}\mathbf{x}$)
3. **Exercise 10.3 (★★)**: 2次元相互相関と2次元畳み込みの等価性（180度反転カーネル $\mathbf{W}'_{mn} = \mathbf{W}_{-m, -n}$）
4. **Exercise 10.4 (★★)**: 畳み込み層におけるバッチ正規化（Batch Normalization）の統計量共有と空間並進等変性（Translation Equivariance）
5. **Exercise 10.5 (★)**: 連続1次元畳み込みの交換法則（Commutativity）の厳密証明
6. **Exercise 10.6 (★)**: 奇数サイズフィルタに対する同サイズ維持パディング公式 ($P = (M - 1) / 2$)
7. **Exercise 10.7 (★★)**: ストライド畳み込みにおける出力解像度公式と剰余画素の取り扱い
8. **Exercise 10.8 (★★)**: VGG-16 における $3 \times 3$ 畳み込み積層によるパラメータ数および MACs 削減効果の解析
9. **Exercise 10.9 (★★)**: 階数1フィルタの可分畳み込み (Separable 2D Convolution) 分解と計算量削減 ($M^2 \to 2M$)
10. **Exercise 10.10 (★)**: DeepDream 目的関数の事前活性化に関する勾配と活性化値の厳密一致証明
11. **Exercise 10.11 (★★)**: 物体検出におけるクラス確率の2つの定式化（単一 $(K+1)$ ソフトマックス vs 物体性シグモイド＋条件付きソフトマックス）
12. **Exercise 10.12 (★★★)**: スライディングウィンドウ探索と全結合畳み込み高速化の乗算回数比較 (1,656 回 vs 504 回)
13. **Exercise 10.13 (★★★)**: 転置畳み込みとダウンサンプリング線形写像の転置行列 $\mathbf{A}^\top$ 双対性の厳密証明
"""
    cells.append(create_cell("markdown", title_md))

    # Cell 1: Environment Setup
    code_setup = r"""# 環境セットアップと共通モジュールのインポート
import os
import sys
from pathlib import Path
import numpy as np

# リポジトリルートをパスに追加
repo_root = Path.cwd().parent if Path.cwd().name == "10" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.exercises_ch10 import (
    solve_exercise_10_1,
    solve_exercise_10_2,
    solve_exercise_10_3,
    solve_exercise_10_4,
    solve_exercise_10_5,
    solve_exercise_10_6,
    solve_exercise_10_7,
    solve_exercise_10_8,
    solve_exercise_10_9,
    solve_exercise_10_10,
    solve_exercise_10_11,
    solve_exercise_10_12,
    solve_exercise_10_13,
)

print("Setup completed successfully. Chapter 10 exercise solvers loaded.")
"""
    cells.append(create_cell("code", code_setup))

    # Exercise 10.1
    ex10_1_md = r"""---
### Exercise 10.1 (★): 単位ノルム制約下での線形射影最大化

#### 問題設定
線形ユニット $y = \mathbf{w}^\top \mathbf{x}$ を考える。ここで $\mathbf{w}$ は固定された重みベクトルであり、入力ベクトル $\mathbf{x}$ は単位ノルム制約 $\|\mathbf{x}\| = 1$ を満たすものとする。
$y$ を最大化する入力ベクトル $\mathbf{x}$ は $\mathbf{w}$ と平行であり、$\mathbf{x}^* = \frac{\mathbf{w}}{\|\mathbf{w}\|}$ で与えられ、その最大値は $y^* = \|\mathbf{w}\|$ であることを示せ。

#### 数学的証明
1. **コーシー・シュワルツの不等式による直接証明**:
   内積の幾何学的定義より、$\mathbf{w}$ と $\mathbf{x}$ のなす角を $\theta$ とすると：
   $$
   y = \mathbf{w}^\top \mathbf{x} = \|\mathbf{w}\| \|\mathbf{x}\| \cos \theta = \|\mathbf{w}\| \cos \theta
   $$
   $-1 \le \cos \theta \le 1$ より、最大値は $\cos \theta = 1$ のときに達成される。
   $\cos \theta = 1$ は $\mathbf{x}$ が $\mathbf{w}$ と同方向（平行）であることを意味し、$\|\mathbf{x}\| = 1$ であるから：
   $$
   \mathbf{x}^* = \frac{\mathbf{w}}{\|\mathbf{w}\|}, \quad y^* = \mathbf{w}^\top \left( \frac{\mathbf{w}}{\|\mathbf{w}\|} \right) = \frac{\|\mathbf{w}\|^2}{\|\mathbf{w}\|} = \|\mathbf{w}\|
   $$

2. **ラグランジュの未定乗数法による証明**:
   制約 $\|\mathbf{x}\|^2 = 1$ のもとでの最大化ラグランジアン：
   $$
   \mathcal{L}(\mathbf{x}, \lambda) = \mathbf{w}^\top \mathbf{x} - \frac{\lambda}{2} (\|\mathbf{x}\|^2 - 1)
   $$
   $\mathbf{x}$ について停留条件を求めると：
   $$
   \nabla_{\mathbf{x}} \mathcal{L} = \mathbf{w} - \lambda \mathbf{x} = \mathbf{0} \implies \mathbf{x} = \frac{\mathbf{w}}{\lambda}
   $$
   ノルム制約 $\|\mathbf{x}\| = 1$ より、$\|\frac{\mathbf{w}}{\lambda}\| = \frac{\|\mathbf{w}\|}{|\lambda|} = 1 \implies \lambda = \pm \|\mathbf{w}\|$。
   最大値を与えるのは正の解 $\lambda = \|\mathbf{w}\|$ であり、$\mathbf{x}^* = \frac{\mathbf{w}}{\|\mathbf{w}\|}$、$y^* = \|\mathbf{w}\|$ となる。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_1_md))

    code_10_1 = r"""# Exercise 10.1 自己採点・数値検証
w = np.array([3.0, 4.0])
res_10_1 = solve_exercise_10_1(w)

assert res_10_1["norm_w"] == 5.0
assert res_10_1["max_y"] == 5.0
np.testing.assert_allclose(res_10_1["optimal_x"], [0.6, 0.8])
assert np.linalg.norm(res_10_1["optimal_x"]) == 1.0

# ランダムな任意の単位ベクトルと比較し、最大値であることを確認
rng = np.random.default_rng(101)
for _ in range(100):
    v = rng.normal(0, 1, 2)
    v /= np.linalg.norm(v)
    assert np.dot(w, v) <= res_10_1["max_y"] + 1e-12

print("✓ Exercise 10.1: Passed successfully (x* = w / ||w|| confirmed as global maximum).")
"""
    cells.append(create_cell("code", code_10_1))

    # Exercise 10.2
    ex10_2_md = r"""---
### Exercise 10.2 (★★): 1次元畳み込みのテプリッツ行列積表現

#### 問題設定
1次元入力ベクトル $\mathbf{x} = (x_1, \dots, x_D)^\top$ および長さ $M$ のフィルタ $\mathbf{w} = (w_0, \dots, w_{M-1})^\top$ を考える。
有効畳み込み（Valid Convolution, パディングなし）の出力ベクトル $\mathbf{y} = (y_1, \dots, y_{D - M + 1})^\top$ が行列積 $\mathbf{y} = \mathbf{A}\mathbf{x}$ として書けることを示し、行列 $\mathbf{A}$ の構造とサイズを求めよ。

#### 数学的導出
有効畳み込みの各出力要素 $y_i$ ($i = 1, \dots, D - M + 1$) は次式で定義されます：
$$
y_i = \sum_{m=0}^{M-1} w_m x_{i + m} = w_0 x_i + w_1 x_{i+1} + \dots + w_{M-1} x_{i + M - 1}
$$
これは、行列 $\mathbf{A}$ の第 $i$ 行の第 $i$ 列から第 $i + M - 1$ 列までにフィルタ係数 $(w_0, \dots, w_{M-1})$ が配置され、他の成分がすべて $0$ であるような帯行列です。
具体的には：
$$
\mathbf{A} = \begin{pmatrix}
w_0 & w_1 & \cdots & w_{M-1} & 0 & \cdots & 0 \\
0 & w_0 & w_1 & \cdots & w_{M-1} & \cdots & 0 \\
\vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & w_0 & w_1 & \cdots & w_{M-1}
\end{pmatrix} \in \mathbb{R}^{(D - M + 1) \times D}
$$
この行列は、対角線に沿って同じ要素が並ぶ**テプリッツ行列 (Toeplitz Matrix)** であり、畳み込みが線形演算子であることを明瞭に示しています。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_2_md))

    code_10_2 = r"""# Exercise 10.2 自己採点・数値検証
D = 6
w = np.array([1.5, -2.0, 0.5])  # M = 3
res_10_2 = solve_exercise_10_2(D, w)

A = res_10_2["matrix_A"]
assert A.shape == (4, 6)

# ランダムな x で行列積とダイレクト畳み込みの一致を検証
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y_mat = A @ x
y_direct = np.array([
    1.5 * x[0] - 2.0 * x[1] + 0.5 * x[2],
    1.5 * x[1] - 2.0 * x[2] + 0.5 * x[3],
    1.5 * x[2] - 2.0 * x[3] + 0.5 * x[4],
    1.5 * x[3] - 2.0 * x[4] + 0.5 * x[5],
])
np.testing.assert_allclose(y_mat, y_direct)
print("✓ Exercise 10.2: Passed successfully (Toeplitz matrix multiplication matches 1D convolution).")
"""
    cells.append(create_cell("code", code_10_2))

    # Exercise 10.3
    ex10_3_md = r"""---
### Exercise 10.3 (★★): 2次元相互相関と2次元畳み込みの等価性

#### 問題設定
画像処理や深層学習における2次元相互相関 (Cross-Correlation) は次のように定義される：
$$
(W \star X)_{ij} = \sum_m \sum_n W_{mn} X_{i+m, j+n}
$$
一方、数学的な2次元畳み込み (Convolution) は次のように定義される：
$$
(W * X)_{ij} = \sum_m \sum_n W_{mn} X_{i-m, j-n}
$$
フィルタ $W$ を水平・垂直両方向に反転させたフィルタ $W'_{mn} = W_{-m, -n}$（すなわち $180^\circ$ 回転させたフィルタ）を用いると、畳み込みと相互相関が等価になることを示せ。

#### 数学的証明
畳み込みの定義において、フィルタのインデックスを $m' = -m, n' = -n$ と変数変換する。
総和はすべての $m, n$ にわたるため、$m', n'$ の走査範囲も同一となる：
$$
(W * X)_{ij} = \sum_{m'} \sum_{n'} W_{-m', -n'} X_{i+m', j+n'}
$$
ここで、反転フィルタ $W'$ の定義 $W'_{m', n'} = W_{-m', -n'}$ を代入すると：
$$
(W * X)_{ij} = \sum_{m'} \sum_{n'} W'_{m', n'} X_{i+m', j+n'} = (W' \star X)_{ij}
$$
逆に、フィルタ $W$ による相互相関は、反転フィルタ $W'$ による畳み込みと完全に一致する：
$$
(W \star X)_{ij} = (W' * X)_{ij}
$$
深層学習フレームワーク（PyTorch, TensorFlow 等）の `Conv2d` レイヤは実装上「相互相関」を実行しているが、フィルタの重みは学習パラメータであり最適化によって自動的に学習されるため、事前に反転させておく必要はなく、数学的畳み込みと表現能力上全く同等である。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_3_md))

    code_10_3 = r"""# Exercise 10.3 自己採点・数値検証
rng = np.random.default_rng(103)
X = rng.normal(0, 1, (8, 8))
W = rng.normal(0, 1, (3, 3))
res_10_3 = solve_exercise_10_3(X, W)

assert res_10_3["is_equivalent"] is True
np.testing.assert_allclose(res_10_3["cross_correlation"], res_10_3["convolution_with_flipped"])
print("✓ Exercise 10.3: Passed successfully (Cross-correlation equals convolution with 180-degree flipped filter).")
"""
    cells.append(create_cell("code", code_10_3))

    # Exercise 10.4
    ex10_4_md = r"""---
### Exercise 10.4 (★★): 畳み込み層におけるバッチ正規化と空間並進等変性

#### 問題設定
全結合層におけるバッチ正規化 (Batch Normalization) は各ユニットごとにミニバッチ $N$ サンプルにわたる平均 $\mu_k$ と分散 $\sigma_k^2$ を計算する。
しかし畳み込み層では、各特徴マップ（チャネル）$k$ に対し、バッチサイズ $N$ だけでなく空間解像度 $H \times W$ にわたる合計 $N \times H \times W$ 個の活性化値から単一の平均 $\mu_k$ と分散 $\sigma_k^2$ を計算する。この設計が必要とされる理由を**並進等変性 (Translation Equivariance)** の観点から論ぜよ。

#### 理論解説と導出
1. **畳み込みの並進等変性**:
   畳み込み層の核となる帰納バイアスは「重み共有」による空間並進等変性です。
   入力画像中の特徴（例：輪郭、角）が $(\Delta x, \Delta y)$ だけ平行移動すると、出力特徴マップ上の活性化も全く同じ量だけ平行移動します。
2. **位置ごとの正規化がもたらす破綻**:
   もし各空間位置 $(i, j)$ ごとに異なる平均 $\mu_{ijk}$ と分散 $\sigma_{ijk}^2$ を用いて正規化してしまうと、画像の中央にあるときと左上にあるときで異なるスケール変換・シフト変換を受けることになります。
   その結果、出力特徴が位置依存となり、**畳み込みが本来持つ並進等変性が完全に破壊**されてしまいます。
3. **結論**:
   特徴マップ全体にわたって一貫した特徴検出器としての意味を保つため、統計量 $\mu_k, \sigma_k^2$ および学習パラメータ（スケール $\gamma_k$、シフト $\beta_k$）は、チャネル $k$ の全空間位置 $(i, j)$ で共有されなければならない。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_4_md))

    code_10_4 = r"""# Exercise 10.4 自己採点・数値検証
res_10_4 = solve_exercise_10_4()

assert "equivariance" in res_10_4["explanation"].lower()
assert res_10_4["independent_dimensions"] == ("channels",)
assert res_10_4["shared_dimensions"] == ("batch_size", "height", "width")

# 4D テンソル (N, H, W, C) での統計量計算の次元確認
rng = np.random.default_rng(104)
tensor = rng.normal(5.0, 2.0, size=(16, 28, 28, 8))
# チャネルごとの平均: 軸 (0, 1, 2) について集約
channel_means = np.mean(tensor, axis=(0, 1, 2))
assert channel_means.shape == (8,)
np.testing.assert_allclose(channel_means, np.full(8, 5.0), atol=0.1)
print("✓ Exercise 10.4: Passed successfully (Channel-wise statistics preserve spatial equivariance).")
"""
    cells.append(create_cell("code", code_10_4))

    # Exercise 10.5
    ex10_5_md = r"""---
### Exercise 10.5 (★): 連続1次元畳み込みの交換法則の証明

#### 問題設定
連続関数 $f(t)$ と $g(t)$ の畳み込みは次のように定義される：
$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
$$
この畳み込み演算が交換法則 $(f * g)(t) = (g * f)(t)$ を満たすことを証明せよ。

#### 数学的証明
積分変数 $\tau$ を $u = t - \tau$ と置換する。
このとき：
$$
\tau = t - u, \quad d\tau = -du
$$
積分の上下限の変化：
- $\tau \to -\infty$ のとき $u = t - (-\infty) = +\infty$
- $\tau \to +\infty$ のとき $u = t - (+\infty) = -\infty$

これを積分に代入すると：
$$
(f * g)(t) = \int_{+\infty}^{-\infty} f(t - u) g(u) (-du)
$$
積分区間の反転により負符号が相殺される：
$$
(f * g)(t) = \int_{-\infty}^{+\infty} g(u) f(t - u) du
$$
これはまさに定義より $(g * f)(t)$ である。したがって、$(f * g)(t) = (g * f)(t)$ が成立する。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_5_md))

    code_10_5 = r"""# Exercise 10.5 自己採点・数値検証
t = np.linspace(0, 3, 60)
f = np.exp(-t)
g = np.cos(2 * np.pi * t)
res_10_5 = solve_exercise_10_5(f, g, dt=3.0 / 60)

assert res_10_5["is_commutative"] is True
assert res_10_5["max_diff"] < 1e-12
print(f"✓ Exercise 10.5: Passed successfully (Commutativity verified, max discrepancy = {res_10_5['max_diff']:.2e}).")
"""
    cells.append(create_cell("code", code_10_5))

    # Exercise 10.6
    ex10_6_md = r"""---
### Exercise 10.6 (★): 同サイズ維持パディング (Same-Padding) 公式の導出

#### 問題設定
幅 $W$ の入力画像に対し、奇数の幅 $M$ を持つフィルタを用いてストライド $S=1$ で畳み込みを行う。
出力画像の幅が入力と同じ $W$ となるようなゼロパディング幅 $P$ が $P = \frac{M - 1}{2}$ で与えられることを示せ。

#### 数学的導出
パディング $P$、フィルタ幅 $M$、ストライド $S$ に対する出力幅の公式は：
$$
W_{\mathrm{out}} = \left\lfloor \frac{W - M + 2P}{S} \right\rfloor + 1
$$
ストライド $S=1$ の場合：
$$
W_{\mathrm{out}} = W - M + 2P + 1
$$
出力幅を入力幅と一致させたい（$W_{\mathrm{out}} = W$）ので：
$$
W - M + 2P + 1 = W \implies 2P = M - 1 \implies P = \frac{M - 1}{2}
$$
$M$ が奇数のとき、$M - 1$ は偶数であるため、$P$ は厳密に整数となる（例：$M=3 \implies P=1$、$M=5 \implies P=2$、$M=7 \implies P=3$）。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_6_md))

    code_10_6 = r"""# Exercise 10.6 自己採点・数値検証
test_filter_sizes = [1, 3, 5, 7, 9, 11]
expected_paddings = [0, 1, 2, 3, 4, 5]

for M, exp_P in zip(test_filter_sizes, expected_paddings):
    res = solve_exercise_10_6(M)
    assert res["same_padding_P"] == exp_P

print("✓ Exercise 10.6: Passed successfully (Same-padding formula P = (M - 1) / 2 verified).")
"""
    cells.append(create_cell("code", code_10_6))

    # Exercise 10.7
    ex10_7_md = r"""---
### Exercise 10.7 (★★): ストライド畳み込み出力次元公式

#### 問題設定
任意の入力幅 $W$、フィルタ幅 $M$、ストライド $S$、パディング幅 $P$ に対する出力解像度の一般公式を導出し、$(W - M + 2P)$ が $S$ で割り切れない場合に端の画素がどのように扱われるかを説明せよ。

#### 数学的導出
両側にパディング $P$ を付加した後の有効入力幅は $W + 2P$ である。
最初のフィルタ位置は左端（インデックス $0$）に置かれ、フィルタの右端は $M - 1$ に位置する。
フィルタを右に $S$ 画素ずつ進めるとき、第 $k$ 回目の移動後のフィルタ右端は $(M - 1) + kS$ となる。
フィルタがパディング後の入力境界 $(W + 2P - 1)$ を超えないための条件は：
$$
(M - 1) + kS \le W + 2P - 1 \implies kS \le W - M + 2P \implies k \le \frac{W - M + 2P}{S}
$$
最初の位置（$k=0$）を含めたステップ数、すなわち出力幅 $W_{\mathrm{out}}$ は最大の非負整数 $k$ に $1$ を足したものである：
$$
W_{\mathrm{out}} = \left\lfloor \frac{W - M + 2P}{S} \right\rfloor + 1
$$
$(W - M + 2P)$ が $S$ で割り切れない場合、剰余 $r = (W - M + 2P) \pmod S$ 個の画素が末端に残るが、フィルタの完全な受容野を形成できないため、これらの余り画素は切り捨て（Discard）される。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_7_md))

    code_10_7 = r"""# Exercise 10.7 自己採点・数値検証
# ケース1: 割り切れる場合 (W=7, M=3, S=2, P=0) -> (7 - 3) // 2 + 1 = 3
r1 = solve_exercise_10_7(W=7, M=3, S=2, P=0)
assert r1["W_out"] == 3
assert r1["has_remainder"] is False
assert r1["discarded_pixels"] == 0

# ケース2: 割り切れない場合 (W=8, M=3, S=2, P=0) -> (8 - 3) // 2 + 1 = 3 (剰余1画素が切り捨て)
r2 = solve_exercise_10_7(W=8, M=3, S=2, P=0)
assert r2["W_out"] == 3
assert r2["has_remainder"] is True
assert r2["discarded_pixels"] == 1

print("✓ Exercise 10.7: Passed successfully (Output dimension formula and remainder handling verified).")
"""
    cells.append(create_cell("code", code_10_7))

    # Exercise 10.8
    ex10_8_md = r"""---
### Exercise 10.8 (★★): VGG-16 における積層 $3 \times 3$ フィルタのパラメータ・計算量削減

#### 問題設定
チャネル数 $C$ の特徴マップに対し、実効受容野（Effective Receptive Field）を等しく保ったまま、大サイズフィルタを複数の $3 \times 3$ フィルタの積層で置き換える際のパラメータ数の削減率を求めよ：
- (a) 2層の $3 \times 3$ 畳み込み vs 1層の $5 \times 5$ 畳み込み（実効受容野 $5 \times 5$）
- (b) 3層の $3 \times 3$ 畳み込み vs 1層の $7 \times 7$ 畳み込み（実効受容野 $7 \times 7$）

#### 数学的導出
1. **(a) $5 \times 5$ 受容野の比較**:
   - 1層の $5 \times 5$ 畳み込み（入出力チャネル $C$）：
     $$
     \text{パラメータ数} = 5 \times 5 \times C \times C = 25 C^2
     $$
   - 2層の $3 \times 3$ 畳み込み（中間チャネル $C$）：
     $$
     \text{パラメータ数} = 2 \times (3 \times 3 \times C \times C) = 18 C^2
     $$
   - 削減比率：
     $$
     \frac{18 C^2}{25 C^2} = \frac{18}{25} = 0.72 \quad (\mathbf{28\% \text{ 削減}})
     $$

2. **(b) $7 \times 7$ 受容野の比較**:
   - 1層の $7 \times 7$ 畳み込み：
     $$
     \text{パラメータ数} = 7 \times 7 \times C \times C = 49 C^2
     $$
   - 3層の $3 \times 3$ 畳み込み：
     $$
     \text{パラメータ数} = 3 \times (3 \times 3 \times C \times C) = 27 C^2
     $$
   - 削減比率：
     $$
     \frac{27 C^2}{49 C^2} \approx 0.551 \quad (\mathbf{45\% \text{ 削減}})
     $$
さらに、積層により非線形活性化関数（ReLU）が複数回挿入されるため、ネットワークの表現能力（決定境界の柔軟性）が向上するという極めて大きな利点が得られます。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_8_md))

    code_10_8 = r"""# Exercise 10.8 自己採点・数値検証
res_10_8 = solve_exercise_10_8(C=128)

assert res_10_8["ratio_5x5"] == 18.0 / 25.0
assert res_10_8["ratio_7x7"] == 27.0 / 49.0

print(f"✓ Exercise 10.8: 2x(3x3) vs 1x(5x5) ratio = {res_10_8['ratio_5x5']:.3f} (28% saving)")
print(f"✓ Exercise 10.8: 3x(3x3) vs 1x(7x7) ratio = {res_10_8['ratio_7x7']:.3f} (45% saving)")
"""
    cells.append(create_cell("code", code_10_8))

    # Exercise 10.9
    ex10_9_md = r"""---
### Exercise 10.9 (★★): 階数1フィルタの可分畳み込み分解と計算量

#### 問題設定
$M \times M$ の2次元フィルタ行列 $\mathbf{W}$ が階数1（Rank-1）であるとき、2つの列ベクトル $\mathbf{u}, \mathbf{v} \in \mathbb{R}^M$ の外積 $\mathbf{W} = \mathbf{u} \mathbf{v}^\top$ として分解できる。
このとき、2次元畳み込みが水平方向の1次元畳み込みと垂直方向の1次元畳み込みの積として計算できることを示し、画素あたりの計算量が $M^2$ から $2M$ へ削減されることを示せ。

#### 数学的証明
1. **分解の導出**:
   $\mathbf{W}$ の要素は $W_{mn} = u_m v_n$ と書ける。入力画像 $\mathbf{X}$ に対する2次元畳み込み（相互相関）の画素 $(i, j)$ の出力は：
   $$
   Y_{ij} = \sum_{m=0}^{M-1} \sum_{n=0}^{M-1} W_{mn} X_{i+m, j+n} = \sum_{m=0}^{M-1} \sum_{n=0}^{M-1} u_m v_n X_{i+m, j+n}
   $$
   総和を積の形に分離する：
   $$
   Y_{ij} = \sum_{m=0}^{M-1} u_m \left( \sum_{n=0}^{M-1} v_n X_{i+m, j+n} \right)
   $$
   内側の括弧 $\tilde{X}_{i+m, j} = \sum_{n=0}^{M-1} v_n X_{i+m, j+n}$ は、各行に対する水平方向の1次元フィルタ $\mathbf{v}^\top$ の畳み込みである。
   外側の総和 $Y_{ij} = \sum_{m=0}^{M-1} u_m \tilde{X}_{i+m, j}$ は、その中間結果に対する列方向（垂直方向）の1次元フィルタ $\mathbf{u}$ の畳み込みである。

2. **計算量解析**:
   - 通常の2次元畳み込み：画素あたり $M \times M = M^2$ 回の乗算
   - 可分畳み込み：水平1D畳み込みで $M$ 回、垂直1D畳み込みで $M$ 回、合計 $2M$ 回の乗算
   - 高速化倍率：$\frac{M^2}{2M} = \frac{M}{2}$ 倍（例：$M=5$ で $2.5$ 倍、$M=7$ で $3.5$ 倍）。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_9_md))

    code_10_9 = r"""# Exercise 10.9 自己採点・数値検証
u = np.array([1.0, 2.0, 1.0])
v = np.array([1.0, 0.0, -1.0])  # Sobel風フィルタ
rng = np.random.default_rng(109)
X = rng.normal(0, 1, (10, 10))

res_10_9 = solve_exercise_10_9(u, v, X)
assert res_10_9["is_equal"] is True
assert res_10_9["ops_standard"] == 9
assert res_10_9["ops_separable"] == 6
assert res_10_9["theoretical_speedup"] == 1.5

np.testing.assert_allclose(res_10_9["out_standard"], res_10_9["out_separable"])
print("✓ Exercise 10.9: Passed successfully (Separable conv output matches 2D conv with M/2 speedup).")
"""
    cells.append(create_cell("code", code_10_9))

    # Exercise 10.10
    ex10_10_md = r"""---
### Exercise 10.10 (★): DeepDream 目的関数の事前活性化に関する勾配

#### 問題設定
DeepDream (Mordvintsev et al., 2015) において、ある選択された層 $l$ の特徴マップの事前活性化（または ReLU 活性化）の二乗ノルムを最大化する目的関数を考える：
$$
E(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^H \sum_{j=1}^W \sum_{k=1}^K a_{ijk}(\mathbf{x})^2
$$
このとき、目的関数の事前活性化 $a_{ijk}$ に関する勾配が、事前活性化の値そのものと厳密に等しい（$\frac{\partial E}{\partial a_{ijk}} = a_{ijk}$）ことを示せ。

#### 数学的証明
各要素 $a_{ijk}$ は独立な変数として扱えるため、特定のインデックス $(i', j', k')$ について偏微分を行うと：
$$
\frac{\partial E}{\partial a_{i'j'k'}} = \frac{\partial}{\partial a_{i'j'k'}} \left( \frac{1}{2} \sum_{i,j,k} a_{ijk}^2 \right) = \frac{1}{2} \cdot 2 a_{i'j'k'} = a_{i'j'k'}
$$
これは極めて深遠な実装上の帰結を持ちます：
誤差逆伝播法（Backpropagation）を実行する際、損失関数からの通常の誤差信号を逆伝播させる代わりに、**選択した層 $l$ における誤差勾配テンソルを、その層自身の順伝播活性化値 $\mathbf{A}^{(l)}$ にセットするだけで、目的関数 $E$ を最大化する入力画像勾配 $\nabla_{\mathbf{x}} E$ を自動微分で計算できる**ことになります。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_10_md))

    code_10_10 = r"""# Exercise 10.10 自己採点・数値検証
rng = np.random.default_rng(110)
a = rng.normal(0, 1, (4, 4, 3))
res_10_10 = solve_exercise_10_10(a)

assert res_10_10["is_identical_to_activation"] is True
np.testing.assert_allclose(res_10_10["analytical_gradient"], a)
print("✓ Exercise 10.10: Passed successfully (dE / da_ijk == a_ijk analytically verified).")
"""
    cells.append(create_cell("code", code_10_10))

    # Exercise 10.11
    ex10_11_md = r"""---
### Exercise 10.11 (★★): 物体検出におけるクラス確率の2つの定式化

#### 問題設定
物体検出器において、クラス $1, \dots, K$ および背景（物体なし）クラスの確率を予測する2つの方式を比較せよ：
1. 背景クラスをクラス $0$ とした $(K + 1)$ クラス単一ソフトマックス：
   $$
   p_k = \frac{\exp(a_k)}{\sum_{j=0}^K \exp(a_j)} \quad (k = 0, \dots, K)
   $$
2. 物体存在確率（ロジスティックシグモイド）と条件付きクラス確率（$K$ クラスソフトマックス）の分離：
   $$
   p_{\mathrm{obj}} = \sigma(a_{\mathrm{obj}}), \quad p(C = k \mid \mathrm{obj}) = \frac{\exp(b_k)}{\sum_{j=1}^K \exp(b_j)} \quad (k = 1, \dots, K)
   $$

#### 理論比較と関係性
- **方式1 (単一ソフトマックス)**: 全クラス（背景含む）が排他的であることを仮定。背景ロジット $a_0$ と物体クラスロジット $a_k$ が競合するため、物体が存在するかどうかの判定とクラス識別が結合している。
- **方式2 (分離定式化 / YOLO 等)**:
  - 物体がそこに存在するかどうかの確信度（Objectness Score）をまずシグモイドで独立に評価。
  - 物体が存在するという前提のもとで、どのカテゴリ $k$ であるかを $K$ クラスソフトマックスで評価。
  - 同時確率は乗法定理により $p(C = k, \mathrm{obj}) = p_{\mathrm{obj}} \cdot p(C = k \mid \mathrm{obj})$ となる。
  - **利点**: 物体性の学習と多クラス識別の学習が分離され、閾値処理や Non-Max Suppression (NMS) のスコアリングが容易になる。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_11_md))

    code_10_11 = r"""# Exercise 10.11 自己採点・数値検証
logits_single = np.array([1.0, 2.5, -0.5, 0.2])  # 0 is background, 1..3 are classes
logit_obj = 1.8
logits_cond = np.array([2.5, -0.5, 0.2])

res_10_11 = solve_exercise_10_11(logits_single, logit_obj, logits_cond)

assert res_10_11["p_single"].shape == (4,)
assert np.sum(res_10_11["p_single"]) == 1.0
assert 0.0 <= res_10_11["p_obj"] <= 1.0
assert np.sum(res_10_11["p_cond"]) == 1.0
assert np.isclose(np.sum(res_10_11["p_joint"]), res_10_11["p_obj"])

print("✓ Exercise 10.11: Passed successfully (Both object detection probability formulations verified).")
"""
    cells.append(create_cell("code", code_10_11))

    # Exercise 10.12
    ex10_12_md = r"""---
### Exercise 10.12 (★★★): スライディングウィンドウ探索と全結合畳み込みの計算量比較

#### 問題設定
入力画像が $8 \times 8$、ベース CNN が $6 \times 6$ 入力を受け取り以下の層構造を持つとする（教科書 p. 352）：
- Conv1: $3 \times 3$ フィルタ、ストライド 1、パディングなし
- Conv2: $3 \times 3$ フィルタ、ストライド 1、パディングなし
- Conv3 (FC相当): $2 \times 2$ フィルタ、ストライド 1、パディングなし
（チャネル数はすべて 1 とする）。
$8 \times 8$ 画像内のすべての可能な $6 \times 6$ 領域（$3 \times 3 = 9$ 箇所）に対して CNN を独立に実行する**素朴なスライディングウィンドウ法**と、画像全体を1度だけ通す**畳み込み走査法**における総乗算回数を比較せよ。

#### 数学的計算
1. **素朴なスライディングウィンドウ（9箇所の窓を独立計算）**:
   各 $6 \times 6$ 入力ウィンドウにおける乗算回数：
   - Conv1: 出力サイズ $(6 - 3 + 1) = 4 \times 4 = 16$。各出力に $3 \times 3 = 9$ 乗算 $\implies 16 \times 9 = 144$
   - Conv2: 出力サイズ $(4 - 3 + 1) = 2 \times 2 = 4$。各出力に $3 \times 3 = 9$ 乗算 $\implies 4 \times 9 = 36$
   - Conv3: 出力サイズ $(2 - 2 + 1) = 1 \times 1 = 1$。各出力に $2 \times 2 = 4$ 乗算 $\implies 1 \times 4 = 4$
   - 1ウィンドウあたりの合計乗算数 $= 144 + 36 + 4 = 184$ 回
   - 9箇所の総乗算回数：
     $$
     N_{\mathrm{naive}} = 9 \times 184 = \mathbf{1,656} \text{ 回}
     $$

2. **全結合畳み込み評価（$8 \times 8$ 画像を1回入力）**:
   - Conv1: $8 \times 8 \to 6 \times 6$ 出力（$36$ 画素）。乗算数 $= 36 \times 9 = 324$
   - Conv2: $6 \times 6 \to 4 \times 4$ 出力（$16$ 画素）。乗算数 $= 16 \times 9 = 144$
   - Conv3: $4 \times 4 \to 3 \times 3$ 出力（$9$ 画素）。乗算数 $= 9 \times 4 = 36$
   - 総乗算回数：
     $$
     N_{\mathrm{conv}} = 324 + 144 + 36 = \mathbf{504} \text{ 回}
     $$

3. **高速化倍率**:
   $$
   \text{Speedup} = \frac{1656}{504} \approx \mathbf{3.286} \text{ 倍}
   $$
   畳み込みの重み共有と空間的重複計算の排除により、全く同じ出力を約 3.3 倍高速に計算できる。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_12_md))

    code_10_12 = r"""# Exercise 10.12 自己採点・数値検証
res_10_12 = solve_exercise_10_12()

assert res_10_12["naive_multiplications"] == 1656
assert res_10_12["conv_multiplications"] == 504
assert np.isclose(res_10_12["speedup_factor"], 1656.0 / 504.0)

print(f"✓ Exercise 10.12: Naive={res_10_12['naive_multiplications']}, Conv={res_10_12['conv_multiplications']}, Speedup={res_10_12['speedup_factor']:.3f}x")
"""
    cells.append(create_cell("code", code_10_12))

    # Exercise 10.13
    ex10_13_md = r"""---
### Exercise 10.13 (★★★): 転置畳み込みと線形写像の転置行列 $\mathbf{A}^\top$ 双対性

#### 問題設定
1次元ストライド畳み込み（ダウンサンプリング）が行列積 $\mathbf{y} = \mathbf{A}\mathbf{x}$（ここで $\mathbf{x} \in \mathbb{R}^D, \mathbf{y} \in \mathbb{R}^{D'}$）として表されるとき、転置畳み込み（アップサンプリング）が転置行列 $\mathbf{A}^\top$ による写像 $\mathbf{z} = \mathbf{A}^\top \mathbf{y}$ に対応することを示せ。
また、任意のベクトル $\mathbf{x}, \mathbf{y}$ に対して内積の双対恒等式 $\langle \mathbf{A}\mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{A}^\top \mathbf{y} \rangle$ が成立することを示せ。

#### 数学的証明
1. **双対性の代数的証明**:
   標準内積 $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v}$ の定義より：
   $$
   \langle \mathbf{A}\mathbf{x}, \mathbf{y} \rangle = (\mathbf{A}\mathbf{x})^\top \mathbf{y} = (\mathbf{x}^\top \mathbf{A}^\top) \mathbf{y} = \mathbf{x}^\top (\mathbf{A}^\top \mathbf{y}) = \langle \mathbf{x}, \mathbf{A}^\top \mathbf{y} \rangle
   $$
   これは線形代数における随伴作用素（Adjoint Operator）の定義そのものである。

2. **転置畳み込みの展開と一致性**:
   ストライド $S$、カーネル $\mathbf{w} = (w_0, \dots, w_{M-1})^\top$ のダウンサンプリング行列 $\mathbf{A}$ は：
   $$
   A_{ij} = \begin{cases}
   w_{j - iS} & (0 \le j - iS < M) \\
   0 & (\text{otherwise})
   \end{cases}
   $$
   転置行列 $\mathbf{A}^\top$ の作用 $\mathbf{z} = \mathbf{A}^\top \mathbf{y}$ は：
   $$
   z_j = \sum_{i} A_{ji}^\top y_i = \sum_{i} A_{ij} y_i = \sum_{i: 0 \le j - iS < M} y_i w_{j - iS}
   $$
   これは、入力 $y_i$ にカーネル $\mathbf{w}$ を乗算したパッチを出力座標 $iS$ に配置し、重なり合う成分を加算する**転置畳み込み（分数ストライド畳み込み）**の定義式と完全に一致する。
   出力サイズは $(D' - 1)S + M = D$ となり、正しく解像度が復元される。 $\blacksquare$
"""
    cells.append(create_cell("markdown", ex10_13_md))

    code_10_13 = r"""# Exercise 10.13 自己採点・数値検証
x = np.array([1.2, -0.5, 2.3, 3.1, -1.8])  # D = 5
w = np.array([0.5, 1.0, -0.5])             # M = 3, S = 2 -> D' = (5-3)//2 + 1 = 2
y = np.array([1.5, -2.0])

res_10_13 = solve_exercise_10_13(x, y, w, S=2)

assert res_10_13["is_dual"] is True
assert res_10_13["matches_direct_transposed_conv"] is True
assert np.isclose(res_10_13["inner_product_Ax_y"], res_10_13["inner_product_x_ATy"])

print(f"✓ Exercise 10.13: <Ax, y> = {res_10_13['inner_product_Ax_y']:.4f} == <x, A^T y> = {res_10_13['inner_product_x_ATy']:.4f}")
print("✓ Exercise 10.13: Transposed convolution matches matrix transpose A^T perfectly.")
"""
    cells.append(create_cell("code", code_10_13))

    # Summary
    summary_md = r"""---
## 全13問 演習完了の総括

本ノートブックにより、第10章「畳み込みネットワーク」におけるすべての理論演習問題（Exercises 10.1 〜 10.13）の完全な証明および自己採点コード検証が完了しました。
- 畳み込みの基礎代数：単位ノルム最大化、テプリッツ行列積、相互相関と畳み込みの反転等価性、連続畳み込みの交換法則
- 幾何学的制約とアーキテクチャ設計：並進等変性と BatchNorm、パディング公式、ストライド公式、VGG-16 の受容野とパラメータ削減
- 高度な CNN 展開：階数1可分畳み込み、DeepDream の勾配等価性、物体検出確率、畳み込みスライディングウィンドウの計算量、転置畳み込みと行列転置双対性

これにより、第10章の全6節（10.1 〜 10.6）および全演習問題（10.1 〜 10.13）のすべての項目が完全に網羅・検証されました。
"""
    cells.append(create_cell("markdown", summary_md))

    return cells


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = build_cells()

    out_path = Path("10/10_Exercises.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook generated at: {out_path}")

    # Execute notebook to verify zero errors
    print("Executing notebook via nbconvert...")
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(out_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Notebook execution failed!")
        print(res.stderr)
        sys.exit(res.returncode)

    print("Notebook executed successfully with 0 errors!")


if __name__ == "__main__":
    main()

"""Build Chapter 9 Exercises notebook (9/9_Exercises.ipynb) and execute all cells.

Bishop & Bishop (2024), Chapter 9, pp. 281-285.
Exercises 9.1 to 9.18 (all 18 exercises).
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
    cells.append(create_cell("markdown", r"""# 第9章 正則化 (Regularization)
## 章末演習問題 (Exercises 9.1 〜 9.18)

本ノートブックは、Christopher M. Bishop & Hugh Bishop (2024)『深層学習：基礎と概念 (Deep Learning: Foundations and Concepts)』第9章「正則化」の**章末演習問題 全18問 (Exercises 9.1 〜 9.18)** に対する完全解答・数学的厳密導出・自己検証コードです。

---

### 演習問題 目次
- **Part 1: 帰納バイアス・対称性・正則化の物理的解釈 [Exercises 9.1 〜 9.4]**
  - **Exercise 9.1 (★)**: 正方形の90度回転群 ($C_4$) および2次元平面並進群 ($\mathbb{R}^2, +$) の群の公理証明
  - **Exercise 9.2 (★★)**: 入力ガウスノイズと非バイアス重みに対する $L_2$ 重み減衰（Weight Decay）の数学的等価性証明 (式 9.52, 9.53)
  - **Exercise 9.3 (★★)**: 二次正則化項の勾配流微分方程式 $\frac{d\mathbf{w}}{dt} = -\eta \mathbf{w}$ と重みの指数関数的減衰証明 (式 9.54, 9.55)
  - **Exercise 9.4 (★)**: 入力・出力の線形スケーリング変換に対するフィードフォワードネットワークの共変性証明 (式 9.6 - 9.13)
- **Part 2: 制約付き最適化・早期終了・ハードパラメータ共有 [Exercises 9.5 〜 9.7]**
  - **Exercise 9.5 (★★)**: ラグランジュ乗数法による $L_2$ 正則化と球状制約領域 $\|\mathbf{w}\|^2 \le \eta$ の等価性証明 (式 9.19, 9.20)
  - **Exercise 9.6 (★★★)**: 二次誤差関数における早期終了 (Early Stopping) と重み減衰の固有値スペクトル的等価性証明 (式 9.56 - 9.61)
  - **Exercise 9.7 (★★)**: ハード制約（共有・拘束重み）下での誤差逆伝播法とプーリング勾配 $\frac{\partial E}{\partial w} = \sum_{i \in S} \frac{\partial E}{\partial w_i}$ の証明
- **Part 3: ソフト重み共有の解析的勾配体系 [Exercises 9.8 〜 9.12]**
  - **Exercise 9.8 (★)**: ベイズの定理によるGMM負担率（事後確率） $\gamma_j(w_i)$ の導出 (式 9.24)
  - **Exercise 9.9 (★★)**: 重み勾配 $\frac{\partial \Omega}{\partial w_i} = \sum_j \gamma_j(w_i) \frac{w_i - \mu_j}{\sigma_j^2}$ の導出 (式 9.25)
  - **Exercise 9.10 (★★)**: クラスター中心勾配 $\frac{\partial \Omega}{\partial \mu_j} = \sum_i \gamma_j(w_i) \frac{\mu_j - w_i}{\sigma_j^2}$ の導出 (式 9.26)
  - **Exercise 9.11 (★★)**: 対数分散パラメータ $\beta_j = \ln(\sigma_j^2)$ に対する勾配導出 (式 9.28)
  - **Exercise 9.12 (★★)**: ソフトマックス混合係数のロジット勾配 $\frac{\partial \Omega}{\partial \gamma_j} = \sum_i (\pi_j - \gamma_j(w_i))$ の導出 (式 9.31)
- **Part 4: 残差結合の展開と委員会アンサンブル理論 [Exercises 9.13 〜 9.17]**
  - **Exercise 9.13 (★)**: 3ブロック残差ネットワークの再帰的展開による $2^3=8$ 個のパスアンサンブル導出 (式 9.40)
  - **Exercise 9.14 (★★)**: 独立無相関誤差における委員会誤差の $1/M$ 縮退定理の導出 (式 9.50)
  - **Exercise 9.15 (★★)**: イェンセンの不等式による二乗損失委員会誤差上限 $E_{\text{COM}} \le E_{\text{AV}}$ の証明 (式 9.64)
  - **Exercise 9.16 (★★)**: 任意の凸誤差関数 $E(y)$ に対する委員会誤差上限の一般化証明
  - **Exercise 9.17 (★★)**: 重み付き委員会予測が常に個別予測の上下限に収まるための必要十分条件（凸結合制約 $\alpha_m \ge 0, \sum \alpha_m = 1$）の証明
- **Part 5: ドロップアウトの統計的等価性 [Exercise 9.18]**
  - **Exercise 9.18 (★★★)**: 最小二乗線形回帰におけるドロップアウトとデータ依存型 $L_2$ 正則化の数学的厳密同値性証明 (式 9.68 - 9.73) および閉形式解 $\mathbf{W}^*$ の導出"""))

    # Cell 1: Environment Setup
    cells.append(create_cell("code", r"""# 環境セットアップと共通モジュールのインポート
import os
import sys
import numpy as np

# プロジェクトルートパスの設定
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.exercises_ch9 import (
    verify_exercise_9_1,
    verify_exercise_9_2,
    verify_exercise_9_3,
    verify_exercise_9_4,
    verify_exercise_9_5,
    verify_exercise_9_6,
    verify_exercise_9_7,
    verify_exercise_9_8,
    verify_exercise_9_9,
    verify_exercise_9_10,
    verify_exercise_9_11,
    verify_exercise_9_12,
    verify_exercise_9_13,
    verify_exercise_9_14,
    verify_exercise_9_15,
    verify_exercise_9_16,
    verify_exercise_9_17,
    verify_exercise_9_18,
)

print("Chapter 9 Exercises module loaded successfully.")"""))

    # -------------------------------------------------------------
    # Exercise 9.1
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.1 (★)
**問題**:
1. 正方形の90度の整数倍のすべての回転の集合が、回転の合成操作に関して群の4つの公理を満たすことを示せ。
2. 2次元平面における物体のすべての連続並進の集合もまた群をなすことを示せ。

**証明**:
群 $(G, \circ)$ の4公理：
1. **閉包性 (Closure)**: $\forall a, b \in G, a \circ b \in G$
2. **結合法則 (Associativity)**: $\forall a, b, c \in G, (a \circ b) \circ c = a \circ (b \circ c)$
3. **単位元の存在 (Identity)**: $\exists e \in G \text{ s.t. } e \circ a = a \circ e = a$
4. **逆元の存在 (Inverse)**: $\forall a \in G, \exists a^{-1} \in G \text{ s.t. } a \circ a^{-1} = a^{-1} \circ a = e$

**Part 1: 正方形の回転群 $C_4 = \{R_0, R_{90}, R_{180}, R_{270}\}$**:
- 角度の合成は $\theta = (\theta_1 + \theta_2) \pmod{360^\circ}$。$90^\circ$ の整数倍同士の和は常に $90^\circ$ の倍数であり、正方形を不変に保つ（閉包性）。
- 関数の合成は一般に結合的である（結合法則）。
- $0^\circ$ 回転 $R_0$ が単位元である（単位元）。
- 各回転 $R_\theta$ に対し、逆元は $R_{360^\circ - \theta}$ である（逆元）。よって位数4の巡回群 $C_4$ をなす。

**Part 2: 2次元平面の連続並進群 $(\mathbb{R}^2, +)$**:
- 2つの並進ベクトル $\mathbf{t}_1, \mathbf{t}_2 \in \mathbb{R}^2$ の合成はベクトル和 $\mathbf{t}_1 + \mathbf{t}_2 \in \mathbb{R}^2$（閉包性）。
- ベクトルの加法は結合的である（結合法則）。
- ゼロベクトル $\mathbf{0} = (0, 0)^T$ が単位元である（単位元）。
- 並進 $\mathbf{t}$ に対し、逆並進 $-\mathbf{t}$ が逆元である（逆元）。よって可換リー群 $(\mathbb{R}^2, +)$ をなす。"""))

    cells.append(create_cell("code", r"""# Exercise 9.1 自己検証コード
res_9_1 = verify_exercise_9_1()
print("Exercise 9.1 Verification:", res_9_1)
assert res_9_1["passed"] is True, "Exercise 9.1 failed!"
print("Exercise 9.1 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.2
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.2 (★★)
**問題**:
線形モデル $y(\mathbf{x}; \mathbf{w}) = w_0 + \sum_{i=1}^D w_i x_i$ と二乗和誤差関数：
$$E_D(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N \{y(\mathbf{x}_n; \mathbf{w}) - t_n\}^2$$
を考える。各入力変数 $x_i$ に平均 $0$、分散 $\sigma^2$ の互いに独立なガウスノイズ $\xi_i$ が加わるとする。
$\mathbb{E}[\xi_i] = 0$ および $\mathbb{E}[\xi_i \xi_j] = \sigma^2 \delta_{ij}$ を用いて、ノイズ分布に関して平均化した誤差関数を最小化することは、ノイズのない入力に対する二乗和誤差にバイアス $w_0$ を除外した重み減衰（Weight Decay）正則化項を加えた誤差関数を最小化することと等価であることを示せ。

**証明**:
ノイズが付加された入力を $\widetilde{\mathbf{x}}_n = \mathbf{x}_n + \boldsymbol{\xi}_n$ とすると、ノイズ付加モデルの出力は：
$$y(\widetilde{\mathbf{x}}_n; \mathbf{w}) = w_0 + \sum_{i=1}^D w_i (x_{ni} + \xi_{ni}) = y(\mathbf{x}_n; \mathbf{w}) + \sum_{i=1}^D w_i \xi_{ni}$$
二乗誤差の各項を展開すると：
$$\begin{aligned}
\{y(\widetilde{\mathbf{x}}_n; \mathbf{w}) - t_n\}^2 &= \left\{ [y(\mathbf{x}_n; \mathbf{w}) - t_n] + \sum_{i=1}^D w_i \xi_{ni} \right\}^2 \\
&= \{y(\mathbf{x}_n; \mathbf{w}) - t_n\}^2 + 2 [y(\mathbf{x}_n; \mathbf{w}) - t_n] \sum_{i=1}^D w_i \xi_{ni} + \left( \sum_{i=1}^D w_i \xi_{ni} \right) \left( \sum_{j=1}^D w_j \xi_{nj} \right)
\end{aligned}$$
ノイズ $\boldsymbol{\xi}$ について期待値を取ると：
- 第2項: $\mathbb{E}[\xi_{ni}] = 0$ より消滅する。
- 第3項: $\mathbb{E}[\xi_{ni} \xi_{nj}] = \sigma^2 \delta_{ij}$ より、$\sum_{i,j} w_i w_j \sigma^2 \delta_{ij} = \sigma^2 \sum_{i=1}^D w_i^2$ となる。
したがって、全 $N$ 点についての期待誤差は：
$$\mathbb{E}_{\boldsymbol{\xi}} [E_D(\mathbf{w})] = \frac{1}{2} \sum_{n=1}^N \{y(\mathbf{x}_n; \mathbf{w}) - t_n\}^2 + \frac{N \sigma^2}{2} \sum_{i=1}^D w_i^2$$
これは、正則化パラメータ $\lambda = N \sigma^2$ を持ち、バイアスパラメータ $w_0$ を含まない厳密な重み減衰項である。"""))

    cells.append(create_cell("code", r"""# Exercise 9.2 自己検証コード
res_9_2 = verify_exercise_9_2()
print("Exercise 9.2 Verification:", res_9_2)
assert res_9_2["passed"] is True, "Exercise 9.2 failed!"
print("Exercise 9.2 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.3
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.3 (★★)
**問題**:
二次正則化項 $\Omega(\mathbf{w}) = \frac{1}{2} \mathbf{w}^T \mathbf{w}$ と勾配降下更新則：
$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} - \eta \nabla \Omega(\mathbf{w}^{(\tau)})$$
を考える。無限小ステップの極限を考えることで、$\mathbf{w}$ の時間発展に関する微分方程式を導き、初期値 $\mathbf{w}_0$ からの解を求め、$\mathbf{w}$ の要素が指数関数的にゼロへ減衰することを示せ。

**証明**:
勾配は $\nabla \Omega(\mathbf{w}) = \mathbf{w}$ である。更新則は：
$$\mathbf{w}^{(\tau+1)} - \mathbf{w}^{(\tau)} = -\eta \mathbf{w}^{(\tau)}$$
時間ステップ幅を $\Delta t = \eta$ とし、連続時間パラメータ $t = \tau \Delta t$ を導入して $\Delta t \to 0$ の極限を取ると：
$$\frac{d\mathbf{w}}{dt} = -\mathbf{w}$$
この一階同次線形微分方程式の解は：
$$\mathbf{w}(t) = \mathbf{w}_0 \exp(-t)$$
元の離散ステップで表せば、$\tau$ 回の更新後の重みは $\mathbf{w}(\tau) = \mathbf{w}_0 e^{-\eta \tau}$ となり、各成分は時間とともに指数関数的にゼロへ減衰する。"""))

    cells.append(create_cell("code", r"""# Exercise 9.3 自己検証コード
res_9_3 = verify_exercise_9_3()
print("Exercise 9.3 Verification:", res_9_3)
assert res_9_3["passed"] is True, "Exercise 9.3 failed!"
print("Exercise 9.3 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.4
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.4 (★)
**問題**:
入力にアフィン変換 $\widetilde{x}_i = a x_i + b$ を施したとき、第1層の重みとバイアスを：
$$\widetilde{w}_{ji} = \frac{1}{a} w_{ji}, \quad \widetilde{b}_j = b_j - \frac{b}{a} \sum_i w_{ji}$$
と変換すればネットワーク関数が不変であることを示せ。同様に出力のアフィン変換 $\widetilde{y}_k = c y_k + d$ に対しても第2層パラメータの変換を示せ。

**証明**:
1. 第1層の活性化前入力：
$$a_j = \sum_i \widetilde{w}_{ji} \widetilde{x}_i + \widetilde{b}_j = \sum_i \left( \frac{1}{a} w_{ji} \right) (a x_i + b) + b_j - \frac{b}{a} \sum_i w_{ji} = \sum_i w_{ji} x_i + \frac{b}{a} \sum_i w_{ji} + b_j - \frac{b}{a} \sum_i w_{ji} = \sum_i w_{ji} x_i + b_j$$
となり、隠れ層への入力は完全に不変に保たれる。
2. 出力層：
$$\widetilde{y}_k = \sum_j \widetilde{w}_{kj} z_j + \widetilde{b}_k = \sum_j (c w_{kj}) z_j + (c b_k + d) = c \left( \sum_j w_{kj} z_j + b_k \right) + d = c y_k + d$$
となり、出力全体が意図したアフィン変換を受ける。"""))

    cells.append(create_cell("code", r"""# Exercise 9.4 自己検証コード
res_9_4 = verify_exercise_9_4()
print("Exercise 9.4 Verification:", res_9_4)
assert res_9_4["passed"] is True, "Exercise 9.4 failed!"
print("Exercise 9.4 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.5
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.5 (★★)
**問題**:
ラグランジュ乗数法を用いて、正則化誤差関数 $\widetilde{E}(\mathbf{w}) = E(\mathbf{w}) + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w}$ の最小化が、不等式制約 $\mathbf{w}^T \mathbf{w} \le \eta$ のもとでの $E(\mathbf{w})$ の最小化と等価であることを示せ。パラメータ $\lambda$ と $\eta$ の関係を議論せよ。

**証明**:
制約付き最適化問題：
$$\min_{\mathbf{w}} E(\mathbf{w}) \quad \text{s.t.} \quad \frac{1}{2} \mathbf{w}^T \mathbf{w} \le \frac{1}{2} \eta$$
ラグランジュ関数は：
$$L(\mathbf{w}, \lambda) = E(\mathbf{w}) + \frac{\lambda}{2} (\mathbf{w}^T \mathbf{w} - \eta)$$
KKT条件（Karush-Kuhn-Tucker conditions）は：
1. $\nabla_{\mathbf{w}} L = \nabla E(\mathbf{w}) + \lambda \mathbf{w} = \mathbf{0}$
2. $\mathbf{w}^T \mathbf{w} \le \eta$
3. $\lambda \ge 0$
4. $\lambda (\mathbf{w}^T \mathbf{w} - \eta) = 0$（相補スラック性）

制約がアクティブ（$\mathbf{w}^T \mathbf{w} = \eta$）な場合、$\lambda > 0$ であり、$L(\mathbf{w}, \lambda)$ の $\mathbf{w}$ に関する停留点は正則化関数 $\widetilde{E}(\mathbf{w})$ の最小解と完全に一致する。
$\eta$（許容ノルム球の半径の二乗）を小さくするほど、より強い制約となり、対応するラグランジュ乗数 $\lambda$（正則化係数）は単調に増大する。"""))

    cells.append(create_cell("code", r"""# Exercise 9.5 自己検証コード
res_9_5 = verify_exercise_9_5()
print("Exercise 9.5 Verification:", res_9_5)
assert res_9_5["passed"] is True, "Exercise 9.5 failed!"
print("Exercise 9.5 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.6
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.6 (★★★)
**問題**:
二次誤差関数 $E = E_0 + \frac{1}{2}(\mathbf{w} - \mathbf{w}^*)^T \mathbf{H} (\mathbf{w} - \mathbf{w}^*)$ に対し、原点 $\mathbf{w}^{(0)} = \mathbf{0}$ からの勾配降下法 $\mathbf{w}^{(\tau)} = \mathbf{w}^{(\tau-1)} - \eta \nabla E$ を適用する。
ヘッセ行列 $\mathbf{H}$ の固有値分解を用いて、$\tau$ ステップ後の重み成分が：
$$w_j^{(\tau)} = \{1 - (1 - \eta \lambda_j)^\tau\} w_j^* \quad \text{(式 9.58)}$$
となることを示し、$(\tau \eta)^{-1}$ が重み減衰の正則化パラメータ $\alpha$ と類似の役割を果たすことを論ぜよ。

**証明**:
勾配は $\nabla E = \mathbf{H}(\mathbf{w} - \mathbf{w}^*)$。
重み更新は：
$$\mathbf{w}^{(\tau)} - \mathbf{w}^* = (\mathbf{I} - \eta \mathbf{H})(\mathbf{w}^{(\tau-1)} - \mathbf{w}^*)$$
$\mathbf{w}^{(0)} = \mathbf{0}$ より、$\tau$ 回の反復後は：
$$\mathbf{w}^{(\tau)} - \mathbf{w}^* = (\mathbf{I} - \eta \mathbf{H})^\tau (-\mathbf{w}^*)$$
固有ベクトル基底 $\mathbf{u}_j$ 上に射影すると（$\mathbf{H} \mathbf{u}_j = \lambda_j \mathbf{u}_j$）：
$$w_j^{(\tau)} - w_j^* = -(1 - \eta \lambda_j)^\tau w_j^* \implies w_j^{(\tau)} = [1 - (1 - \eta \lambda_j)^\tau] w_j^*$$
一方、重み減衰（Weight Decay $\alpha$）の解は：
$$w_j(\alpha) = \frac{\lambda_j}{\lambda_j + \alpha} w_j^*$$
$(1 - \eta \lambda_j)^\tau \approx \exp(-\tau \eta \lambda_j)$ であり、テイラー展開すると：
$$\lambda_j \gg (\tau \eta)^{-1} \implies w_j^{(\tau)} \approx w_j^*, \quad \lambda_j \ll (\tau \eta)^{-1} \implies w_j^{(\tau)} \approx \tau \eta \lambda_j w_j^* \ll w_j^*$$
これは重み減衰において $\lambda_j \gg \alpha$ で減衰なし、$\lambda_j \ll \alpha$ で強く減衰する振る舞いと完全に同等であり、$(\tau \eta)^{-1}$ は実効的正則化係数として機能する。"""))

    cells.append(create_cell("code", r"""# Exercise 9.6 自己検証コード
res_9_6 = verify_exercise_9_6()
print("Exercise 9.6 Verification:", res_9_6)
assert res_9_6["passed"] is True, "Exercise 9.6 failed!"
print("Exercise 9.6 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.7
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.7 (★★)
**問題**:
ニューラルネットワークにおいて複数の重みが同一の値を持つよう拘束されている場合（ハード重み共有）、誤差関数の微係数を評価する標準的な逆伝播アルゴリズムをどのように修正すべきか論ぜよ。

**証明**:
共有される単一のパラメータを $w$ とし、ネットワーク内の複数の接続重み $\{w_i\}_{i \in S}$ がすべて $w_i = w$ であるとする。
多変数の合成関数の連鎖律（Chain Rule）より：
$$\frac{\partial E}{\partial w} = \sum_{i \in S} \frac{\partial E}{\partial w_i} \frac{\partial w_i}{\partial w}$$
制約 $w_i = w$ より $\frac{\partial w_i}{\partial w} = 1$ であるから：
$$\frac{\partial E}{\partial w} = \sum_{i \in S} \frac{\partial E}{\partial w_i}$$
したがって、通常の誤差逆伝播法を修正なしに適用して各結合に対する局所勾配 $\frac{\partial E}{\partial w_i}$ を通常通り計算し、**同一の共有グループに属するすべての勾配を足し合わせる（プーリングする）**だけで厳密な勾配が得られる。CNNの畳み込み層における空間方向の勾配加算もこの定理の直接の帰結である。"""))

    cells.append(create_cell("code", r"""# Exercise 9.7 自己検証コード
res_9_7 = verify_exercise_9_7()
print("Exercise 9.7 Verification:", res_9_7)
assert res_9_7["passed"] is True, "Exercise 9.7 failed!"
print("Exercise 9.7 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.8
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.8 (★)
**問題**:
混合分布 $p(w) = \sum_{j=1}^M \pi_j \mathcal{N}(w | \mu_j, \sigma_j^2)$ を考える。ベイズの定理を用いて、事後確率（負担率） $p(j | w)$ が式 (9.24) で与えられることを示せ。

**証明**:
$\pi_j = p(j)$ は第 $j$ 成分が選ばれる事前確率、$p(w | j) = \mathcal{N}(w | \mu_j, \sigma_j^2)$ は条件付き密度である。
ベイズの定理より：
$$\gamma_j(w) = p(j | w) = \frac{p(j) p(w | j)}{p(w)} = \frac{\pi_j \mathcal{N}(w | \mu_j, \sigma_j^2)}{\sum_{k=1}^M \pi_k \mathcal{N}(w | \mu_k, \sigma_k^2)}$$
これは式 (9.24) そのものである。"""))

    cells.append(create_cell("code", r"""# Exercise 9.8 自己検証コード
res_9_8 = verify_exercise_9_8()
print("Exercise 9.8 Verification:", res_9_8)
assert res_9_8["passed"] is True, "Exercise 9.8 failed!"
print("Exercise 9.8 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.9
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.9 (★★)
**問題**:
式 (9.21), (9.22), (9.23), (9.24) を用いて、重み $w_i$ に関する正則化項の導関数 (9.25) を検証せよ：
$$\frac{\partial \Omega}{\partial w_i} = \sum_{j=1}^M \gamma_j(w_i) \frac{w_i - \mu_j}{\sigma_j^2}$$

**証明**:
$\Omega(\mathbf{w}) = -\sum_i \ln p(w_i)$ である。各項について：
$$\frac{\partial}{\partial w_i} [-\ln p(w_i)] = -\frac{1}{p(w_i)} \frac{\partial p(w_i)}{\partial w_i}$$
ガウス分布の導関数は：
$$\frac{\partial}{\partial w_i} \mathcal{N}(w_i | \mu_j, \sigma_j^2) = -\frac{w_i - \mu_j}{\sigma_j^2} \mathcal{N}(w_i | \mu_j, \sigma_j^2)$$
したがって：
$$\frac{\partial p(w_i)}{\partial w_i} = -\sum_{j=1}^M \pi_j \mathcal{N}(w_i | \mu_j, \sigma_j^2) \frac{w_i - \mu_j}{\sigma_j^2}$$
両辺を $-p(w_i)$ で割れば：
$$\frac{\partial \Omega}{\partial w_i} = \sum_{j=1}^M \underbrace{\frac{\pi_j \mathcal{N}(w_i | \mu_j, \sigma_j^2)}{p(w_i)}}_{= \gamma_j(w_i)} \frac{w_i - \mu_j}{\sigma_j^2} = \sum_{j=1}^M \gamma_j(w_i) \frac{w_i - \mu_j}{\sigma_j^2}$$
となり式 (9.25) が証明された。"""))

    cells.append(create_cell("code", r"""# Exercise 9.9 自己検証コード
res_9_9 = verify_exercise_9_9()
print("Exercise 9.9 Verification:", res_9_9)
assert res_9_9["passed"] is True, "Exercise 9.9 failed!"
print("Exercise 9.9 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.10
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.10 (★★)
**問題**:
クラスター中心 $\mu_j$ に関する正則化項の導関数 (9.26) を検証せよ：
$$\frac{\partial \Omega}{\partial \mu_j} = \sum_{i=1}^W \gamma_j(w_i) \frac{\mu_j - w_i}{\sigma_j^2}$$

**証明**:
$\Omega(\mathbf{w}) = -\sum_i \ln p(w_i)$ より：
$$\frac{\partial \Omega}{\partial \mu_j} = -\sum_{i=1}^W \frac{1}{p(w_i)} \frac{\partial p(w_i)}{\partial \mu_j}$$
$\mu_j$ に関する微分は第 $j$ 成分のみに作用する：
$$\frac{\partial}{\partial \mu_j} \mathcal{N}(w_i | \mu_j, \sigma_j^2) = \frac{w_i - \mu_j}{\sigma_j^2} \mathcal{N}(w_i | \mu_j, \sigma_j^2)$$
したがって：
$$\frac{\partial \Omega}{\partial \mu_j} = -\sum_{i=1}^W \frac{\pi_j \mathcal{N}(w_i | \mu_j, \sigma_j^2)}{p(w_i)} \frac{w_i - \mu_j}{\sigma_j^2} = \sum_{i=1}^W \gamma_j(w_i) \frac{\mu_j - w_i}{\sigma_j^2}$$
となり式 (9.26) が示された。"""))

    cells.append(create_cell("code", r"""# Exercise 9.10 自己検証コード
res_9_10 = verify_exercise_9_10()
print("Exercise 9.10 Verification:", res_9_10)
assert res_9_10["passed"] is True, "Exercise 9.10 failed!"
print("Exercise 9.10 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.11
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.11 (★★)
**問題**:
$\beta_j = \ln(\sigma_j^2)$ とするとき、$\beta_j$ に関する導関数 (9.28) を検証せよ：
$$\frac{\partial \Omega}{\partial \beta_j} = \frac{1}{2} \sum_{i=1}^W \gamma_j(w_i) \left[ 1 - \frac{(w_i - \mu_j)^2}{\sigma_j^2} \right]$$

**証明**:
$\mathcal{N}(w_i | \mu_j, \sigma_j^2) = (2\pi)^{-1/2} e^{-\beta_j / 2} \exp\left( -\frac{1}{2} e^{-\beta_j} (w_i - \mu_j)^2 \right)$ である。
対数を取って微分すると：
$$\frac{\partial \ln \mathcal{N}}{\partial \beta_j} = -\frac{1}{2} + \frac{1}{2} e^{-\beta_j} (w_i - \mu_j)^2 = -\frac{1}{2} \left[ 1 - \frac{(w_i - \mu_j)^2}{\sigma_j^2} \right]$$
したがって：
$$\frac{\partial \mathcal{N}}{\partial \beta_j} = \mathcal{N} \cdot \left\{ -\frac{1}{2} \left[ 1 - \frac{(w_i - \mu_j)^2}{\sigma_j^2} \right] \right\}$$
$\Omega$ の微分に代入すれば：
$$\frac{\partial \Omega}{\partial \beta_j} = -\sum_{i=1}^W \frac{\pi_j}{p(w_i)} \frac{\partial \mathcal{N}}{\partial \beta_j} = \frac{1}{2} \sum_{i=1}^W \gamma_j(w_i) \left[ 1 - \frac{(w_i - \mu_j)^2}{\sigma_j^2} \right]$$
となり式 (9.28) が証明された。"""))

    cells.append(create_cell("code", r"""# Exercise 9.11 自己検証コード
res_9_11 = verify_exercise_9_11()
print("Exercise 9.11 Verification:", res_9_11)
assert res_9_11["passed"] is True, "Exercise 9.11 failed!"
print("Exercise 9.11 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.12
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.12 (★★)
**問題**:
混合係数 $\pi_k = \frac{\exp(\eta_k)}{\sum_l \exp(\eta_l)}$ の補助パラメータ $\eta_j$ に関する微係数が：
$$\frac{\partial \pi_k}{\partial \eta_j} = \pi_j (\delta_{jk} - \pi_k) \quad \text{(式 9.63)}$$
で与えられることを示し、制約 $\sum_k \gamma_k(w_i) = 1$ を用いて結果 (9.31) $\frac{\partial \Omega}{\partial \eta_j} = \sum_i (\pi_j - \gamma_j(w_i))$ を導け。

**証明**:
1. $k = j$ のとき：
$$\frac{\partial \pi_j}{\partial \eta_j} = \frac{e^{\eta_j} \sum e^{\eta_l} - (e^{\eta_j})^2}{(\sum e^{\eta_l})^2} = \pi_j - \pi_j^2 = \pi_j (1 - \pi_j)$$
$k \neq j$ のとき：
$$\frac{\partial \pi_k}{\partial \eta_j} = \frac{0 - e^{\eta_k} e^{\eta_j}}{(\sum e^{\eta_l})^2} = -\pi_j \pi_k$$
まとめて $\frac{\partial \pi_k}{\partial \eta_j} = \pi_j (\delta_{jk} - \pi_k)$ となる。

2. $\Omega$ の微分：
$$\frac{\partial \Omega}{\partial \eta_j} = -\sum_{i=1}^W \frac{1}{p(w_i)} \sum_{k=1}^M \frac{\partial \pi_k}{\partial \eta_j} \mathcal{N}(w_i | \mu_k, \sigma_k^2) = -\sum_{i=1}^W \frac{1}{p(w_i)} \sum_{k=1}^M \pi_j (\delta_{jk} - \pi_k) \mathcal{N}_k$$
展開すると：
$$= -\sum_{i=1}^W \left[ \pi_j \frac{\mathcal{N}_j}{p(w_i)} - \pi_j \sum_{k=1}^M \frac{\pi_k \mathcal{N}_k}{p(w_i)} \right] = -\sum_{i=1}^W [\gamma_j(w_i) - \pi_j \cdot 1] = \sum_{i=1}^W [\pi_j - \gamma_j(w_i)]$$
となり式 (9.31) が証明された。"""))

    cells.append(create_cell("code", r"""# Exercise 9.12 自己検証コード
res_9_12 = verify_exercise_9_12()
print("Exercise 9.12 Verification:", res_9_12)
assert res_9_12["passed"] is True, "Exercise 9.12 failed!"
print("Exercise 9.12 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.13
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.13 (★)
**問題**:
残差変換式 (9.35), (9.36), (9.37) を結合することで、ネットワーク全体が式 (9.40) の展開形式になることを検証せよ。

**証明**:
$$\begin{aligned}
\mathbf{z}_1 &= \mathbf{x} + \mathbf{F}_1(\mathbf{x}) \\
\mathbf{z}_2 &= \mathbf{z}_1 + \mathbf{F}_2(\mathbf{z}_1) = \mathbf{x} + \mathbf{F}_1(\mathbf{x}) + \mathbf{F}_2(\mathbf{x} + \mathbf{F}_1(\mathbf{x})) \\
\mathbf{y} &= \mathbf{z}_2 + \mathbf{F}_3(\mathbf{z}_2) = [\mathbf{x} + \mathbf{F}_1(\mathbf{x}) + \mathbf{F}_2(\mathbf{x} + \mathbf{F}_1(\mathbf{x}))] + \mathbf{F}_3(\mathbf{x} + \mathbf{F}_1(\mathbf{x}) + \mathbf{F}_2(\mathbf{x} + \mathbf{F}_1(\mathbf{x})))
\end{aligned}$$
これは、入力 $\mathbf{x}$ から出力に至る可能な $2^3 = 8$ 個のパスの重ね合わせであり、一般の $L$ ブロックに対して $\mathbf{z}_L = \sum_{S \subseteq \{1, \dots, L\}} F_S(\mathbf{x})$ となる。"""))

    cells.append(create_cell("code", r"""# Exercise 9.13 自己検証コード
res_9_13 = verify_exercise_9_13()
print("Exercise 9.13 Verification:", res_9_13)
assert res_9_13["passed"] is True, "Exercise 9.13 failed!"
print("Exercise 9.13 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.14
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.14 (★★)
**問題**:
単純委員会モデルの期待二乗和誤差 $E_{\text{AV}}$ および $E_{\text{COM}}$ に対し、個別誤差がゼロ平均かつ無相関（式 9.48, 9.49）であると仮定して、結果 (9.50) $E_{\text{COM}} = \frac{1}{M} E_{\text{AV}}$ を導出せよ。

**証明**:
$$E_{\text{COM}} = \mathbb{E}_{\mathbf{x}} \left[ \left( \frac{1}{M} \sum_{m=1}^M \epsilon_m(\mathbf{x}) \right)^2 \right] = \frac{1}{M^2} \sum_{m=1}^M \sum_{l=1}^M \mathbb{E}_{\mathbf{x}} [\epsilon_m(\mathbf{x}) \epsilon_l(\mathbf{x})]$$
無相関仮定 $\mathbb{E}[\epsilon_m \epsilon_l] = 0 \ (m \neq l)$ より、交差項がすべて消去される：
$$= \frac{1}{M^2} \sum_{m=1}^M \mathbb{E}_{\mathbf{x}} [\epsilon_m(\mathbf{x})^2] = \frac{1}{M} \left( \frac{1}{M} \sum_{m=1}^M \mathbb{E}_{\mathbf{x}} [\epsilon_m(\mathbf{x})^2] \right) = \frac{1}{M} E_{\text{AV}}$$
が成立する。"""))

    cells.append(create_cell("code", r"""# Exercise 9.14 自己検証コード
res_9_14 = verify_exercise_9_14()
print("Exercise 9.14 Verification:", res_9_14)
assert res_9_14["passed"] is True, "Exercise 9.14 failed!"
print("Exercise 9.14 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.15
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.15 (★★)
**問題**:
凸関数 $f(u) = u^2$ に対するイェンセンの不等式を用いて、委員会の期待二乗和誤差 $E_{\text{COM}}$ と個別モデルの平均誤差 $E_{\text{AV}}$ が常に：
$$E_{\text{COM}} \le E_{\text{AV}} \quad \text{(式 9.64)}$$
を満たすことを示せ。

**証明**:
イェンセンの不等式（式 2.102）：凸関数 $f$ に対し $f\left( \frac{1}{M} \sum_{m=1}^M u_m \right) \le \frac{1}{M} \sum_{m=1}^M f(u_m)$。
$f(u) = u^2$ は二階微分 $f''(u) = 2 > 0$ より狭義凸関数である。
各データ点 $\mathbf{x}$ において $u_m = \epsilon_m(\mathbf{x})$ と置くと：
$$\left( \frac{1}{M} \sum_{m=1}^M \epsilon_m(\mathbf{x}) \right)^2 \le \frac{1}{M} \sum_{m=1}^M \epsilon_m(\mathbf{x})^2$$
両辺の入力分布 $\mathbf{x}$ に関する期待値を取ると：
$$\mathbb{E}_{\mathbf{x}} \left[ \left( \frac{1}{M} \sum_{m=1}^M \epsilon_m(\mathbf{x}) \right)^2 \right] \le \frac{1}{M} \sum_{m=1}^M \mathbb{E}_{\mathbf{x}} [\epsilon_m(\mathbf{x})^2]$$
左辺は $E_{\text{COM}}$、右辺は $E_{\text{AV}}$ であるから、$E_{\text{COM}} \le E_{\text{AV}}$ が証明された。"""))

    cells.append(create_cell("code", r"""# Exercise 9.15 自己検証コード
res_9_15 = verify_exercise_9_15()
print("Exercise 9.15 Verification:", res_9_15)
assert res_9_15["passed"] is True, "Exercise 9.15 failed!"
print("Exercise 9.15 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.16
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.16 (★★)
**問題**:
イェンセンの不等式を用いて、誤差関数 $E(y)$ が $y$ の凸関数である限り、二乗誤差に限らず任意の誤差関数に対して結果 (9.64) $E_{\text{COM}} \le E_{\text{AV}}$ が成り立つことを示せ。

**証明**:
誤差関数 $E(y)$ が凸関数であるとき、イェンセンの不等式より：
$$E(y_{\text{COM}}(\mathbf{x})) = E\left( \frac{1}{M} \sum_{m=1}^M y_m(\mathbf{x}) \right) \le \frac{1}{M} \sum_{m=1}^M E(y_m(\mathbf{x}))$$
両辺の期待値を取れば直ちに：
$$\mathbb{E}_{\mathbf{x}} [E(y_{\text{COM}}(\mathbf{x}))] \le \frac{1}{M} \sum_{m=1}^M \mathbb{E}_{\mathbf{x}} [E(y_m(\mathbf{x}))]$$
となり、絶対値誤差（$L_1$）、Huber損失、交差エントロピー誤差など、任意の凸損失関数について委員会モデルの期待誤差は個別モデルの平均誤差以下となる。"""))

    cells.append(create_cell("code", r"""# Exercise 9.16 自己検証コード
res_9_16 = verify_exercise_9_16()
print("Exercise 9.16 Verification:", res_9_16)
assert res_9_16["passed"] is True, "Exercise 9.16 failed!"
print("Exercise 9.16 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.17
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.17 (★★)
**問題**:
重み付き委員会 $y_{\text{COM}}(\mathbf{x}) = \sum_{m=1}^M \alpha_m y_m(\mathbf{x})$ において、任意の $\mathbf{x}$ で予測値が常にメンバーモデルの最小値と最大値の間に収まる：
$$y_{\min}(\mathbf{x}) \le y_{\text{COM}}(\mathbf{x}) \le y_{\max}(\mathbf{x}) \quad \text{(式 9.66)}$$
ための必要十分条件が $\alpha_m \ge 0$ かつ $\sum_{m=1}^M \alpha_m = 1$（凸結合）であることを示せ。

**証明**:
1. **十分性**:
   $\alpha_m \ge 0$ かつ $\sum_m \alpha_m = 1$ のとき、
   $$y_{\text{COM}} = \sum_m \alpha_m y_m \ge \sum_m \alpha_m y_{\min} = y_{\min} \sum_m \alpha_m = y_{\min}$$
   同様に $y_{\text{COM}} \le y_{\max} \sum_m \alpha_m = y_{\max}$。
2. **必要性**:
   - すべての $y_m = c$（定数）とすると、$c \le c \sum \alpha_m \le c$ より $\sum \alpha_m = 1$ が必須。
   - もしある $k$ で $\alpha_k < 0$ だとすると、$y_k = 0, y_{m \neq k} = 1$ と選べば $y_{\min} = 0$ だが、$y_{\text{COM}} = \sum_{m \neq k} \alpha_m = 1 - \alpha_k > 1 = y_{\max}$ となり矛盾。
   したがって $\alpha_m \ge 0$ かつ $\sum \alpha_m = 1$ が必要十分である。"""))

    cells.append(create_cell("code", r"""# Exercise 9.17 自己検証コード
res_9_17 = verify_exercise_9_17()
print("Exercise 9.17 Verification:", res_9_17)
assert res_9_17["passed"] is True, "Exercise 9.17 failed!"
print("Exercise 9.17 PASSED!")"""))

    # -------------------------------------------------------------
    # Exercise 9.18
    # -------------------------------------------------------------
    cells.append(create_cell("markdown", r"""---
### Exercise 9.18 (★★★)
**問題**:
線形モデル $y_k = \sum_{i=1}^D w_{ki} x_i$ と、パラメータ $\rho$ のベルヌーイ分布に従うドロップアウト行列 $R_{ni} \in \{0, 1\}$ を持つ二乗和誤差：
$$E(\mathbf{W}) = \sum_{n=1}^N \sum_{k=1}^K \left\{ t_{nk} - \sum_{i=1}^D w_{ki} \frac{R_{ni}}{\rho} x_{ni} \right\}^2$$
を考える。
1. $\mathbb{E}[R_{ni}] = \rho$ および $\mathbb{E}[R_{ni} R_{nj}] = \rho \delta_{ij} + \rho^2 (1 - \delta_{ij})$ を示せ。
2. ドロップアウトの期待誤差関数が式 (9.72)-(9.73) の正則化二乗和誤差となることを示せ。
3. この期待誤差関数を最小化する重み行列 $\mathbf{W}^*$ の閉形式解を導け。

**証明**:
1. ベルヌーイ確率変数 $R_{ni} \in \{0, 1\}$（$P(R=1) = \rho$）より：
   $$\mathbb{E}[R_{ni}] = 1 \cdot \rho + 0 \cdot (1 - \rho) = \rho$$
   $i = j$ のとき $R_{ni}^2 = R_{ni}$ より $\mathbb{E}[R_{ni}^2] = \rho$。
   $i \neq j$ のとき独立性より $\mathbb{E}[R_{ni} R_{nj}] = \mathbb{E}[R_{ni}]\mathbb{E}[R_{nj}] = \rho^2$。
   これをクロネッカーのデルタを用いてまとめると $\mathbb{E}[R_{ni} R_{nj}] = \rho \delta_{ij} + \rho^2 (1 - \delta_{ij})$ となる。

2. 誤差の二乗を展開して期待値を取ると：
   $$\mathbb{E}_{\mathbf{R}} [E(\mathbf{W})] = \sum_{n=1}^N \sum_{k=1}^K \left[ t_{nk}^2 - 2 t_{nk} \sum_i w_{ki} x_{ni} + \frac{1}{\rho^2} \sum_{i,j} w_{ki} w_{kj} x_{ni} x_{nj} \mathbb{E}[R_{ni} R_{nj}] \right]$$
   最後の項は：
   $$\frac{1}{\rho^2} \left[ \rho^2 \sum_{i \neq j} w_{ki} w_{kj} x_{ni} x_{nj} + \rho \sum_i w_{ki}^2 x_{ni}^2 \right] = \left( \sum_i w_{ki} x_{ni} \right)^2 + \frac{1 - \rho}{\rho} \sum_i w_{ki}^2 x_{ni}^2$$
   したがって：
   $$\mathbb{E}[E(\mathbf{W})] = \sum_{n,k} \left( t_{nk} - \sum_i w_{ki} x_{ni} \right)^2 + \frac{1 - \rho}{\rho} \sum_{n,k,i} w_{ki}^2 x_{ni}^2$$
   となり式 (9.72)-(9.73) が導かれた。

3. 行列記法：
   $\mathbf{X}^T \mathbf{X}$ の対角成分を $\operatorname{diag}(\mathbf{X}^T \mathbf{X})$ とすると、重み勾配は：
   $$\nabla_{\mathbf{W}} \mathbb{E}[E] = 2 \left( \mathbf{X}^T \mathbf{X} \mathbf{W} - \mathbf{X}^T \mathbf{T} + \frac{1 - \rho}{\rho} \operatorname{diag}(\mathbf{X}^T \mathbf{X}) \mathbf{W} \right) = \mathbf{0}$$
   したがって閉形式解は：
   $$\mathbf{W}^* = \left( \mathbf{X}^T \mathbf{X} + \frac{1 - \rho}{\rho} \operatorname{diag}(\mathbf{X}^T \mathbf{X}) \right)^{-1} \mathbf{X}^T \mathbf{T}$$
   となる。"""))

    cells.append(create_cell("code", r"""# Exercise 9.18 自己検証コード
res_9_18 = verify_exercise_9_18()
print("Exercise 9.18 Verification:", res_9_18)
assert res_9_18["passed"] is True, "Exercise 9.18 failed!"
print("Exercise 9.18 PASSED!")"""))

    # Summary
    cells.append(create_cell("markdown", r"""---
## 全18問 検証サマリー

すべての演習問題（Exercises 9.1 〜 9.18）に対する数学的導出および数値検証が成功しました。
これをもって、**第9章 正則化 (Regularization)** の全ユニット（セクション9.1〜9.6、演習問題）が完全に完了しました。"""))

    cells.append(create_cell("code", r"""# 全問題一括検証テスト
all_results = [
    ("Exercise 9.1", verify_exercise_9_1()["passed"]),
    ("Exercise 9.2", verify_exercise_9_2()["passed"]),
    ("Exercise 9.3", verify_exercise_9_3()["passed"]),
    ("Exercise 9.4", verify_exercise_9_4()["passed"]),
    ("Exercise 9.5", verify_exercise_9_5()["passed"]),
    ("Exercise 9.6", verify_exercise_9_6()["passed"]),
    ("Exercise 9.7", verify_exercise_9_7()["passed"]),
    ("Exercise 9.8", verify_exercise_9_8()["passed"]),
    ("Exercise 9.9", verify_exercise_9_9()["passed"]),
    ("Exercise 9.10", verify_exercise_9_10()["passed"]),
    ("Exercise 9.11", verify_exercise_9_11()["passed"]),
    ("Exercise 9.12", verify_exercise_9_12()["passed"]),
    ("Exercise 9.13", verify_exercise_9_13()["passed"]),
    ("Exercise 9.14", verify_exercise_9_14()["passed"]),
    ("Exercise 9.15", verify_exercise_9_15()["passed"]),
    ("Exercise 9.16", verify_exercise_9_16()["passed"]),
    ("Exercise 9.17", verify_exercise_9_17()["passed"]),
    ("Exercise 9.18", verify_exercise_9_18()["passed"]),
]

for name, passed in all_results:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")

assert all(passed for _, passed in all_results), "Some exercises failed!"
print("\n>>> ALL 18 EXERCISES IN CHAPTER 9 SUCCESSFULLY VERIFIED! <<<")"""))

    return cells


def main():
    cells = build_cells()
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = "9/9_Exercises.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Wrote {out_path} with {len(cells)} cells.")

    print(f"Executing notebook: {sys.executable} -m jupyter nbconvert --to notebook --execute --inplace {out_path}")
    res = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", out_path],
        capture_output=True,
        text=True
    )

    if res.returncode != 0:
        print("Execution failed!")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit(res.returncode)
    else:
        print(f"Successfully executed {out_path} with 0 errors!")


if __name__ == "__main__":
    main()

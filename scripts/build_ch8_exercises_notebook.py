"""Build Chapter 8 Exercises notebook (8/8_Exercises.ipynb) and execute all cells.

Bishop & Bishop (2024), Chapter 8, pp. 250-252.
Exercises 8.1 to 8.18 (all 18 exercises).
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
    cells.append(create_cell("markdown", r"""# 第8章 誤差逆伝播法 (Backpropagation)
## 章末演習問題 (Exercises 8.1 〜 8.18)

本ノートブックは、Christopher M. Bishop & Hugh Bishop (2024)『深層学習：基礎と概念 (Deep Learning: Foundations and Concepts)』第8章「誤差逆伝播法」の**章末演習問題 全18問 (Exercises 8.1 〜 8.18)** に対する完全解答・数学的厳密導出・自己検証コードです。

---

### 演習問題 目次
- **Part 1: 誤差関数勾配の評価と誤差逆伝播法の基礎理論 [Exercises 8.1 〜 8.4]**
  - **Exercise 8.1 (★)**: 連鎖律と誤差伝播方程式 (8.13) $\delta_j = h'(a_j) \sum_k w_{kj} \delta_k$ の厳密証明
  - **Exercise 8.2 (★★)**: 行列・ベクトル記法による多層逆伝播方程式 $\boldsymbol{\delta}^{(l)} = h'(\mathbf{a}^{(l)}) \odot ((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)})$ の定式化
  - **Exercise 8.3 (★)**: テイラー展開による中心差分 $O(\epsilon^2)$ 誤差および奇数次 $O(\epsilon)$ 項相殺の証明 (Eq 8.25)
  - **Exercise 8.4 (★★)**: 入力から出力へのスキップ結合（Skip-layer connections）をもつ2層ネットワークの誤差勾配導出
- **Part 2: ヤコビ行列・厳密ヘッセ行列・外積近似 [Exercises 8.5 〜 8.8]**
  - **Exercise 8.5 (★★★)**: 順方向伝播（Forward propagation）に基づくネットワークのヤコビ行列計算法の定式化
  - **Exercise 8.6 (★★★)**: 2層ネットワークの厳密ヘッセ行列成分 $\delta_k, M_{kk'}$ による導出（層2-層2, 層1-層1, 層2-層1）
  - **Exercise 8.7 (★★★)**: スキップ結合を含む2層ネットワークの厳密ヘッセ行列成分の導出
  - **Exercise 8.8 (★★)**: 多出力ネットワークに対する二乗和誤差の外積（Gauss-Newton）ヘッセ行列近似 $\mathbf{H} \approx \sum_n \mathbf{J}_n^T \mathbf{J}_n$ の導出
- **Part 3: ヘッセ行列近似の統計的性質と逐次更新アルゴリズム [Exercises 8.9 〜 8.12]**
  - **Exercise 8.9 (★★)**: 二乗損失の期待ヘッセ行列が真の条件付き期待値 $y(\mathbf{x}) = \mathbb{E}[t|\mathbf{x}]$ において勾配外積期待値と一致することの証明 (Eq 8.79)
  - **Exercise 8.10 (★★)**: 単出力ロジスティックシグモイド＋二値交差エントロピーの外積ヘッセ行列導出 (Eq 8.41)
  - **Exercise 8.11 (★★)**: $K$ クラスSoftmax＋多クラス交差エントロピーの外積ヘッセ行列導出
  - **Exercise 8.12 (★★★)**: Sherman-Morrison / Woodbury 公式に基づく逆ヘッセ行列のオンライン逐次更新アルゴリズム導出 (Eq 8.80)
- **Part 4: 自動微分 (Autodiff) と数式微分の爆発 [Exercises 8.13 〜 8.18]**
  - **Exercise 8.13 (★★)**: 2層合成soft ReLUの数式微分における式の爆発 (Eq 8.48) の証明と自動微分の対比
  - **Exercise 8.14 (★★)**: ロジスティック写像 $L_n(x)$ における多項式次数の指数関数的増大と自動微分の線形計算量
  - **Exercise 8.15 (★★)**: 例題関数 (8.49) に対するフォワードモード接線方程式 (8.58)-(8.64) の厳密導出
  - **Exercise 8.16 (★★)**: 例題関数 (8.49) に対するリバースモード随伴方程式 (8.70)-(8.76) の厳密導出
  - **Exercise 8.17 (★★★)**: $(x_1=1, x_2=2)$ における解析値 $2 + 2e^2$ とフォワード・リバースモードの完全数値一致検証
  - **Exercise 8.18 (★★)**: 任意方向ベクトル $\mathbf{r}$ に対するヤコビ・ベクトル積 $\mathbf{J}\mathbf{r}$ が1回のフォワードパスで求まることの証明"""))

    # Cell 1: Environment Setup
    cells.append(create_cell("code", r"""# 環境セットアップと共通モジュールのインポート
import os
import sys
import numpy as np

# プロジェクトルートパスの設定
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.exercises_ch8 import (
    verify_exercise_8_1,
    verify_exercise_8_2,
    verify_exercise_8_3,
    verify_exercise_8_4,
    verify_exercise_8_5,
    verify_exercise_8_6,
    verify_exercise_8_7,
    verify_exercise_8_8,
    verify_exercise_8_9,
    verify_exercise_8_10,
    verify_exercise_8_11,
    verify_exercise_8_12,
    verify_exercise_8_13,
    verify_exercise_8_14,
    verify_exercise_8_15,
    verify_exercise_8_16,
    verify_exercise_8_17,
    verify_exercise_8_18,
    verify_all_ch8_exercises,
)
from common.automatic_differentiation import DualNumber, Node

print("Chapter 8 Exercises module loaded successfully.")"""))

    # Part 1 Header
    cells.append(create_cell("markdown", r"""---
## Part 1: 誤差関数勾配の評価と誤差逆伝播法の基礎理論 [Exercises 8.1 〜 8.4]"""))

    # Exercise 8.1
    cells.append(create_cell("markdown", r"""### Exercise 8.1: 連鎖律と誤差逆伝播方程式 (8.13) の厳密証明

#### 問題の要約
教科書の式 (8.5), (8.6), (8.8), (8.12) を用いて、誤差関数の微分を評価するための逆伝播方程式 (8.13) を証明せよ。

#### 数学的導出と証明
教科書の基本定義式は以下の通りである：
1. 各ユニットの入力活性化値 (8.5):
   $$a_j = \sum_i w_{ji} z_i$$
2. 活性化関数を通したユニット出力 (8.6):
   $$z_j = h(a_j)$$
3. 誤差勾配 $\delta_j$ の定義 (8.8):
   $$\delta_j \equiv \frac{\partial E_n}{\partial a_j}$$
4. 多変数の連鎖律（Chain Rule）(8.12):
   $$\frac{\partial E_n}{\partial a_j} = \sum_k \frac{\partial E_n}{\partial a_k} \frac{\partial a_k}{\partial a_j}$$
   ここで和 $\sum_k$ はユニット $j$ から接続を受け取る後続層のすべてのユニット $k$ にわたる。

後続層のユニット $k$ の活性化値 $a_k$ は以下のように書ける：
$$a_k = \sum_l w_{kl} z_l = \sum_l w_{kl} h(a_l)$$
これを $a_j$ で偏微分すると、$l = j$ の項のみが残り：
$$\frac{\partial a_k}{\partial a_j} = \frac{\partial}{\partial a_j} \left( w_{kj} h(a_j) \right) = w_{kj} h'(a_j)$$
これを連鎖律 (8.12) に代入すると：
$$\delta_j = \frac{\partial E_n}{\partial a_j} = \sum_k \delta_k \left( w_{kj} h'(a_j) \right) = h'(a_j) \sum_k w_{kj} \delta_k$$
となり、逆伝播方程式 (8.13) が導かれる。

さらに、重みパラメータ $w_{ji}$ に関する誤差勾配は：
$$\frac{\partial E_n}{\partial w_{ji}} = \frac{\partial E_n}{\partial a_j} \frac{\partial a_j}{\partial w_{ji}} = \delta_j z_i$$
である。"""))

    cells.append(create_cell("code", r"""# Exercise 8.1 の数値検証
res_8_1 = verify_exercise_8_1()
print(f"Exercise 8.1 Verified: {res_8_1['verified']}")
print(f"Max Gradient Error on W1 (vs Central Diff): {res_8_1['max_err_W1']:.4e}")
assert res_8_1["verified"], "Exercise 8.1 verification failed!"
"""))

    # Exercise 8.2
    cells.append(create_cell("markdown", r"""### Exercise 8.2: 行列・ベクトル記法による多層逆伝播方程式の定式化

#### 問題の要約
層構造をもつネットワークを考え、順伝播方程式 (6.19) から出発して逆伝播方程式 (8.13) を行列記法で書き直せ。結果に行列の転置が含まれることに注目せよ。

#### 数学的導出と証明
層 $l-1$ から層 $l$ への順伝播方程式 (6.19) は：
$$\mathbf{a}^{(l)} = \mathbf{W}^{(l)} \mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}, \quad \mathbf{z}^{(l)} = h(\mathbf{a}^{(l)})$$
要素ごとの逆伝播式 (8.13) は：
$$\delta_j^{(l)} = h'(a_j^{(l)}) \sum_k w_{kj}^{(l+1)} \delta_k^{(l+1)}$$
ここで和 $\sum_k w_{kj}^{(l+1)} \delta_k^{(l+1)}$ は、重み行列 $\mathbf{W}^{(l+1)}$ の第 $j$ 列と誤差ベクトル $\boldsymbol{\delta}^{(l+1)}$ の内積である。
したがって、転置行列 $(\mathbf{W}^{(l+1)})^T$ を用いると、第 $j$ 成分は $((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)})_j$ となる。

これより、アダマール積（要素ごとの積 $\odot$）を用いてベクトル形式で表すと：
$$\boldsymbol{\delta}^{(l)} = h'(\mathbf{a}^{(l)}) \odot \left( (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \right)$$
重み行列およびバイアスベクトルに対する勾配は、外積を用いて以下のように簡潔に表される：
$$\frac{\partial E_n}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{z}^{(l-1)})^T, \quad \frac{\partial E_n}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$
順伝播では $\mathbf{W}^{(l)}$ による乗算が行われるのに対し、逆伝播では転置行列 $(\mathbf{W}^{(l+1)})^T$ による乗算が行われる。"""))

    cells.append(create_cell("code", r"""# Exercise 8.2 の数値検証
res_8_2 = verify_exercise_8_2()
print(f"Exercise 8.2 Verified: {res_8_2['verified']}")
print(f"Max Gradient Error on W0: {res_8_2['max_err_W0']:.4e}")
print(f"Delta shapes across layers: {res_8_2['deltas_shapes']}")
assert res_8_2["verified"], "Exercise 8.2 verification failed!"
"""))

    # Exercise 8.3
    cells.append(create_cell("markdown", r"""### Exercise 8.3: テイラー展開による中心差分 $O(\epsilon^2)$ 誤差の証明

#### 問題の要約
テイラー展開を用いて、中心差分公式 (8.25) の右辺において $O(\epsilon)$ の誤差項が相殺されて消えることを確かめよ。

#### 数学的導出と証明
点 $w$ の近傍における誤差関数 $E(w)$ の前方向および後方向のテイラー展開は：
$$E(w + \epsilon) = E(w) + \epsilon E'(w) + \frac{\epsilon^2}{2} E''(w) + \frac{\epsilon^3}{6} E'''(w) + O(\epsilon^4)$$
$$E(w - \epsilon) = E(w) - \epsilon E'(w) + \frac{\epsilon^2}{2} E''(w) - \frac{\epsilon^3}{6} E'''(w) + O(\epsilon^4)$$
これら2式の差をとると、偶数次項（$E(w)$ および $\frac{\epsilon^2}{2} E''(w)$）が厳密に相殺される：
$$E(w + \epsilon) - E(w - \epsilon) = 2 \epsilon E'(w) + \frac{\epsilon^3}{3} E'''(w) + O(\epsilon^5)$$
両辺を $2\epsilon$ で割ると：
$$\frac{E(w + \epsilon) - E(w - \epsilon)}{2\epsilon} = E'(w) + \frac{\epsilon^2}{6} E'''(w) + O(\epsilon^4) = E'(w) + O(\epsilon^2)$$
したがって、中心差分による数値微分の打ち切り誤差は $O(\epsilon^2)$ であり、$O(\epsilon)$ の項は完全に相殺されている。
対照的に、片側前進差分では：
$$\frac{E(w + \epsilon) - E(w)}{\epsilon} = E'(w) + \frac{\epsilon}{2} E''(w) + O(\epsilon^2) = E'(w) + O(\epsilon)$$
となり、$O(\epsilon)$ の誤差が残る。"""))

    cells.append(create_cell("code", r"""# Exercise 8.3 の数値検証
res_8_3 = verify_exercise_8_3()
print(f"Exercise 8.3 Verified: {res_8_3['verified']}")
print(f"Log-log slope for Central Differences: {res_8_3['slope_central']:.4f} (Theoretical: 2.0)")
print(f"Log-log slope for Forward Differences: {res_8_3['slope_forward']:.4f} (Theoretical: 1.0)")
assert res_8_3["verified"], "Exercise 8.3 verification failed!"
"""))

    # Exercise 8.4
    cells.append(create_cell("markdown", r"""### Exercise 8.4: スキップ結合（Skip Connections）をもつ2層ネットワークの誤差勾配導出

#### 問題の要約
入力から出力ユニットへ直接接続するスキップ結合パラメータをもつ2層ネットワークを考え、これらの追加パラメータに関する誤差関数の微分の表式を導出せよ。

#### 数学的導出と証明
ネットワークのモデル構造：
1. 隠れ層：
   $$a_j = \sum_{i=1}^D w_{ji}^{(1)} x_i + b_j^{(1)}, \quad z_j = h(a_j)$$
2. スキップ結合を含む出力層：
   $$a_k = \sum_{j=1}^M w_{kj}^{(2)} z_j + \sum_{i=1}^D w_{ki}^{(s)} x_i + b_k^{(2)}, \quad y_k = \sigma(a_k)$$
   ここで $w_{ki}^{(s)}$ が入力 $i$ から出力 $k$ へのスキップ結合の重みパラメータである。

二乗和誤差または交差エントロピー誤差 $E_n$ に対し、出力ユニットの誤差信号を：
$$\delta_k \equiv \frac{\partial E_n}{\partial a_k}$$
と定義する。スキップ結合重み $w_{ki}^{(s)}$ に関する連鎖律は：
$$\frac{\partial E_n}{\partial w_{ki}^{(s)}} = \frac{\partial E_n}{\partial a_k} \frac{\partial a_k}{\partial w_{ki}^{(s)}} = \delta_k x_i$$
なお、隠れユニット $j$ への誤差信号 $\delta_j$ の伝播式は：
$$\delta_j = h'(a_j) \sum_k w_{kj}^{(2)} \delta_k$$
であり、スキップ結合は隠れ層をバイパスするため、隠れ層パラメータの逆伝播式には直接影響を与えない。"""))

    cells.append(create_cell("code", r"""# Exercise 8.4 の数値検証
res_8_4 = verify_exercise_8_4()
print(f"Exercise 8.4 Verified: {res_8_4['verified']}")
print(f"Max Gradient Error on Skip Weights Ws: {res_8_4['max_err_Ws']:.4e}")
assert res_8_4["verified"], "Exercise 8.4 verification failed!"
"""))

    # Part 2 Header
    cells.append(create_cell("markdown", r"""---
## Part 2: ヤコビ行列・厳密ヘッセ行列・外積近似 [Exercises 8.5 〜 8.8]"""))

    # Exercise 8.5
    cells.append(create_cell("markdown", r"""### Exercise 8.5: 順方向伝播に基づくヤコビ行列計算法の定式化

#### 問題の要約
8.1.5項で逆伝播により導出されたヤコビ行列 $J_{ki} = \frac{\partial y_k}{\partial x_i}$ に対し、順方向伝播に基づく代替アルゴリズムを導出せよ。

#### 数学的導出と証明
入力 $x_i$ を基準とする接線変数（Tangent variables）の順方向伝播として定式化する：
1. 入力層：
   $$\frac{\partial x_m}{\partial x_i} = \delta_{mi} \quad (\text{Kronecker delta})$$
2. 第1層活性化値：
   $$\frac{\partial a_j}{\partial x_i} = \sum_m w_{jm}^{(1)} \frac{\partial x_m}{\partial x_i} = w_{ji}^{(1)}$$
3. 第1層出力：
   $$\frac{\partial z_j}{\partial x_i} = h'(a_j) \frac{\partial a_j}{\partial x_i} = h'(a_j) w_{ji}^{(1)}$$
4. 出力層活性化値：
   $$\frac{\partial a_k}{\partial x_i} = \sum_j w_{kj}^{(2)} \frac{\partial z_j}{\partial x_i} = \sum_j w_{kj}^{(2)} h'(a_j) w_{ji}^{(1)}$$
5. ネットワーク出力（ヤコビ行列の各成分）：
   $$J_{ki} = \frac{\partial y_k}{\partial x_i} = \sigma'(a_k) \frac{\partial a_k}{\partial x_i} = \sigma'(a_k) \sum_j w_{kj}^{(2)} h'(a_j) w_{ji}^{(1)}$$
各入力 $i \in \{1, \dots, D\}$ についてこの順伝播を1回実行することで、ヤコビ行列の第 $i$ 列 $\mathbf{J}_{:, i}$ が求まり、計 $D$ 回の順伝播で $K \times D$ のヤコビ行列全体が求まる。"""))

    cells.append(create_cell("code", r"""# Exercise 8.5 の数値検証
res_8_5 = verify_exercise_8_5()
print(f"Exercise 8.5 Verified: {res_8_5['verified']}")
print(f"Max Error between Forward Jacobian and Finite Differences: {res_8_5['max_err']:.4e}")
assert res_8_5["verified"], "Exercise 8.5 verification failed!"
"""))

    # Exercise 8.6
    cells.append(create_cell("markdown", r"""### Exercise 8.6: 2層ネットワークの厳密ヘッセ行列成分の導出

#### 問題の要約
2層ニューラルネットワークに対し、以下の量を定義する：
$$\delta_k = \frac{\partial E_n}{\partial a_k}, \quad M_{kk'} \equiv \frac{\partial^2 E_n}{\partial a_k \partial a_{k'}} \quad (8.77)$$
(i) 両方の重みが第2層にある場合、(ii) 両方の重みが第1層にある場合、(iii) 各層に1つずつ重みがある場合について、ヘッセ行列の要素を $\delta_k$ と $M_{kk'}$ を用いて表せ。

#### 数学的導出と証明
第2層の勾配は $\frac{\partial E_n}{\partial w_{kj}^{(2)}} = \delta_k z_j$、第1層の勾配は $\frac{\partial E_n}{\partial w_{ji}^{(1)}} = \delta_j x_i$（ただし $\delta_j = h'(a_j) \sum_k w_{kj}^{(2)} \delta_k$）である。

1. **(i) 両方の重みが第2層にある場合**:
   $$\frac{\partial^2 E_n}{\partial w_{kj}^{(2)} \partial w_{k'j'}^{(2)}} = \frac{\partial}{\partial w_{k'j'}^{(2)}} (\delta_k z_j) = z_j \sum_l \frac{\partial \delta_k}{\partial a_l} \frac{\partial a_l}{\partial w_{k'j'}^{(2)}} = z_j z_{j'} M_{kk'}$$

2. **(ii) 両方の重みが第1層にある場合**:
   $$\frac{\partial^2 E_n}{\partial w_{ji}^{(1)} \partial w_{j'i'}^{(1)}} = x_i \frac{\partial \delta_j}{\partial w_{j'i'}^{(1)}} = x_i \frac{\partial}{\partial w_{j'i'}^{(1)}} \left[ h'(a_j) \sum_k w_{kj}^{(2)} \delta_k \right]$$
   積の微分則を適用すると：
   $$\frac{\partial^2 E_n}{\partial w_{ji}^{(1)} \partial w_{j'i'}^{(1)}} = x_i x_{i'} \left[ \delta_{jj'} h''(a_j) \sum_k w_{kj}^{(2)} \delta_k + h'(a_j) h'(a_{j'}) \sum_k \sum_{k'} w_{kj}^{(2)} w_{k'j'}^{(2)} M_{kk'} \right]$$

3. **(iii) 各層に1つずつ重みがある場合**:
   $$\frac{\partial^2 E_n}{\partial w_{kj}^{(2)} \partial w_{j'i'}^{(1)}} = \frac{\partial}{\partial w_{j'i'}^{(1)}} (\delta_k z_j) = z_j \frac{\partial \delta_k}{\partial w_{j'i'}^{(1)}} + \delta_k \frac{\partial z_j}{\partial w_{j'i'}^{(1)}}$$
   $\frac{\partial z_j}{\partial w_{j'i'}^{(1)}} = \delta_{jj'} h'(a_j) x_{i'}$ および $\frac{\partial \delta_k}{\partial w_{j'i'}^{(1)}} = \sum_{k'} M_{kk'} w_{k'j'}^{(2)} h'(a_{j'}) x_{i'}$ を代入すると：
   $$\frac{\partial^2 E_n}{\partial w_{kj}^{(2)} \partial w_{j'i'}^{(1)}} = x_{i'} \left[ \delta_{jj'} \delta_k h'(a_j) + z_j h'(a_{j'}) \sum_{k'} w_{k'j'}^{(2)} M_{kk'} \right]$$"""))

    cells.append(create_cell("code", r"""# Exercise 8.6 の数値検証
res_8_6 = verify_exercise_8_6()
print(f"Exercise 8.6 Verified: {res_8_6['verified']}")
print(f"Error on Layer 2 - Layer 2 Hessian: {res_8_6['err_H22']:.4e}")
print(f"Error on Layer 2 - Layer 1 Hessian: {res_8_6['err_H21']:.4e}")
assert res_8_6["verified"], "Exercise 8.6 verification failed!"
"""))

    # Exercise 8.7
    cells.append(create_cell("markdown", r"""### Exercise 8.7: スキップ結合を含む厳密ヘッセ行列の導出

#### 問題の要約
Exercise 8.6 の2層ネットワークの厳密ヘッセ行列の表式を、入力から出力へのスキップ結合 $w_{ki}^{(s)}$ を含む場合に拡張せよ。

#### 数学的導出と証明
スキップ結合が存在する場合、出力ユニットの活性化値は：
$$a_k = \sum_j w_{kj}^{(2)} z_j + \sum_i w_{ki}^{(s)} x_i + b_k^{(2)}$$
であり、1階微分は $\frac{\partial E_n}{\partial w_{ki}^{(s)}} = \delta_k x_i$ である。

追加される2階微分の成分は以下の3通りである：
1. **スキップ結合同士の2階微分**:
   $$\frac{\partial^2 E_n}{\partial w_{ki}^{(s)} \partial w_{k'i'}^{(s)}} = x_i \frac{\partial \delta_k}{\partial w_{k'i'}^{(s)}} = x_i \sum_l M_{kl} \frac{\partial a_l}{\partial w_{k'i'}^{(s)}} = x_i x_{i'} M_{kk'}$$

2. **スキップ結合と第2層重みの2階微分**:
   $$\frac{\partial^2 E_n}{\partial w_{ki}^{(s)} \partial w_{k'j}^{(2)}} = x_i \frac{\partial \delta_k}{\partial w_{k'j}^{(2)}} = x_i z_j M_{kk'}$$

3. **スキップ結合と第1層重みの2階微分**:
   $$\frac{\partial^2 E_n}{\partial w_{ki}^{(s)} \partial w_{j'i'}^{(1)}} = x_i \frac{\partial \delta_k}{\partial w_{j'i'}^{(1)}} = x_i x_{i'} h'(a_{j'}) \sum_{k'} M_{kk'} w_{k'j'}^{(2)}$$"""))

    cells.append(create_cell("code", r"""# Exercise 8.7 の数値検証
res_8_7 = verify_exercise_8_7()
print(f"Exercise 8.7 Verified: {res_8_7['verified']}")
print(f"Error on Skip-Skip Hessian: {res_8_7['err_H_skip_skip']:.4e}")
assert res_8_7["verified"], "Exercise 8.7 verification failed!"
"""))

    # Exercise 8.8
    cells.append(create_cell("markdown", r"""### Exercise 8.8: 多出力二乗和誤差に対する外積ヘッセ行列近似の導出

#### 問題の要約
単出力ネットワークの二乗和誤差に対するヘッセ行列の外積近似 (8.40) を、多出力ネットワークの場合に拡張せよ。

#### 数学的導出と証明
$K$ 次元の多出力をもつネットワークの二乗和誤差は：
$$E_n = \frac{1}{2} \sum_{k=1}^K (y_{nk} - t_{nk})^2$$
パラメータ $w_r$ に関する1階微分は：
$$\frac{\partial E_n}{\partial w_r} = \sum_{k=1}^K (y_{nk} - t_{nk}) \frac{\partial y_{nk}}{\partial w_r}$$
これをさらに $w_s$ で微分すると：
$$\frac{\partial^2 E_n}{\partial w_r \partial w_s} = \sum_{k=1}^K \frac{\partial y_{nk}}{\partial w_r} \frac{\partial y_{nk}}{\partial w_s} + \sum_{k=1}^K (y_{nk} - t_{nk}) \frac{\partial^2 y_{nk}}{\partial w_r \partial w_s}$$
モデルが真の条件付き期待値をよく近似している場合、残差 $(y_{nk} - t_{nk})$ は平均 0 の確率変数であり、データ点全体の総和をとると第2項は相殺されて無視できる。
したがって、全データセットに対するヘッセ行列の外積近似（Gauss-Newtonヘッセ行列）は：
$$H_{rs} \approx \sum_{n=1}^N \sum_{k=1}^K \frac{\partial y_{nk}}{\partial w_r} \frac{\partial y_{nk}}{\partial w_s}$$
行列記法では、重みに関する出力のヤコビ行列 $\mathbf{J}_n = \frac{\partial \mathbf{y}_n}{\partial \mathbf{w}} \in \mathbb{R}^{K \times W}$ を用いて：
$$\mathbf{H} \approx \sum_{n=1}^N \mathbf{J}_n^T \mathbf{J}_n$$
と表される。"""))

    cells.append(create_cell("code", r"""# Exercise 8.8 の数値検証
res_8_8 = verify_exercise_8_8()
print(f"Exercise 8.8 Verified: {res_8_8['verified']}")
print(f"Max Error on Multi-output Outer Product Hessian: {res_8_8['max_err']:.4e}")
assert res_8_8["verified"], "Exercise 8.8 verification failed!"
"""))

    # Part 3 Header
    cells.append(create_cell("markdown", r"""---
## Part 3: ヘッセ行列近似の統計的性質と逐次更新アルゴリズム [Exercises 8.9 〜 8.12]"""))

    # Exercise 8.9
    cells.append(create_cell("markdown", r"""### Exercise 8.9: 二乗損失の期待ヘッセ行列が真の条件付き期待値において勾配外積と厳密一致することの証明

#### 問題の要約
二乗損失関数 (8.78):
$$E(\mathbf{w}) = \frac{1}{2} \iint \{y(\mathbf{x}, \mathbf{w}) - t\}^2 p(\mathbf{x}, t) \, d\mathbf{x} \, dt$$
を考え、誤差を最小化する関数が条件付き期待値 $y(\mathbf{x}, \mathbf{w}) = \mathbb{E}[t|\mathbf{x}]$ であることを用いて、期待ヘッセ行列が (8.79):
$$\frac{\partial^2 E}{\partial w_r \partial w_s} = \int \frac{\partial y}{\partial w_r} \frac{\partial y}{\partial w_s} p(\mathbf{x}) \, d\mathbf{x}$$
で与えられることを証明せよ。

#### 数学的導出と証明
$E(\mathbf{w})$ を $w_r$ で微分すると：
$$\frac{\partial E}{\partial w_r} = \iint (y(\mathbf{x}, \mathbf{w}) - t) \frac{\partial y(\mathbf{x}, \mathbf{w})}{\partial w_r} p(\mathbf{x}, t) \, d\mathbf{x} \, dt$$
さらに $w_s$ で微分すると：
$$\frac{\partial^2 E}{\partial w_r \partial w_s} = \iint \frac{\partial y}{\partial w_r} \frac{\partial y}{\partial w_s} p(\mathbf{x}, t) \, d\mathbf{x} \, dt + \iint (y(\mathbf{x}, \mathbf{w}) - t) \frac{\partial^2 y}{\partial w_r \partial w_s} p(\mathbf{x}, t) \, d\mathbf{x} \, dt$$
第1項の被積分関数は $t$ に依存しないため、$t$ について周辺化積分すると $p(\mathbf{x})$ が得られる：
$$\int \frac{\partial y}{\partial w_r} \frac{\partial y}{\partial w_s} p(\mathbf{x}) \, d\mathbf{x}$$
第2項について、$p(\mathbf{x}, t) = p(t|\mathbf{x}) p(\mathbf{x})$ と分解して $t$ について先に積分すると：
$$\int \left[ \int (y(\mathbf{x}, \mathbf{w}) - t) p(t|\mathbf{x}) \, dt \right] \frac{\partial^2 y}{\partial w_r \partial w_s} p(\mathbf{x}) \, d\mathbf{x}$$
最適な回帰関数 $y(\mathbf{x}, \mathbf{w}) = \mathbb{E}[t|\mathbf{x}] = \int t p(t|\mathbf{x}) \, dt$ を代入すると、大括弧内の積分は：
$$\int (y(\mathbf{x}, \mathbf{w}) - t) p(t|\mathbf{x}) \, dt = y(\mathbf{x}, \mathbf{w}) - \mathbb{E}[t|\mathbf{x}] = 0$$
したがって第2項は恒等的にゼロとなり、期待ヘッセ行列は厳密に勾配外積の期待値 (8.79) に一致する。"""))

    cells.append(create_cell("code", r"""# Exercise 8.9 の数値検証
res_8_9 = verify_exercise_8_9()
print(f"Exercise 8.9 Verified: {res_8_9['verified']}")
print(f"Outer Product Term: {res_8_9['outer_product_term']:.4f}")
print(f"Residual Curvature Term: {res_8_9['residual_curvature_term']:.4e} (Converges to 0)")
print(f"Ratio of Residual to Outer Product: {res_8_9['ratio_residual_to_outer']:.4e}")
assert res_8_9["verified"], "Exercise 8.9 verification failed!"
"""))

    # Exercise 8.10
    cells.append(create_cell("markdown", r"""### Exercise 8.10: 単出力ロジスティックシグモイド＋二値交差エントロピーの外積ヘッセ行列導出

#### 問題の要約
単一の出力ユニットをもち、ロジスティックシグモイド活性化関数と交差エントロピー誤差関数をもつネットワークに対する外積ヘッセ行列近似 (8.41) を導出せよ。

#### 数学的導出と証明
モデルの出力は $y_n = \sigma(a_n)$、二値交差エントロピー誤差は：
$$E_n = - [t_n \ln y_n + (1 - t_n) \ln(1 - y_n)]$$
活性化値 $a_n$ に関する1階微分は：
$$\frac{\partial E_n}{\partial a_n} = y_n - t_n$$
パラメータ $w_r$ に関する勾配は：
$$\frac{\partial E_n}{\partial w_r} = (y_n - t_n) \frac{\partial a_n}{\partial w_r}$$
これをさらに $w_s$ で微分すると：
$$\frac{\partial^2 E_n}{\partial w_r \partial w_s} = \frac{\partial y_n}{\partial w_s} \frac{\partial a_n}{\partial w_r} + (y_n - t_n) \frac{\partial^2 a_n}{\partial w_r \partial w_s}$$
$\sigma'(a_n) = y_n(1 - y_n)$ より $\frac{\partial y_n}{\partial w_s} = y_n(1 - y_n) \frac{\partial a_n}{\partial w_s}$ である。
$y_n = \mathbb{E}[t_n|\mathbf{x}_n]$ のもとで残差項 $(y_n - t_n)$ の期待値はゼロとなるため、第2項を省略すると：
$$H_{rs} \approx \sum_{n=1}^N y_n (1 - y_n) \frac{\partial a_n}{\partial w_r} \frac{\partial a_n}{\partial w_s}$$
ベクトル記法では、教科書の式 (8.41) が得られる：
$$\mathbf{H} \approx \sum_{n=1}^N y_n (1 - y_n) \nabla a_n \nabla a_n^T$$"""))

    cells.append(create_cell("code", r"""# Exercise 8.10 の数値検証
res_8_10 = verify_exercise_8_10()
print(f"Exercise 8.10 Verified: {res_8_10['verified']}")
print(f"Max Error on Logistic Sigmoid Outer Product Hessian: {res_8_10['max_err']:.4e}")
assert res_8_10["verified"], "Exercise 8.10 verification failed!"
"""))

    # Exercise 8.11
    cells.append(create_cell("markdown", r"""### Exercise 8.11: $K$ クラスSoftmax＋交差エントロピーの外積ヘッセ行列導出

#### 問題の要約
$K$ 個の出力をもち、Softmax活性化関数と多クラス交差エントロピー誤差関数をもつネットワークに対する外積ヘッセ行列近似の表式を導出せよ。

#### 数学的導出と証明
Softmax出力は $y_{nk} = \frac{\exp(a_{nk})}{\sum_j \exp(a_{nj})}$、誤差関数は $E_n = - \sum_{k=1}^K t_{nk} \ln y_{nk}$ である。
1階微分は：
$$\frac{\partial E_n}{\partial a_{nk}} = y_{nk} - t_{nk}$$
パラメータ $w_r$ に関する勾配は：
$$\frac{\partial E_n}{\partial w_r} = \sum_{k=1}^K (y_{nk} - t_{nk}) \frac{\partial a_{nk}}{\partial w_r}$$
これを $w_s$ で微分すると：
$$\frac{\partial^2 E_n}{\partial w_r \partial w_s} = \sum_{k=1}^K \frac{\partial y_{nk}}{\partial w_s} \frac{\partial a_{nk}}{\partial w_r} + \sum_{k=1}^K (y_{nk} - t_{nk}) \frac{\partial^2 a_{nk}}{\partial w_r \partial w_s}$$
Softmaxの微分は $\frac{\partial y_{nk}}{\partial a_{nl}} = y_{nk}(\delta_{kl} - y_{nl})$ であるため：
$$\frac{\partial y_{nk}}{\partial w_s} = \sum_{l=1}^K y_{nk}(\delta_{kl} - y_{nl}) \frac{\partial a_{nl}}{\partial w_s}$$
残差 $(y_{nk} - t_{nk})$ を無視すると、外積近似は：
$$H_{rs} \approx \sum_{n=1}^N \sum_{k=1}^K \sum_{l=1}^K y_{nk}(\delta_{kl} - y_{nl}) \frac{\partial a_{nk}}{\partial w_r} \frac{\partial a_{nl}}{\partial w_s}$$
行列記法では、$\mathbf{R}_n = \text{diag}(\mathbf{y}_n) - \mathbf{y}_n \mathbf{y}_n^T$ および活性化ヤコビ行列 $\mathbf{J}_{a, n} = \frac{\partial \mathbf{a}_n}{\partial \mathbf{w}} \in \mathbb{R}^{K \times W}$ を用いて：
$$\mathbf{H} \approx \sum_{n=1}^N \mathbf{J}_{a, n}^T \left( \text{diag}(\mathbf{y}_n) - \mathbf{y}_n \mathbf{y}_n^T \right) \mathbf{J}_{a, n}$$
と表される。"""))

    cells.append(create_cell("code", r"""# Exercise 8.11 の数値検証
res_8_11 = verify_exercise_8_11()
print(f"Exercise 8.11 Verified: {res_8_11['verified']}")
print(f"Max Error on Softmax Outer Product Hessian: {res_8_11['max_err']:.4e}")
assert res_8_11["verified"], "Exercise 8.11 verification failed!"
"""))

    # Exercise 8.12
    cells.append(create_cell("markdown", r"""### Exercise 8.12: Sherman-Morrison 公式に基づく逆ヘッセ行列のオンライン逐次更新アルゴリズム

#### 問題の要約
Woodburyの公式の特殊形である Sherman-Morrison 行列恒等式 (8.80):
$$(\mathbf{M} + \mathbf{v} \mathbf{v}^T)^{-1} = \mathbf{M}^{-1} - \frac{(\mathbf{M}^{-1} \mathbf{v}) (\mathbf{v}^T \mathbf{M}^{-1})}{1 + \mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}}$$
をヘッセ行列の外積近似 (8.40) に適用し、訓練データを1回通過するだけで逆ヘッセ行列を逐次更新して計算するアルゴリズムを導出せよ。

#### 数学的導出と証明
ヘッセ行列の外積近似に正則化項 $\alpha \mathbf{I}$ を加えた推定量は：
$$\mathbf{H}_N = \alpha \mathbf{I} + \sum_{n=1}^N \mathbf{g}_n \mathbf{g}_n^T$$
ここで $\mathbf{g}_n = \nabla y_n$ である。データ点 $n$ までの累積ヘッセ行列 $\mathbf{H}_n$ は以下の再帰式を満たす：
$$\mathbf{H}_n = \mathbf{H}_{n-1} + \mathbf{g}_n \mathbf{g}_n^T, \quad \mathbf{H}_0 = \alpha \mathbf{I}$$
逆行列 $\mathbf{V}_n \equiv \mathbf{H}_n^{-1}$ に対し、式 (8.80) において $\mathbf{M} = \mathbf{H}_{n-1}, \mathbf{v} = \mathbf{g}_n$ と置くと：
$$\mathbf{V}_n = \mathbf{V}_{n-1} - \frac{\mathbf{V}_{n-1} \mathbf{g}_n \mathbf{g}_n^T \mathbf{V}_{n-1}}{1 + \mathbf{g}_n^T \mathbf{V}_{n-1} \mathbf{g}_n}$$

#### 逐次更新アルゴリズム
1. 初期化：$\mathbf{V}_0 = \frac{1}{\alpha} \mathbf{I}$
2. $n = 1, \dots, N$ について：
   - ベクトル $\mathbf{u}_n = \mathbf{V}_{n-1} \mathbf{g}_n$ を計算（計算量 $O(W^2)$）
   - スカラー $\gamma_n = 1 + \mathbf{g}_n^T \mathbf{u}_n$ を計算（計算量 $O(W)$）
   - ランク1更新：$\mathbf{V}_n = \mathbf{V}_{n-1} - \frac{1}{\gamma_n} \mathbf{u}_n \mathbf{u}_n^T$（計算量 $O(W^2)$）
3. 最終的に得られる $\mathbf{V}_N$ が逆ヘッセ行列の近似となる。

一括計算での逆行列計算が $O(W^3)$ を要するのに対し、本アルゴリズムはデータ点あたり $O(W^2)$ の計算量で済み、全データセットに対して $O(NW^2)$ の極めて効率的な計算が可能となる。"""))

    cells.append(create_cell("code", r"""# Exercise 8.12 の数値検証
res_8_12 = verify_exercise_8_12()
print(f"Exercise 8.12 Verified: {res_8_12['verified']}")
print(f"Max Absolute Error: {res_8_12['max_abs_err']:.4e}")
print(f"Max Relative Error (vs Exact Batch Inv): {res_8_12['max_rel_err']:.4e}")
assert res_8_12["verified"], "Exercise 8.12 verification failed!"
"""))

    # Part 4 Header
    cells.append(create_cell("markdown", r"""---
## Part 4: 自動微分 (Autodiff) と数式微分の爆発 [Exercises 8.13 〜 8.18]"""))

    # Exercise 8.13
    cells.append(create_cell("markdown", r"""### Exercise 8.13: 2層合成soft ReLUの数式微分における式の爆発の検証

#### 問題の要約
式 (8.47) の関数の微分が式 (8.48) で与えられることを確かめよ。

#### 数学的導出と証明
soft ReLU（softplus）関数とその微分は：
$$h(a) = \ln(1 + e^a), \quad h'(a) = \frac{e^a}{1 + e^a} = \sigma(a)$$
関数 (8.47) は以下のように合成されている：
$$y(x) = h(w_2 h(w_1 x + b_1) + b_2)$$
中間変数を定義する：
$$u = w_1 x + b_1, \quad v = h(u) = \ln(1 + e^u)$$
$$z = w_2 v + b_2 = w_2 \ln(1 + e^{w_1 x + b_1}) + b_2, \quad y = h(z) = \ln(1 + e^z)$$
連鎖律により $w_1$ に関する微分は：
$$\frac{\partial y}{\partial w_1} = h'(z) \cdot \frac{\partial z}{\partial w_1} = \frac{e^z}{1 + e^z} \cdot w_2 h'(u) x = w_2 x \frac{e^u}{1 + e^u} \frac{e^z}{1 + e^z} = \frac{w_2 x \exp(u + z)}{(1 + e^u)(1 + e^z)}$$
分子の指数部分 $u + z$ を展開すると：
$$u + z = w_1 x + b_1 + b_2 + w_2 \ln[1 + e^{w_1 x + b_1}]$$
分母は：
$$(1 + e^u)(1 + e^z) = (1 + e^{w_1 x + b_1}) \left(1 + \exp\left(b_2 + w_2 \ln[1 + e^{w_1 x + b_1}]\right)\right)$$
したがって、式 (8.48):
$$\frac{\partial y}{\partial w_1} = \frac{w_2 x \exp\left(w_1 x + b_1 + b_2 + w_2 \ln[1 + e^{w_1 x + b_1}]\right)}{(1 + e^{w_1 x + b_1}) \left(1 + \exp\left(b_2 + w_2 \ln[1 + e^{w_1 x + b_1}]\right)\right)}$$
が厳密に証明された。元の関数に比べて数式が極度に肥大化し（式の爆発）、同一の小式が複数回重複出現していることがわかる。"""))

    cells.append(create_cell("code", r"""# Exercise 8.13 の数値検証
res_8_13 = verify_exercise_8_13()
print(f"Exercise 8.13 Verified: {res_8_13['verified']}")
print(f"Symbolic Value (Eq 8.48): {res_8_13['symbolic_value']:.10f}")
print(f"Autodiff Value (DualNumber): {res_8_13['autodiff_value']:.10f}")
print(f"Error between Symbolic and Autodiff: {res_8_13['err_autodiff']:.4e}")
assert res_8_13["verified"], "Exercise 8.13 verification failed!"
"""))

    # Exercise 8.14
    cells.append(create_cell("markdown", r"""### Exercise 8.14: ロジスティック写像における多項式次数の指数関数的増大

#### 問題の要約
ロジスティック写像 $L_{n+1}(x) = 4 L_n(x)(1 - L_n(x))$ （$L_1(x) = x$）に対し、$L_2(x), L_3(x), L_4(x)$ の評価トレース式および導関数 $L'_1(x), L'_2(x), L'_3(x), L'_4(x)$ の表式を書き下し、導関数の数式複雑度の増大を観察せよ。

#### 数学的導出と証明
1. **評価トレース式**:
   $$L_1(x) = x$$
   $$L_2(x) = 4 x (1 - x) = 4x - 4x^2$$
   $$L_3(x) = 4 L_2(x)(1 - L_2(x)) = 4(4x - 4x^2)(1 - 4x + 4x^2)$$
   $$L_4(x) = 4 L_3(x)(1 - L_3(x))$$

2. **導関数の再帰式**:
   $$L'_1(x) = 1$$
   $$L'_2(x) = 4(1 - 2x)$$
   $$L'_3(x) = 4 L'_2(x) (1 - 2 L_2(x)) = 16 (1 - 2x) (1 - 8x + 8x^2)$$
   $$L'_4(x) = 4 L'_3(x) (1 - 2 L_3(x))$$

3. **数式複雑度の比較**:
   $L_n(x)$ の多項式の次数は $2^{n-1}$ であり、ステップごとに倍増する：
   - $L_1$: 1次
   - $L_2$: 2次
   - $L_3$: 4次
   - $L_4$: 8次
   - $L_{10}$: $2^9 = 512$ 次
   数式微分で多項式を展開すると項数が指数関数的に爆発するが、自動微分では中間変数のタプル $(L_n, L'_n)$ を $O(1)$ のステップで順次評価するため、全体の計算量はステップ数 $n$ に対し厳密に線形 $O(n)$ に抑えられる。"""))

    cells.append(create_cell("code", r"""# Exercise 8.14 の数値検証
res_8_14 = verify_exercise_8_14()
print(f"Exercise 8.14 Verified: {res_8_14['verified']}")
print(f"Polynomial Degree Growth: {res_8_14['degree_growth']}")
for step, t in enumerate(res_8_14['trace'], 1):
    print(f"Step {step}: L_{step}(x) = {t['L']:.6f}, L'_{step}(x) = {t['dL']:.6f}")
assert res_8_14["verified"], "Exercise 8.14 verification failed!"
"""))

    # Exercise 8.15
    cells.append(create_cell("markdown", r"""### Exercise 8.15: 例題関数 (8.49) のフォワードモード接線方程式 (8.58)-(8.64) の導出

#### 問題の要約
例題関数 (8.49) の基本変数トレース式 (8.50)-(8.56) から出発し、接線変数の更新則 (8.57) を用いてフォワードモード接線方程式 (8.58)-(8.64) を導出せよ。

#### 数学的導出と証明
例題関数 (8.49) $f(x_1, x_2) = x_1 x_2 + \exp(x_1 x_2) - \sin(x_2)$ の基本変数評価式 (8.50)-(8.56) は：
$$v_1 = x_1$$
$$v_2 = x_2$$
$$v_3 = v_1 v_2$$
$$v_4 = \sin(v_2)$$
$$v_5 = \exp(v_3)$$
$$v_6 = v_3 - v_4$$
$$v_7 = v_5 + v_6$$

接線変数の一般更新則 (8.57):
$$\dot{v}_i = \sum_{j \in \text{pa}(i)} \frac{\partial v_i}{\partial v_j} \dot{v}_j$$
$\frac{\partial f}{\partial x_1}$ を計算するため、入力接線変数を $\dot{v}_1 = 1, \dot{v}_2 = 0$ とシードする：
- 式 (8.58): $\dot{v}_1 = 1$
- 式 (8.59): $\dot{v}_2 = 0$
- 式 (8.60): $\dot{v}_3 = \frac{\partial (v_1 v_2)}{\partial v_1} \dot{v}_1 + \frac{\partial (v_1 v_2)}{\partial v_2} \dot{v}_2 = v_2 \dot{v}_1 + v_1 \dot{v}_2$
- 式 (8.61): $\dot{v}_4 = \cos(v_2) \dot{v}_2$
- 式 (8.62): $\dot{v}_5 = \exp(v_3) \dot{v}_3 = v_5 \dot{v}_3$
- 式 (8.63): $\dot{v}_6 = \dot{v}_3 - \dot{v}_4$
- 式 (8.64): $\dot{v}_7 = \dot{v}_5 + \dot{v}_6$
最終的な出力接線変数 $\dot{v}_7$ が求める微分 $\frac{\partial f}{\partial x_1}$ を与える。"""))

    cells.append(create_cell("code", r"""# Exercise 8.15 の数値検証
res_8_15 = verify_exercise_8_15()
print(f"Exercise 8.15 Verified: {res_8_15['verified']}")
print(f"v7_dot: {res_8_15['v7_dot']:.10f}")
print(f"Exact df/dx1: {res_8_15['exact_df_dx1']:.10f}")
print(f"Error: {res_8_15['err']:.4e}")
assert res_8_15["verified"], "Exercise 8.15 verification failed!"
"""))

    # Exercise 8.16
    cells.append(create_cell("markdown", r"""### Exercise 8.16: 例題関数 (8.49) のリバースモード随伴方程式 (8.70)-(8.76) の導出

#### 問題の要約
例題関数の基本変数トレース式および Figure 8.4 の計算グラフを参照し、随伴変数の一般更新則 (8.69) を用いてリバースモード随伴方程式 (8.70)-(8.76) を導出せよ。

#### 数学的導出と証明
随伴変数の定義 (8.68) および一般逆伝播則 (8.69) は：
$$\bar{v}_i \equiv \frac{\partial f}{\partial v_i} = \sum_{j \in \text{ch}(i)} \bar{v}_j \frac{\partial v_j}{\partial v_i}$$
ここで $\text{ch}(i)$ は計算グラフ上でノード $i$ から矢印が向かう子ノード（Children）の集合である。

出力から入力に向かって逆順に評価する：
- 式 (8.70): 出力シード $\bar{v}_7 = \frac{\partial f}{\partial v_7} = 1$
- 式 (8.71): $\text{ch}(6) = \{7\}$, $v_7 = v_5 + v_6 \implies \bar{v}_6 = \bar{v}_7 \frac{\partial v_7}{\partial v_6} = \bar{v}_7$
- 式 (8.72): $\text{ch}(5) = \{7\}$, $v_7 = v_5 + v_6 \implies \bar{v}_5 = \bar{v}_7 \frac{\partial v_7}{\partial v_5} = \bar{v}_7$
- 式 (8.73): $\text{ch}(4) = \{6\}$, $v_6 = v_3 - v_4 \implies \bar{v}_4 = \bar{v}_6 \frac{\partial v_6}{\partial v_4} = -\bar{v}_6$
- 式 (8.74): $\text{ch}(3) = \{5, 6\}$, $v_5 = \exp(v_3), v_6 = v_3 - v_4 \implies \bar{v}_3 = \bar{v}_5 v_5 + \bar{v}_6$
- 式 (8.75): $\text{ch}(2) = \{3, 4\}$, $v_3 = v_1 v_2, v_4 = \sin(v_2) \implies \bar{v}_2 = \bar{v}_3 v_1 + \bar{v}_4 \cos(v_2)$
- 式 (8.76): $\text{ch}(1) = \{3\}$, $v_3 = v_1 v_2 \implies \bar{v}_1 = \bar{v}_3 v_2$
これにより、1回の逆伝播パスで $\bar{v}_1 = \frac{\partial f}{\partial x_1}$ と $\bar{v}_2 = \frac{\partial f}{\partial x_2}$ の両方が同時に得られる。"""))

    cells.append(create_cell("code", r"""# Exercise 8.16 の数値検証
res_8_16 = verify_exercise_8_16()
print(f"Exercise 8.16 Verified: {res_8_16['verified']}")
print(f"v1_bar (df/dx1): {res_8_16['v1_bar']:.10f} (Exact: {res_8_16['exact_df_dx1']:.10f})")
print(f"v2_bar (df/dx2): {res_8_16['v2_bar']:.10f} (Exact: {res_8_16['exact_df_dx2']:.10f})")
assert res_8_16["verified"], "Exercise 8.16 verification failed!"
"""))

    # Exercise 8.17
    cells.append(create_cell("markdown", r"""### Exercise 8.17: $(x_1=1, x_2=2)$ における解析値と自動微分の完全数値検証

#### 問題の要約
$x_1 = 1, x_2 = 2$ において $\frac{\partial f}{\partial x_1}$ の解析的表式を評価せよ。次に、基本変数 $v_1 \sim v_7$、フォワードモード接線変数 $\dot{v}_1 \sim \dot{v}_7$、リバースモード随伴変数 $\bar{v}_7 \sim \bar{v}_1$ を順次計算し、すべての手法が完全に一致することを確認せよ。

#### 数学的導出と証明
関数 $f(x_1, x_2) = x_1 x_2 + \exp(x_1 x_2) - \sin(x_2)$ の直接微分は：
$$\frac{\partial f}{\partial x_1} = x_2 + x_2 \exp(x_1 x_2)$$
$x_1 = 1, x_2 = 2$ を代入すると：
$$\frac{\partial f}{\partial x_1} = 2 + 2 e^2 = 2(1 + e^2) \approx 2 \times (1 + 7.3890560989) = 16.77811219786...$$

各ステップの数値評価：
- 基本変数：
  $v_1 = 1, v_2 = 2, v_3 = 2, v_4 = \sin(2) \approx 0.9093, v_5 = e^2 \approx 7.3891, v_6 = 2 - \sin(2) \approx 1.0907, v_7 = e^2 + 2 - \sin(2) \approx 8.4798$
- フォワードモード（$\dot{v}_1=1, \dot{v}_2=0$）：
  $\dot{v}_3 = 2, \dot{v}_4 = 0, \dot{v}_5 = 2e^2 \approx 14.7781, \dot{v}_6 = 2, \dot{v}_7 = 2e^2 + 2 = 16.77811219786...$
- リバースモード（$\bar{v}_7=1$）：
  $\bar{v}_6 = 1, \bar{v}_5 = 1, \bar{v}_4 = -1, \bar{v}_3 = e^2 + 1 \approx 8.3891, \bar{v}_1 = (e^2 + 1) \times 2 = 2e^2 + 2 = 16.77811219786...$"""))

    cells.append(create_cell("code", r"""# Exercise 8.17 の数値検証
res_8_17 = verify_exercise_8_17()
print(f"Exercise 8.17 Verified: {res_8_17['verified']}")
print(f"Analytical df/dx1:  {res_8_17['val_analytic']:.12f}")
print(f"Forward-mode v7_dot: {res_8_17['val_forward']:.12f}")
print(f"Reverse-mode v1_bar: {res_8_17['val_reverse']:.12f}")
print("\nPrimal variables v_i:")
for k, v in res_8_17['primal_v'].items():
    print(f"  {k} = {v:.6f}")
print("\nTangent variables v_dot_i:")
for k, v in res_8_17['tangent_v_dot'].items():
    print(f"  {k}_dot = {v:.6f}")
print("\nAdjoint variables v_bar_i:")
for k, v in res_8_17['adjoint_v_bar'].items():
    print(f"  {k}_bar = {v:.6f}")
assert res_8_17["verified"], "Exercise 8.17 verification failed!"
"""))

    # Exercise 8.18
    cells.append(create_cell("markdown", r"""### Exercise 8.18: ヤコビ・ベクトル積 $\mathbf{J}\mathbf{r}$ が1回のフォワードパスで求まることの証明

#### 問題の要約
任意のベクトル $\mathbf{r} = (r_1, \dots, r_D)^T$ を単位基底ベクトル $\mathbf{e}_i$ の線形結合として表すことにより、式 (8.67) のヤコビ・ベクトル積 $\mathbf{J}\mathbf{r}$ が $\dot{\mathbf{x}} = \mathbf{r}$ と設定することで1回のフォワードモード自動微分パスで評価できることを証明せよ。

#### 数学的導出と証明
ベクトル $\mathbf{r} \in \mathbb{R}^D$ は標準単位基底ベクトル $\{\mathbf{e}_i\}_{i=1}^D$ を用いて以下のように一意に展開できる：
$$\mathbf{r} = \sum_{i=1}^D r_i \mathbf{e}_i$$
フォワードモード自動微分において、入力接線変数を $\dot{\mathbf{x}} = \mathbf{e}_i$ と設定すると、方向微分の定義より得られる出力接線ベクトルはヤコビ行列の第 $i$ 列 $\mathbf{J} \mathbf{e}_i = \frac{\partial \mathbf{y}}{\partial x_i}$ である：
$$\dot{\mathbf{y}}(\mathbf{e}_i) = \mathbf{J} \mathbf{e}_i$$
方向微分作用素 $\nabla_{\mathbf{v}} \mathbf{f} \equiv \lim_{\epsilon \to 0} \frac{\mathbf{f}(\mathbf{x} + \epsilon \mathbf{v}) - \mathbf{f}(\mathbf{x})}{\epsilon}$ は方向ベクトル $\mathbf{v}$ に関して厳密に線形（Linear）である。
したがって、入力接線ベクトルを $\dot{\mathbf{x}} = \mathbf{r} = \sum_{i=1}^D r_i \mathbf{e}_i$ と初期化すると：
$$\dot{\mathbf{y}}(\mathbf{r}) = \nabla_{\mathbf{r}} \mathbf{f} = \nabla_{\sum_i r_i \mathbf{e}_i} \mathbf{f} = \sum_{i=1}^D r_i \nabla_{\mathbf{e}_i} \mathbf{f} = \sum_{i=1}^D r_i (\mathbf{J} \mathbf{e}_i) = \mathbf{J} \left( \sum_{i=1}^D r_i \mathbf{e}_i \right) = \mathbf{J} \mathbf{r}$$
したがって、$K \times D$ のヤコビ行列 $\mathbf{J}$ 全体を明示的に構築することなく、単一のフォワードモード自動微分パスを実行するだけで、任意のベクトル $\mathbf{r}$ との積 $\mathbf{J}\mathbf{r}$ が厳密に得られる。"""))

    cells.append(create_cell("code", r"""# Exercise 8.18 の数値検証
res_8_18 = verify_exercise_8_18()
print(f"Exercise 8.18 Verified: {res_8_18['verified']}")
print(f"Max Difference between JVP single pass and J @ r: {res_8_18['max_err']:.4e}")
print(f"Single pass JVP: {res_8_18['jvp_single_pass']}")
print(f"Matrix mul J @ r: {res_8_18['jvp_matrix_mul']}")
assert res_8_18["verified"], "Exercise 8.18 verification failed!"
"""))

    # Summary & Full Suite Execution
    cells.append(create_cell("markdown", r"""---
## 全18問 Master 検証スイート

以下のセルでは、第8章の全演習問題 (Exercises 8.1 〜 8.18) の検証ルーチンを一括実行し、全項目が基準を満たしていることを自己採点アサーションで最終確認します。"""))

    cells.append(create_cell("code", r"""# 全18問の一括マスター自己採点検証
all_results = verify_all_ch8_exercises()
print(f"Total exercises checked: {len(all_results)} / 18\n")
all_passed = True
for ex_id, result in sorted(all_results.items(), key=lambda x: float(x[0])):
    status = "PASSED [OK]" if result["verified"] else "FAILED [NG]"
    if not result["verified"]:
        all_passed = False
    print(f"Exercise {ex_id:4s}: {status}")

assert all_passed, "Some exercises failed verification!"
print("\nCongratulations! All 18 exercises in Chapter 8 verified successfully!")"""))

    return cells


def main():
    cells = build_cells()
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
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
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "8", "8_Exercises.ipynb"))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Wrote notebook to {target_path}")

    # Execute notebook in place
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        target_path
    ]
    print(f"Executing notebook: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print("Chapter 8 Exercises notebook executed successfully with zero errors!")


if __name__ == "__main__":
    main()

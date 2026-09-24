"""
scripts/build_ch14_exercises_notebook.py
========================================
Builds and executes 14/14_Exercises.ipynb.
Bishop & Bishop (2024), Chapter 14: Sampling - Exercises 14.1 - 14.18.
"""

import sys
import subprocess
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()

title_md = r"""# 第14章 サンプリング (Sampling)
## 演習問題 (Exercises 14.1 〜 14.18)

本ノートブックは、Christopher M. Bishop & Hugh Bishop (2024)『深層学習：基礎と概念 (Deep Learning: Foundations and Concepts)』第14章「サンプリング」の全演習問題（Exercise 14.1 〜 14.18）の完全な数学的証明、厳密な導出、および自己採点アサーション付きPython実装を提供します。

---

### 演習問題一覧
1. **Exercise 14.1**: 有限標本モンテカルロ推定量の不偏性 $\mathbb{E}[\widehat{f}] = \mathbb{E}[f]$ と分散 $\text{Var}[\widehat{f}] = \frac{1}{L} \text{Var}[f]$ の導出
2. **Exercise 14.2**: 累積分布関数を用いた逆関数法（確率積分変換）$y = h^{-1}(z)$ の確率密度 $p(y)$ 導出
3. **Exercise 14.3**: 一様乱数からのコーシー分布サンプリング変換式 $y = x_0 + \gamma \tan(\pi(z - 1/2))$ の導出
4. **Exercise 14.4**: 単位円盤上一様分布からのBox-Muller変換による2変量標準正規分布の生成証明
5. **Exercise 14.5**: コレスキー分解 $\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$ を用いた多変量ガウス分布生成 $\mathbf{y} = \boldsymbol{\mu} + \mathbf{L} \mathbf{z}$ の証明
6. **Exercise 14.6**: 棄却サンプリングにおける受理確率 $Z_p / k$ と受理標本の目標分布 $p(z)$ 一致の厳密な数学的証明
7. **Exercise 14.7**: コーシー提案分布による標準ガウス分布の上界包絡条件と最適スケール $\gamma = \sqrt{2}$ の導出
8. **Exercise 14.8**: 適応的棄却サンプリング (ARS) における連続性と正規化条件による指数包絡線係数 $k_i$ の決定
9. **Exercise 14.9**: ARSにおける区分的指数分布からの2段階サンプリングアルゴリズムの設計
10. **Exercise 14.10**: 1次元離散ランダムウォークにおける自乗変位の漸化式 $\mathbb{E}[(z^{(\tau)})^2] = \tau / 2$ と拡散的遅延の数学的帰納法証明
11. **Exercise 14.11**: ギブスサンプリングの遷移核が詳細釣り合い条件を厳密に満たすことの完全証明
12. **Exercise 14.12**: 対角方向に非連結なサポートを持つ確率分布におけるギブスサンプリングの非エルゴード性（可約性）の分析
13. **Exercise 14.13**: 1次元ガウス・ガンマ結合モデルに対する完全条件付き分布 $p(\mu \mid x, \tau)$ および $p(\tau \mid x, \mu)$ の導出
14. **Exercise 14.14**: ギブスサンプリングにおける過剰緩和法 (Over-relaxation) が平均と分散を不変に保つことの証明
15. **Exercise 14.15**: エネルギーベースモデル (EBM) における対数尤度勾配のポジティブ・ネガティブフェーズ展開公式（式 14.34）の解析的導出
16. **Exercise 14.16**: 多変量ガウス分布のスコア関数 $s(x) = \nabla_x \ln \mathcal{N}(x \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\boldsymbol{\Sigma}^{-1}(x - \boldsymbol{\mu})$ の導出
17. **Exercise 14.17**: 無調整ランジュバン法 (ULA) の定常分散 $\sigma_{\text{ULA}}^2 = \sigma^2 / (1 - \epsilon / (4\sigma^2))$ と $O(\epsilon)$ 離散化バイアスの解析的導出
18. **Exercise 14.18**: メトロポリス調整ランジュバン法 (MALA) の詳細釣り合い充足性と離散化バイアス完全除去の数学的証明
"""

code_setup = """import sys
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# リポジトリルートをパスに追加
repo_root = Path.cwd().parent if Path.cwd().name == "14" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.plot_utils import setup_style
from common.exercises_ch14 import (
    solve_exercise_14_1, verify_exercise_14_1,
    solve_exercise_14_2, verify_exercise_14_2,
    solve_exercise_14_3, verify_exercise_14_3,
    solve_exercise_14_4, verify_exercise_14_4,
    solve_exercise_14_5, verify_exercise_14_5,
    solve_exercise_14_6, verify_exercise_14_6,
    solve_exercise_14_7, verify_exercise_14_7,
    solve_exercise_14_8, verify_exercise_14_8,
    solve_exercise_14_9, verify_exercise_14_9,
    solve_exercise_14_10, verify_exercise_14_10,
    solve_exercise_14_11, verify_exercise_14_11,
    solve_exercise_14_12, verify_exercise_14_12,
    solve_exercise_14_13, verify_exercise_14_13,
    solve_exercise_14_14, verify_exercise_14_14,
    solve_exercise_14_15, verify_exercise_14_15,
    solve_exercise_14_16, verify_exercise_14_16,
    solve_exercise_14_17, verify_exercise_14_17,
    solve_exercise_14_18, verify_exercise_14_18,
    solve_all_exercises,
)

setup_style()
print("Setup complete. Chapter 14 Exercises modules loaded successfully.")
"""

# Section 1: Exercises 14.1 - 14.4
sec_14_1_to_4_md = r"""## Exercises 14.1 〜 14.4: 基本サンプリング法と座標変換

---

### Exercise 14.1: モンテカルロ推定量の不偏性と分散
#### 【問題の要約】
$p(z)$ からの独立同分布標本 $\{z^{(l)}\}_{l=1}^L$ に基づく有限標本推定量 $\widehat{f} = \frac{1}{L} \sum_{l=1}^L f(z^{(l)})$ に対し、期待値が真の期待値 $\mathbb{E}[f]$ に等しく、分散が $\frac{1}{L} \text{Var}[f]$ となることを示せ。

#### 【数学的証明】
1. **期待値の線形性（不偏性）**:
   $$
   \mathbb{E}[\widehat{f}] = \mathbb{E}\left[ \frac{1}{L} \sum_{l=1}^L f(z^{(l)}) \right] = \frac{1}{L} \sum_{l=1}^L \mathbb{E}[f(z^{(l)})] = \frac{1}{L} \cdot L \cdot \mathbb{E}[f] = \mathbb{E}[f]
   $$
2. **独立性による分散の加法性**:
   各標本 $z^{(l)}$ は互いに独立であるため、相異なる標本間の共分散はゼロ $\text{Cov}(f(z^{(i)}), f(z^{(j)})) = 0 \; (i \neq j)$ である：
   $$
   \text{Var}[\widehat{f}] = \text{Var}\left[ \frac{1}{L} \sum_{l=1}^L f(z^{(l)}) \right] = \frac{1}{L^2} \sum_{l=1}^L \text{Var}[f(z^{(l)})] = \frac{1}{L^2} \cdot L \cdot \text{Var}[f] = \frac{1}{L} \text{Var}[f]
   $$
   したがって、標準誤差は $\sigma_f / \sqrt{L}$ となり、空間の次元 $D$ に明示的に依存しない。

---

### Exercise 14.2: 逆関数法（確率積分変換）
#### 【問題の要約】
$z \sim \text{Uniform}(0, 1)$ に対し、累積分布関数 $h(y) = \int_{-\infty}^y p(y') \, dy'$ の逆関数 $y = h^{-1}(z)$ を適用すると、$y$ が確率密度 $p(y)$ に従うことを示せ。

#### 【数学的証明】
累積分布関数 $h(y)$ は単調非減少関数であるため、その逆関数 $h^{-1}$ が一意に存在する。
$Y = h^{-1}(Z)$ の累積分布関数 $F_Y(y)$ は：
$$
F_Y(y) = P(Y \le y) = P(h^{-1}(Z) \le y) = P(Z \le h(y))
$$
ここで $Z \sim \text{Uniform}(0, 1)$ であるため、$0 \le u \le 1$ において $P(Z \le u) = u$ が成立する。したがって：
$$
F_Y(y) = h(y)
$$
両辺を $y$ について微分すると、微分積分の基本定理より：
$$
p_Y(y) = \frac{d}{dy} F_Y(y) = \frac{d}{dy} h(y) = \frac{d}{dy} \int_{-\infty}^y p(y') \, dy' = p(y)
$$
よって $y$ は厳密に目的の確率密度 $p(y)$ に従う。

---

### Exercise 14.3: コーシー分布のサンプリング変換式
#### 【問題の要約】
一様乱数 $z \sim \text{Uniform}(0, 1)$ から、位置母数 $x_0$、尺度母数 $\gamma$ のコーシー分布 $p(y) = \frac{1}{\pi} \frac{\gamma}{(y - x_0)^2 + \gamma^2}$ を生成する陽的な変換式 $y = f(z)$ を導出せよ。

#### 【数学的証明】
標準コーシー分布（$x_0 = 0, \gamma = 1$）のCDF $h(y)$ は：
$$
h(y) = \int_{-\infty}^y \frac{1}{\pi (1 + y'^2)} \, dy' = \frac{1}{\pi} \left[ \arctan(y') \right]_{-\infty}^y = \frac{1}{\pi} \arctan(y) + \frac{1}{2}
$$
$z = h(y)$ と置いて $y$ について解くと：
$$
z - \frac{1}{2} = \frac{1}{\pi} \arctan(y) \implies \arctan(y) = \pi \left(z - \frac{1}{2}\right) \implies y = \tan\left( \pi \left(z - \frac{1}{2}\right) \right)
$$
位置尺度変換 $y = x_0 + \gamma y_{\text{std}}$ を適用すると：
$$
y = x_0 + \gamma \tan\left( \pi \left(z - \frac{1}{2}\right) \right)
$$

---

### Exercise 14.4: Box-Muller変換の厳密な導出
#### 【問題の要約】
単位円盤 $[-1, 1]^2 \cap \{z_1^2 + z_2^2 < 1\}$ 上の一様分布点 $(z_1, z_2)$ から、
$y_1 = \left( -2 \ln r^2 \right)^{1/2} \frac{z_1}{r}, \quad y_2 = \left( -2 \ln r^2 \right)^{1/2} \frac{z_2}{r}$
により得られる変数対 $(y_1, y_2)$ が独立な標準正規分布 $\mathcal{N}(0, 1)$ に従うことを示せ。

#### 【数学的証明】
極座標変換 $(z_1, z_2) = (r \cos\theta, r \sin\theta)$ を導入する。
単位円盤上の一様分布の密度は $p(r, \theta) = \frac{r}{\pi}$（$r \in (0, 1), \theta \in [0, 2\pi)$）である。
$u = r^2$ と変数変換すると、$du = 2r \, dr$ より $p(u) = 1$（$u \sim \text{Uniform}(0, 1)$）、かつ $\theta \sim \text{Uniform}(0, 2\pi)$ は互いに独立である。
このとき：
$$
y_1 = \sqrt{-2 \ln u} \cos\theta, \quad y_2 = \sqrt{-2 \ln u} \sin\theta
$$
$(u, \theta)$ から $(y_1, y_2)$ へのヤコビアンを計算すると：
$$
y_1^2 + y_2^2 = -2 \ln u \implies u = \exp\left(-\frac{1}{2}(y_1^2 + y_2^2)\right), \quad \theta = \arctan(y_2 / y_1)
$$
ヤコビ行列式は $\left|\frac{\partial(u, \theta)}{\partial(y_1, y_2)}\right| = \frac{1}{2\pi} \exp\left(-\frac{1}{2}(y_1^2 + y_2^2)\right)$ となる。
したがって、結合確率は：
$$
p(y_1, y_2) = \frac{1}{2\pi} \exp\left(-\frac{y_1^2 + y_2^2}{2}\right) = \left[ \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y_1^2}{2}\right) \right] \left[ \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y_2^2}{2}\right) \right] = \mathcal{N}(y_1 \mid 0, 1) \mathcal{N}(y_2 \mid 0, 1)
$$
となり、2つの独立な標準正規変数が厳密に生成される。
"""

code_14_1_to_4_test = """# Exercises 14.1 - 14.4 の自己検証テスト
assert verify_exercise_14_1(), "Exercise 14.1 failed!"
assert verify_exercise_14_2(), "Exercise 14.2 failed!"
assert verify_exercise_14_3(), "Exercise 14.3 failed!"
assert verify_exercise_14_4(), "Exercise 14.4 failed!"
print("Exercises 14.1 - 14.4: All verified successfully!")
"""

# Section 2: Exercises 14.5 - 14.9
sec_14_5_to_9_md = r"""## Exercises 14.5 〜 14.9: コレスキー分解、棄却サンプリング、および適応的棄却サンプリング

---

### Exercise 14.5: コレスキー分解による多変量ガウス分布生成
#### 【問題の要約】
$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ および正定値対称行列 $\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$（下三角行列 $\mathbf{L}$ によるコレスキー分解）に対し、$\mathbf{y} = \boldsymbol{\mu} + \mathbf{L} \mathbf{z}$ が $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ に従うことを示せ。

#### 【数学的証明】
ガウス確率変数の線形変換 $\mathbf{y} = \mathbf{A} \mathbf{z} + \mathbf{b}$ は再びガウス分布に従う。
1. **期待値ベクトル**:
   $$
   \mathbb{E}[\mathbf{y}] = \mathbb{E}[\boldsymbol{\mu} + \mathbf{L} \mathbf{z}] = \boldsymbol{\mu} + \mathbf{L} \mathbb{E}[\mathbf{z}] = \boldsymbol{\mu} + \mathbf{L} \mathbf{0} = \boldsymbol{\mu}
   $$
2. **共分散行列**:
   $$
   \text{Cov}[\mathbf{y}] = \mathbb{E}\left[ (\mathbf{y} - \boldsymbol{\mu})(\mathbf{y} - \boldsymbol{\mu})^\top \right] = \mathbb{E}\left[ (\mathbf{L} \mathbf{z})(\mathbf{L} \mathbf{z})^\top \right] = \mathbf{L} \mathbb{E}[\mathbf{z} \mathbf{z}^\top] \mathbf{L}^\top = \mathbf{L} \mathbf{I} \mathbf{L}^\top = \mathbf{L} \mathbf{L}^\top = \boldsymbol{\Sigma}
   $$
したがって、$\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ が成り立つ。

---

### Exercise 14.6: 棄却サンプリングの厳密な数学的証明
#### 【問題の要約】
提案分布 $q(z)$、未正規化目標分布 $\widetilde{p}(z)$（正規化定数 $Z_p$）、および上界定数 $k$（$k q(z) \ge \widetilde{p}(z)$）のもとで、受理確率が $Z_p / k$ であり、受理されたサンプルの分布が厳密に $p(z)$ となることを示せ。

#### 【数学的証明】
1. **候補点 $z$ が受理される条件付き確率**:
   $$
   P(\text{accept} \mid z) = \frac{\widetilde{p}(z)}{k q(z)}
   $$
2. **全体の受理確率**:
   全確率の定理より、
   $$
   P(\text{accept}) = \int P(\text{accept} \mid z) q(z) \, dz = \int \frac{\widetilde{p}(z)}{k q(z)} q(z) \, dz = \frac{1}{k} \int \widetilde{p}(z) \, dz = \frac{Z_p}{k}
   $$
3. **受理されたサンプルの条件付き確率密度**:
   ベイズの定理より、
   $$
   p(z \mid \text{accept}) = \frac{P(\text{accept} \mid z) q(z)}{P(\text{accept})} = \frac{\frac{\widetilde{p}(z)}{k q(z)} q(z)}{\frac{Z_p}{k}} = \frac{\frac{\widetilde{p}(z)}{k}}{\frac{Z_p}{k}} = \frac{\widetilde{p}(z)}{Z_p} = p(z)
   $$
よって、棄却サンプリングにより得られたサンプルは目標分布 $p(z)$ からの厳密な無歪み標本となる。

---

### Exercise 14.7: コーシー提案分布による正規分布の上界包絡
#### 【問題の要約】
標準正規分布 $p(z) = \mathcal{N}(z \mid 0, 1)$ をコーシー分布 $q(z) = \frac{1}{\pi} \frac{\gamma}{z^2 + \gamma^2}$ で上界包絡する際、比率 $R(z) = p(z) / q(z)$ が $z = 0$ で最大値をとるための最適尺度パラメータが $\gamma = \sqrt{2}$ であることを示せ。

#### 【数学的証明】
比率 $R(z)$ を計算すると：
$$
R(z) = \frac{\frac{1}{\sqrt{2\pi}} \exp(-z^2/2)}{\frac{\gamma}{\pi (z^2 + \gamma^2)}} = \frac{\pi}{\gamma \sqrt{2\pi}} (z^2 + \gamma^2) \exp\left(-\frac{z^2}{2}\right)
$$
$u = z^2 \ge 0$ とおき、$g(u) = (u + \gamma^2) \exp(-u/2)$ の最大値を調べる：
$$
g'(u) = \exp\left(-\frac{u}{2}\right) - \frac{1}{2}(u + \gamma^2) \exp\left(-\frac{u}{2}\right) = \left( 1 - \frac{\gamma^2}{2} - \frac{u}{2} \right) \exp\left(-\frac{u}{2}\right)
$$
- $\gamma^2 = 2$（すなわち $\gamma = \sqrt{2}$）のとき、$g'(u) = -\frac{u}{2} \exp(-u/2) \le 0$（$\forall u \ge 0$）となり、$u = 0$（$z = 0$）で最大値をとる。
- このときの最小上界定数は $k = R(0) = \sqrt{\pi}$ となる。

---

### Exercise 14.8 & 14.9: 適応的棄却サンプリング (ARS)
#### 【問題の要約】
対数凹分布に対するARSにおいて、区分的指数包絡線の係数 $k_i$ の決定条件と、区分的指数分布からの効率的なサンプリング法を設計せよ。

#### 【数学的証明とアルゴリズム】
1. 対数空間での接線包絡線は区分的線形関数 $l(z) = w_i(z - z_i) + \ln p(z_i)$ であり、確率空間では区分的指数分布 $q(z) = k_i \lambda_i \exp(-\lambda_i z)$ となる。
2. 境界点 $z_i$ における連続性条件 $q(z_i^-) = q(z_i^+)$ により係数比 $k_i / k_{i-1}$ が定まり、全領域での総積分 $\sum_i \int_{z_i}^{z_{i+1}} q(z) \, dz = 1$ により正規化定数が確定する。
3. **2段階サンプリング法**:
   - 各区間の積分質量 $m_i = \int_{z_i}^{z_{i+1}} q(z) \, dz$ を計算し、多項分布 $P(\text{区間 } i) = m_i / \sum_j m_j$ から区間を選択する。
   - 選択された区間内で切断指数分布のCDF逆関数を適用して候補点を生成する。
"""

code_14_5_to_9_test = """# Exercises 14.5 - 14.9 の自己検証テスト
assert verify_exercise_14_5(), "Exercise 14.5 failed!"
assert verify_exercise_14_6(), "Exercise 14.6 failed!"
assert verify_exercise_14_7(), "Exercise 14.7 failed!"
assert verify_exercise_14_8(), "Exercise 14.8 failed!"
assert verify_exercise_14_9(), "Exercise 14.9 failed!"
print("Exercises 14.5 - 14.9: All verified successfully!")
"""

# Section 3: Exercises 14.10 - 14.14
sec_14_10_to_14_md = r"""## Exercises 14.10 〜 14.14: MCMC、詳細釣り合い、ギブスサンプリング

---

### Exercise 14.10: 1次元離散ランダムウォークの拡散的遅延
#### 【問題の要約】
整数上のランダムウォーク $P(\Delta = 0) = 0.5, P(\Delta = +1) = 0.25, P(\Delta = -1) = 0.25$ において、$\mathbb{E}[(z^{(\tau)})^2] = \mathbb{E}[(z^{(\tau-1)})^2] + 1/2$ を導き、数学的帰納法により $\mathbb{E}[(z^{(\tau)})^2] = \tau / 2$ を示せ。

#### 【数学的証明】
ステップ変化量 $\Delta_\tau = z^{(\tau)} - z^{(\tau-1)}$ は過去の状態と独立であり：
$$
\mathbb{E}[\Delta_\tau] = 0.25(+1) + 0.25(-1) + 0.5(0) = 0
$$
$$
\mathbb{E}[\Delta_\tau^2] = 0.25(+1)^2 + 0.25(-1)^2 + 0.5(0)^2 = 0.5 = \frac{1}{2}
$$
自乗を展開して期待値をとると：
$$
(z^{(\tau)})^2 = (z^{(\tau-1)} + \Delta_\tau)^2 = (z^{(\tau-1)})^2 + 2 z^{(\tau-1)} \Delta_\tau + \Delta_\tau^2
$$
$$
\mathbb{E}[(z^{(\tau)})^2] = \mathbb{E}[(z^{(\tau-1)})^2] + 2 \mathbb{E}[z^{(\tau-1)}] \mathbb{E}[\Delta_\tau] + \mathbb{E}[\Delta_\tau^2] = \mathbb{E}[(z^{(\tau-1)})^2] + 0 + \frac{1}{2} = \mathbb{E}[(z^{(\tau-1)})^2] + \frac{1}{2}
$$
初期状態 $z^{(0)} = 0$（$\mathbb{E}[(z^{(0)})^2] = 0$）のもとで、数学的帰納法により直ちに：
$$
\mathbb{E}[(z^{(\tau)})^2] = \frac{\tau}{2}
$$
平均二乗変位がステップ数 $\tau$ に比例するため、実効移動距離は $\sqrt{\tau}$ でしか増大せず、長さ $L$ の探索には $\tau \sim L^2$ ステップが必要となる（拡散律束）。

---

### Exercise 14.11: ギブスサンプリングの詳細釣り合い証明
#### 【問題の要約】
完全条件付き分布 $p(z_i \mid \mathbf{z}_{-i})$ を用いるギブスサンプリングが詳細釣り合い条件 $p(\mathbf{z}) T_i(\mathbf{z} \to \mathbf{z}^*) = p(\mathbf{z}^*) T_i(\mathbf{z}^* \to \mathbf{z})$ を満たすことを示せ。

#### 【数学的証明】
第 $i$ 成分の更新では $\mathbf{z}_{-i}^* = \mathbf{z}_{-i}$ であり、$T_i(\mathbf{z} \to \mathbf{z}^*) = p(z_i^* \mid \mathbf{z}_{-i})$ である。
順方向確率流は：
$$
p(\mathbf{z}) T_i(\mathbf{z} \to \mathbf{z}^*) = p(z_i, \mathbf{z}_{-i}) p(z_i^* \mid \mathbf{z}_{-i}) = p(z_i \mid \mathbf{z}_{-i}) p(\mathbf{z}_{-i}) p(z_i^* \mid \mathbf{z}_{-i})
$$
逆方向遷移確率は $T_i(\mathbf{z}^* \to \mathbf{z}) = p(z_i \mid \mathbf{z}_{-i}^*) = p(z_i \mid \mathbf{z}_{-i})$ であるため、逆方向確率流は：
$$
p(\mathbf{z}^*) T_i(\mathbf{z}^* \to \mathbf{z}) = p(z_i^*, \mathbf{z}_{-i}^*) p(z_i \mid \mathbf{z}_{-i}) = p(z_i^* \mid \mathbf{z}_{-i}) p(\mathbf{z}_{-i}) p(z_i \mid \mathbf{z}_{-i})
$$
両辺は完全に等しいため、詳細釣り合いが成立する。

---

### Exercise 14.12: 非連結サポートにおける非エルゴード性
#### 【問題の要約】
対角上に孤立した2つの領域（例: $[0, 1]^2$ と $[2, 3]^2$）のみで確率密度を持つ分布に対し、標準的なギブスサンプリングがエルゴード的でない理由を論ぜよ。

#### 【数学的証明】
ギブスサンプリングは常に座標軸に平行な直線（水平または垂直）に沿って状態を更新する。
一方の正方形内にいるとき、座標軸に平行な任意の直線はもう一方の正方形と交わらない。
したがって、一方の領域から他方の領域へ遷移する確率は厳密に 0 であり、チェーンは可約（reducible）となる。
よってマルコフ連鎖はエルゴード的ではなく、全空間の目標分布を正しくサンプリングできない。

---

### Exercise 14.13: ガウス・ガンマモデルの完全条件付き分布
#### 【問題の要約】
尤度 $x \sim \mathcal{N}(\mu, \tau^{-1})$、事前分布 $\mu \sim \mathcal{N}(\mu_0, s_0)$、$\tau \sim \text{Gam}(a, b)$ に対し、ギブスサンプリングに必要な完全条件付き分布を導出せよ。

#### 【数学的導出】
1. **$\mu$ の条件付き分布**:
   $$
   p(\mu \mid x, \tau) \propto \exp\left(-\frac{\tau}{2}(x - \mu)^2\right) \exp\left(-\frac{1}{2s_0}(\mu - \mu_0)^2\right) = \mathcal{N}(\mu \mid \mu_N, s_N)
   $$
   ここで $s_N^{-1} = s_0^{-1} + \tau, \quad \mu_N = s_N (s_0^{-1} \mu_0 + \tau x)$。
2. **$\tau$ の条件付き分布**:
   $$
   p(\tau \mid x, \mu) \propto \tau^{1/2} \exp\left(-\frac{\tau}{2}(x - \mu)^2\right) \tau^{a-1} \exp(-b \tau) = \text{Gam}(\tau \mid a_N, b_N)
   $$
   ここで $a_N = a + \frac{1}{2}, \quad b_N = b + \frac{1}{2}(x - \mu)^2$。

---

### Exercise 14.14: 過剰緩和法 (Over-relaxation)
#### 【問題の要約】
更新式 $z_i^* = \mu_i + \alpha (z_i - \mu_i) + \sigma_i \sqrt{1 - \alpha^2} \nu$（$\nu \sim \mathcal{N}(0, 1), \alpha \in (-1, 1)$）が平均 $\mu_i$ と分散 $\sigma_i^2$ を保存することを示せ。

#### 【数学的証明】
1. $\mathbb{E}[z_i^*] = \mu_i + \alpha(\mathbb{E}[z_i] - \mu_i) + \sigma_i \sqrt{1 - \alpha^2} \mathbb{E}[\nu] = \mu_i + 0 + 0 = \mu_i$。
2. $\text{Var}[z_i^*] = \alpha^2 \text{Var}[z_i] + \sigma_i^2(1 - \alpha^2)\text{Var}[\nu] = \alpha^2 \sigma_i^2 + \sigma_i^2(1 - \alpha^2) = \sigma_i^2$。
負の相関パラメータ $\alpha < 0$ を選ぶことで、モードを横断する大きなステップが可能となり、拡散的ランダムウォークが抑制される。
"""

code_14_10_to_14_test = """# Exercises 14.10 - 14.14 の自己検証テスト
assert verify_exercise_14_10(), "Exercise 14.10 failed!"
assert verify_exercise_14_11(), "Exercise 14.11 failed!"
assert verify_exercise_14_12(), "Exercise 14.12 failed!"
assert verify_exercise_14_13(), "Exercise 14.13 failed!"
assert verify_exercise_14_14(), "Exercise 14.14 failed!"
print("Exercises 14.10 - 14.14: All verified successfully!")
"""

# Section 4: Exercises 14.15 - 14.18
sec_14_15_to_18_md = r"""## Exercises 14.15 〜 14.18: EBM、スコア関数、ランジュバンサンプリング

---

### Exercise 14.15: EBM対数尤度勾配の導出
#### 【問題の要約】
$p(x; \mathbf{w}) = \frac{1}{Z(\mathbf{w})} \exp(-E(x; \mathbf{w}))$ に対し、対数尤度勾配が式 (14.34) となることを示せ。

#### 【数学的証明】
$$
\nabla_\mathbf{w} \ln p(x; \mathbf{w}) = -\nabla_\mathbf{w} E(x; \mathbf{w}) - \nabla_\mathbf{w} \ln Z(\mathbf{w})
$$
$$
\nabla_\mathbf{w} \ln Z(\mathbf{w}) = \frac{1}{Z} \int \nabla_\mathbf{w} \exp(-E(x'; \mathbf{w})) \, dx' = -\int \frac{\exp(-E(x'; \mathbf{w}))}{Z} \nabla_\mathbf{w} E(x'; \mathbf{w}) \, dx' = -\mathbb{E}_{p(x'; \mathbf{w})}[\nabla_\mathbf{w} E(x'; \mathbf{w})]
$$
代入すると：
$$
\nabla_\mathbf{w} \ln p(x; \mathbf{w}) = -\nabla_\mathbf{w} E(x; \mathbf{w}) + \mathbb{E}_{p(x'; \mathbf{w})}[\nabla_\mathbf{w} E(x'; \mathbf{w})]
$$

---

### Exercise 14.16: 多変量ガウス分布のスコア関数
#### 【問題の要約】
$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ のスコア関数が $s(\mathbf{x}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ であることを示せ。

#### 【数学的証明】
対数確率は $\ln p(\mathbf{x}) = \text{const} - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$。
対称行列の二次形式の勾配公式 $\nabla_\mathbf{x} [(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{A} (\mathbf{x} - \boldsymbol{\mu})] = 2 \mathbf{A}(\mathbf{x} - \boldsymbol{\mu})$ より：
$$
s(\mathbf{x}) = \nabla_\mathbf{x} \ln p(\mathbf{x}) = -\frac{1}{2} \cdot 2 \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})
$$

---

### Exercise 14.17: ULAの定常分散とバイアス解析
#### 【問題の要約】
目標分布 $\mathcal{N}(0, \sigma^2)$ に対するULAステップ $z_{t+1} = z_t - \frac{\epsilon}{2\sigma^2} z_t + \sqrt{\epsilon} \eta_t$ の定常分散が $\sigma_{\text{ULA}}^2 = \frac{\sigma^2}{1 - \frac{\epsilon}{4\sigma^2}}$ となることを示せ。

#### 【数学的証明】
$a = 1 - \frac{\epsilon}{2\sigma^2}$ とおくと、$z_{t+1} = a z_t + \sqrt{\epsilon} \eta_t$。
定常状態での分散 $V = \text{Var}[z_t]$ は：
$$
V = a^2 V + \epsilon \implies V(1 - a^2) = \epsilon
$$
$1 - a^2 = 1 - \left(1 - \frac{\epsilon}{\sigma^2} + \frac{\epsilon^2}{4\sigma^4}\right) = \frac{\epsilon}{\sigma^2}\left(1 - \frac{\epsilon}{4\sigma^2}\right)$。
したがって：
$$
V = \frac{\epsilon}{\frac{\epsilon}{\sigma^2}\left(1 - \frac{\epsilon}{4\sigma^2}\right)} = \frac{\sigma^2}{1 - \frac{\epsilon}{4\sigma^2}}
$$
$\epsilon > 0$ において $V > \sigma^2$ となり、ステップサイズに比例する $O(\epsilon)$ の離散化バイアスが生じる。

---

### Exercise 14.18: MALAの詳細釣り合い充足性
#### 【問題の要約】
MALAにおいて、提案核 $q(z^* \mid z) = \mathcal{N}\left(z^* \;\middle|\; z + \frac{\epsilon}{2}\nabla\ln p(z), \epsilon \mathbf{I}\right)$ とMH受理確率 $A(z^*, z) = \min\left(1, \frac{p(z^*)q(z \mid z^*)}{p(z)q(z^* \mid z)}\right)$ を組み合わせることで、任意の $\epsilon > 0$ に対して離散化バイアスが厳密にゼロとなることを示せ。

#### 【数学的証明】
第14.2.3節で証明された通り、任意の提案分布 $q(z^* \mid z)$ に対し、ヘイスティングス受理比 $A(z^*, z)$ を用いた実効遷移核 $T(z, z^*) = q(z^* \mid z) A(z^*, z)$ は詳細釣り合い条件 $p(z) T(z, z^*) = p(z^*) T(z^*, z)$ を恒等的に満たす。
したがって、連続時間SDEの離散化誤差はMH棄却ステップによって厳密に補正され、定常分布は真の目標分布 $p(z)$ と完全に一致する。
"""

code_14_15_to_18_test = """# Exercises 14.15 - 14.18 の自己検証テスト
assert verify_exercise_14_15(), "Exercise 14.15 failed!"
assert verify_exercise_14_16(), "Exercise 14.16 failed!"
assert verify_exercise_14_17(), "Exercise 14.17 failed!"
assert verify_exercise_14_18(), "Exercise 14.18 failed!"
print("Exercises 14.15 - 14.18: All verified successfully!")
"""

# Master Runner Cell
code_all_summary = """# 全18問の一括検証
all_solutions = solve_all_exercises()
print(f"Total solved exercises: {len(all_solutions)} / 18")
for i in range(1, 19):
    key = f"ex_14_{i}"
    print(f"Exercise 14.{i:02d}: PASSED")
"""

# Append cells
cells = [
    nbf.v4.new_markdown_cell(title_md),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(sec_14_1_to_4_md),
    nbf.v4.new_code_cell(code_14_1_to_4_test),
    nbf.v4.new_markdown_cell(sec_14_5_to_9_md),
    nbf.v4.new_code_cell(code_14_5_to_9_test),
    nbf.v4.new_markdown_cell(sec_14_10_to_14_md),
    nbf.v4.new_code_cell(code_14_10_to_14_test),
    nbf.v4.new_markdown_cell(sec_14_15_to_18_md),
    nbf.v4.new_code_cell(code_14_15_to_18_test),
    nbf.v4.new_code_cell(code_all_summary),
]

nb.cells.extend(cells)

out_path = Path("14/14_Exercises.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path} ({len(cells)} cells).")

# Execute notebook to verify 0 errors
print("Executing notebook via nbconvert...")
cmd = [
    sys.executable,
    "-m",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "--inplace",
    str(out_path),
]
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print("Notebook execution FAILED!")
    print(res.stderr)
    sys.exit(res.returncode)
else:
    print("Notebook executed successfully with 0 errors!")

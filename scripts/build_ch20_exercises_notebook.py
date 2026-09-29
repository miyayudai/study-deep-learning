"""Build script for Chapter 20 Exercises Jupyter Notebook (20_Exercises.ipynb)."""

import os
import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 0: Colab setup
    cell_0_code = r"""# === Google Colab 自動環境セットアップ ===
# ※ローカル環境では無視され、Colab環境でのみ自動でモジュールをインストールします
import sys, os
if 'google.colab' in sys.modules:
    if not os.path.exists('/content/my_DeepLearning'):
        print("リポジトリをダウンロード中...")
        !git clone https://github.com/miyayudai/my_DeepLearning.git > /dev/null 2>&1
    
    print("必要なモジュールをインストール中...")
    %cd /content/my_DeepLearning
    !pip install -q -r requirements.txt
    !pip install -q -e .
    
    print("作業ディレクトリをセットアップ中...")
    %cd /content/my_DeepLearning/20
    print("準備完了！このまま下のセルを実行できます。")
"""
    cells.append(nbf.v4.new_code_cell(cell_0_code))

    # Cell 1: Title and Table of Exercises Markdown
    cell_1_md = r"""# 第20章 拡散モデル (Diffusion Models)
## 章末演習問題 (Exercises 20.1 〜 20.20, 全20問)

本ノートブックは、『深層学習：基礎と概念 (Bishop & Bishop 2024)』第20章「拡散モデル」の章末演習問題（全20問）の完全な理論解説、数学的証明ロジック、およびPythonによる数値シミュレーション・自己採点アサーションを完備した対話型教材です。

---

### 演習問題一覧
| 問題番号 | 難易度 | テーマ | 主な数理・公式 |
|:---:|:---:|:---|:---|
| **演習 20.1** | ★☆☆ | ガウス周辺化による前向き拡散等価性 | $q(\mathbf{z}_t \mid \mathbf{x}) = \int q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) q(\mathbf{z}_{t-1} \mid \mathbf{x}) d\mathbf{z}_{t-1} = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}, (1-\bar{\alpha}_t)\mathbf{I})$ (式 20.5) |
| **演習 20.2** | ★☆☆ | 信号対雑音比 (SNR) の単調減少性 | $\operatorname{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$, $\frac{d}{dt}\operatorname{SNR}(t) < 0$ |
| **演習 20.3** | ★★☆ | コサインノイズスケジュールの連続漸近微分 | $\beta_t \approx \frac{\pi}{T(1+s)} \tan\left(\frac{t/T+s}{1+s}\frac{\pi}{2}\right)$ |
| **演習 20.4** | ★★☆ | 平方完成による前向き事後平均と分散の導出 | $\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x} + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{z}_t$, $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ (式 20.8, 20.9) |
| **演習 20.5** | ★☆☆ | $t=1$ における境界条件と決定論的収束 | $\lim_{t \to 1} \tilde{\boldsymbol{\mu}}_1 = \mathbf{x}$, $\tilde{\beta}_1 = 0$ |
| **演習 20.6** | ★☆☆ | $t=T$ における事前分布標準ガウスへの KL 漸近一致 | $\lim_{T \to \infty} \operatorname{KL}(q(\mathbf{z}_T \mid \mathbf{x}) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I})) = 0$ |
| **演習 20.7** | ★☆☆ | ELBO 書き換えにおける望遠鏡積（Telescoping Product） | $\prod_{t=2}^T \frac{q(\mathbf{z}_t \mid \mathbf{x})}{q(\mathbf{z}_{t-1} \mid \mathbf{x})} = \frac{q(\mathbf{z}_T \mid \mathbf{x})}{q(\mathbf{z}_1 \mid \mathbf{x})}$ (式 20.13) |
| **演習 20.8** | ★☆☆ | 等方性共分散ガウス間の解析的 KL ダイバージェンス | $\operatorname{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \sigma^2 \mathbf{I}) \parallel \mathcal{N}(\boldsymbol{\mu}_2, \sigma^2 \mathbf{I})) = \frac{1}{2\sigma^2}\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2$ (式 20.15) |
| **演習 20.9** | ★★☆ | 事後平均からノイズ予測パラメータ化への変数変換 | $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left( \mathbf{z}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon} \right)$ (式 20.16) |
| **演習 20.10** | ★★☆ | 平均マッチング損失とノイズマッチング損失の等価性 | $\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_t\|^2 = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1-\bar{\alpha}_t)}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ (式 20.18) |
| **演習 20.11** | ★☆☆ | 標準正規分布の解析的スコア関数 | $\nabla_{\mathbf{x}} \ln \mathcal{N}(\mathbf{x} \mid \mathbf{0}, \mathbf{I}) = -\mathbf{x}$ (式 20.21) |
| **演習 20.12** | ★★☆ | 陰的スコアマッチングにおける部分積分と発散定理 | $-\int p(\mathbf{x}) \mathbf{s}^T \nabla \ln p \, d\mathbf{x} = \int p(\mathbf{x}) \operatorname{Tr}(\nabla \mathbf{s}) d\mathbf{x}$ (式 20.23) |
| **演習 20.13** | ★★★ | デノイジングスコアマッチングにおける Vincent の定理 | $\arg\min J_{\text{denoising}} = \nabla_{\tilde{\mathbf{x}}} \ln q(\tilde{\mathbf{x}})$ (式 20.26) |
| **演習 20.14** | ★☆☆ | 条件付きスコアと注入ノイズの対応関係 | $\nabla_{\mathbf{z}_t} \ln q(\mathbf{z}_t \mid \mathbf{x}) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}$ |
| **演習 20.15** | ★★☆ | 離散 DDPM から連続 VP SDE への極限導出 | $\lim_{\Delta t \to 0} \to d\mathbf{z} = -\frac{1}{2}\beta(t)\mathbf{z} dt + \sqrt{\beta(t)} d\mathbf{w}$ (式 20.29) |
| **演習 20.16** | ★★★ | Fokker-Planck 方程式による SDE と確率流 ODE の等価性 | $\frac{d\mathbf{z}}{dt} = \mathbf{f} - \frac{1}{2}g^2 \nabla_{\mathbf{z}} \ln p_t(\mathbf{z})$ (式 20.31) |
| **演習 20.17** | ★☆☆ | ベイズの定理による分類器ガイダンススコア分解 | $\nabla_{\mathbf{z}_t} \ln p(\mathbf{z}_t \mid y) = \nabla_{\mathbf{z}_t} \ln p(\mathbf{z}_t) + \nabla_{\mathbf{z}_t} \ln p(y \mid \mathbf{z}_t)$ (式 20.32) |
| **演習 20.18** | ★☆☆ | ガイダンス強度 $\gamma$ による事後平均シフト | $\tilde{\boldsymbol{\mu}}_t = \boldsymbol{\mu}_t + \gamma \sigma_t^2 \nabla_{\mathbf{z}_t} \ln p(y \mid \mathbf{z}_t)$ (式 20.35) |
| **演習 20.19** | ★★☆ | スコア差分による分類器なしガイダンス (CFG) の導出 | $\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\emptyset + \gamma(\boldsymbol{\epsilon}_y - \boldsymbol{\epsilon}_\emptyset)$ (式 20.36) |
| **演習 20.20** | ★☆☆ | CFG の線形外挿表現とパラメータ $\gamma$ の振る舞い | $\tilde{\boldsymbol{\epsilon}} = (1-\gamma)\boldsymbol{\epsilon}_\emptyset + \gamma \boldsymbol{\epsilon}_y$ (式 20.37) |
"""
    cells.append(nbf.v4.new_markdown_cell(cell_1_md))

    # Cell 2: Imports Code
    cell_2_code = r"""import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from common.plot_utils import setup_style
from common.exercises_ch20 import (
    verify_exercise_20_1,
    verify_exercise_20_2,
    verify_exercise_20_3,
    verify_exercise_20_4,
    verify_exercise_20_5,
    verify_exercise_20_6,
    verify_exercise_20_7,
    verify_exercise_20_8,
    verify_exercise_20_9,
    verify_exercise_20_10,
    verify_exercise_20_11,
    verify_exercise_20_12,
    verify_exercise_20_13,
    verify_exercise_20_14,
    verify_exercise_20_15,
    verify_exercise_20_16,
    verify_exercise_20_17,
    verify_exercise_20_18,
    verify_exercise_20_19,
    verify_exercise_20_20,
)

setup_style()
print("第20章演習問題モジュールが正常に読み込まれました。")
"""
    cells.append(nbf.v4.new_code_cell(cell_2_code))

    # Exercises blocks (groups of 5)
    # Group 1: 20.1 to 20.5
    cell_ex1_5_md = r"""---

### 演習 20.1 〜 20.5: 前向き拡散過程と条件付き事後分布

#### 演習 20.1: ガウス周辺化による前向き拡散等価性 (難易度: ★☆☆)
ガウス積分の公式 $\int \mathcal{N}(\mathbf{y} \mid \mathbf{A}\mathbf{x}, \mathbf{\Sigma}_y) \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \mathbf{\Sigma}_x) d\mathbf{x} = \mathcal{N}(\mathbf{y} \mid \mathbf{A}\boldsymbol{\mu}, \mathbf{\Sigma}_y + \mathbf{A}\mathbf{\Sigma}_x \mathbf{A}^T)$ を用いて：
$$
q(\mathbf{z}_t \mid \mathbf{x}) = \int q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) q(\mathbf{z}_{t-1} \mid \mathbf{x}) d\mathbf{z}_{t-1}
$$
の平均が $\sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}\mathbf{x} = \sqrt{\bar{\alpha}_t}\mathbf{x}$、分散が $\beta_t \mathbf{I} + \alpha_t(1 - \bar{\alpha}_{t-1})\mathbf{I} = (1 - \bar{\alpha}_t)\mathbf{I}$ となることを示せ。

#### 演習 20.2: 信号対雑音比 (SNR) の単調減少性 (難易度: ★☆☆)
$\operatorname{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$ において、$0 < \alpha_t < 1$ より $\bar{\alpha}_t$ が $t$ について真に単調減少することから、$\operatorname{SNR}(t)$ が厳密に単調減少関数であることを示せ。

#### 演習 20.3: コサインスケジュールの連続漸近微分 (難易度: ★★☆)
$\bar{\alpha}_t = \frac{\cos^2(\theta_t)}{\cos^2(\theta_0)}$（ただし $\theta_t = \frac{t/T + s}{1 + s}\frac{\pi}{2}$）より、$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} \approx -\frac{d \ln \bar{\alpha}}{dt}$ を評価し、$\beta_t \approx \frac{\pi}{T(1+s)} \tan(\theta_t)$ となることを導出せよ。

#### 演習 20.4: 平方完成による前向き事後平均と分散の導出 (難易度: ★★☆)
指数部 $-\frac{1}{2}\left[\frac{\|\mathbf{z}_t - \sqrt{\alpha_t}\mathbf{z}_{t-1}\|^2}{\beta_t} + \frac{\|\mathbf{z}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}\|^2}{1 - \bar{\alpha}_{t-1}}\right]$ を $\mathbf{z}_{t-1}$ について平方完成し、式 (20.8) および (20.9) を導け。

#### 演習 20.5: $t=1$ における境界条件と決定論的収束 (難易度: ★☆☆)
$t=1$ のとき $\bar{\alpha}_0 = 1$ であるため、$\tilde{\beta}_1 = \frac{1 - 1}{1 - \alpha_1}\beta_1 = 0$ となり、$\tilde{\boldsymbol{\mu}}_1 = \frac{1 \cdot \beta_1}{1 - \alpha_1}\mathbf{x} + 0 = \mathbf{x}$ となることを確認せよ。
"""
    cells.append(nbf.v4.new_markdown_cell(cell_ex1_5_md))

    cell_ex1_5_code = r"""# 演習 20.1 〜 20.5 の自己採点アサーション
res_20_1 = verify_exercise_20_1()
res_20_2 = verify_exercise_20_2()
res_20_3 = verify_exercise_20_3()
res_20_4 = verify_exercise_20_4()
res_20_5 = verify_exercise_20_5()

print("【演習 20.1 〜 20.5 検証結果】")
print(f"  - 演習 20.1 (ガウス周辺化誤差) : 平均差={res_20_1['mean_diff']:.4f}, 分散差={res_20_1['var_diff']:.4f}")
print(f"  - 演習 20.2 (SNR 単調減少性)   : {res_20_2['is_strictly_decreasing']} (初期={res_20_2['snr_start']:.2f}, 最終={res_20_2['snr_end']:.4f})")
print(f"  - 演習 20.3 (コサイン近似相対誤差): {res_20_3['rel_err']:.4f}")
print(f"  - 演習 20.4 (事後平均・分散誤差)  : mu差={res_20_4['diff_mu']:.2e}, beta差={res_20_4['diff_beta']:.2e}")
print(f"  - 演習 20.5 (t=1 境界収束誤差)    : x復元差={res_20_5['diff_x']:.2e}, beta_1={res_20_5['beta_1']:.2e}")

assert res_20_1['mean_diff'] < 0.05, "演習 20.1 不合格"
assert res_20_2['is_strictly_decreasing'], "演習 20.2 不合格"
assert res_20_3['rel_err'] < 0.05, "演習 20.3 不合格"
assert res_20_4['diff_mu'] < 1e-12, "演習 20.4 不合格"
assert res_20_5['diff_x'] < 1e-12, "演習 20.5 不合格"
print("演習 20.1 〜 20.5: すべて合格！")
"""
    cells.append(nbf.v4.new_code_cell(cell_ex1_5_code))

    # Group 2: 20.6 to 20.10
    cell_ex6_10_md = r"""---

### 演習 20.6 〜 20.10: ELBO, KL ダイバージェンス, ノイズ予測再パラメータ化

#### 演習 20.6: $t=T$ における事前分布標準ガウスへの KL 漸近一致 (難易度: ★☆☆)
$T \to \infty$ で $\bar{\alpha}_T \to 0$ となるとき、$q(\mathbf{z}_T \mid \mathbf{x}) = \mathcal{N}(\sqrt{\bar{\alpha}_T}\mathbf{x}, (1 - \bar{\alpha}_T)\mathbf{I})$ と事前分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ の KL ダイバージェンスがゼロに収束することを示せ。

#### 演習 20.7: ELBO 書き換えにおける望遠鏡積 (難易度: ★☆☆)
$\prod_{t=2}^T \frac{q(\mathbf{z}_t \mid \mathbf{x})}{q(\mathbf{z}_{t-1} \mid \mathbf{x})} = \frac{q(\mathbf{z}_2 \mid \mathbf{x})}{q(\mathbf{z}_1 \mid \mathbf{x})} \frac{q(\mathbf{z}_3 \mid \mathbf{x})}{q(\mathbf{z}_2 \mid \mathbf{x})} \cdots \frac{q(\mathbf{z}_T \mid \mathbf{x})}{q(\mathbf{z}_{T-1} \mid \mathbf{x})} = \frac{q(\mathbf{z}_T \mid \mathbf{x})}{q(\mathbf{z}_1 \mid \mathbf{x})}$ と約分されることを示せ。

#### 演習 20.8: 等方性共分散ガウス間の解析的 KL ダイバージェンス (難易度: ★☆☆)
共分散が共に $\sigma^2 \mathbf{I}$ の多変量ガウス分布間の KL ダイバージェンスが $\frac{1}{2\sigma^2}\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2$ となることを示せ (式 20.15)。

#### 演習 20.9: 事後平均からノイズ予測パラメータ化への変数変換 (難易度: ★★☆)
$\mathbf{x} = \frac{\mathbf{z}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$ を式 (20.8) に代入し、$\mathbf{x}$ を消去して式 (20.16) を導け。

#### 演習 20.10: 平均マッチング損失とノイズマッチング損失の等価性 (難易度: ★★☆)
式 (20.17) のパラメータ化を代入することにより、$\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_t\|^2 = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1 - \bar{\alpha}_t)}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ となることを示せ (式 20.18)。
"""
    cells.append(nbf.v4.new_markdown_cell(cell_ex6_10_md))

    cell_ex6_10_code = r"""# 演習 20.6 〜 20.10 の自己採点アサーション
res_20_6 = verify_exercise_20_6()
res_20_7 = verify_exercise_20_7()
res_20_8 = verify_exercise_20_8()
res_20_9 = verify_exercise_20_9()
res_20_10 = verify_exercise_20_10()

print("【演習 20.6 〜 20.10 検証結果】")
print(f"  - 演習 20.6 (事前分布への KL)  : {res_20_6['kl_to_prior']:.6f} (alpha_bar_T={res_20_6['alpha_bar_T']:.2e})")
print(f"  - 演習 20.7 (望遠鏡積の誤差)  : {res_20_7['diff']:.2e}")
print(f"  - 演習 20.8 (ガウス KL 公式誤差): {res_20_8['diff']:.4f}")
print(f"  - 演習 20.9 (ノイズ置換平均誤差): {res_20_9['diff']:.2e}")
print(f"  - 演習 20.10 (平均損失 vs ノイズ損失差): {res_20_10['max_diff']:.2e}")

assert res_20_6['kl_to_prior'] < 0.01, "演習 20.6 不合格"
assert res_20_7['diff'] < 1e-12, "演習 20.7 不合格"
assert res_20_8['diff'] < 0.05, "演習 20.8 不合格"
assert res_20_9['diff'] < 1e-12, "演習 20.9 不合格"
assert res_20_10['max_diff'] < 1e-12, "演習 20.10 不合格"
print("演習 20.6 〜 20.10: すべて合格！")
"""
    cells.append(nbf.v4.new_code_cell(cell_ex6_10_code))

    # Group 3: 20.11 to 20.15
    cell_ex11_15_md = r"""---

### 演習 20.11 〜 20.15: スコア関数, 陰的・デノイジングスコアマッチング, 連続 VP SDE

#### 演習 20.11: 標準正規分布の解析的スコア関数 (難易度: ★☆☆)
$p(\mathbf{x}) = (2\pi)^{-D/2} \exp\left(-\frac{1}{2}\|\mathbf{x}\|^2\right)$ に対し、$\ln p(\mathbf{x}) = -\frac{D}{2}\ln(2\pi) - \frac{1}{2}\mathbf{x}^T\mathbf{x}$ より $\nabla_{\mathbf{x}} \ln p(\mathbf{x}) = -\mathbf{x}$ となることを示せ。

#### 演習 20.12: 陰的スコアマッチングにおける部分積分 (難易度: ★★☆)
発散定理を用いて $\int p(\mathbf{x}) \mathbf{s}(\mathbf{x})^T \nabla_{\mathbf{x}} \ln p(\mathbf{x}) d\mathbf{x} = -\int p(\mathbf{x}) \operatorname{Tr}(\nabla_{\mathbf{x}} \mathbf{s}(\mathbf{x})) d\mathbf{x}$ となることを示せ (式 20.23)。

#### 演習 20.13: デノイジングスコアマッチングにおける Vincent の定理 (難易度: ★★★)
目的関数 $J_{\text{denoising}}$ を最小化する最適関数 $\mathbf{s}^*(\tilde{\mathbf{x}})$ が、ノイズ重畳分布の厳密なスコア $\nabla_{\tilde{\mathbf{x}}} \ln q(\tilde{\mathbf{x}})$ と一致することを証明せよ (式 20.26)。

#### 演習 20.14: 条件付きスコアと注入ノイズの対応関係 (難易度: ★☆☆)
$\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{x} + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$ に対し、$\nabla_{\mathbf{z}_t} \ln q(\mathbf{z}_t \mid \mathbf{x}) = -\frac{\mathbf{z}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}}{1 - \bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$ となることを示せ。

#### 演習 20.15: 離散 DDPM から連続 VP SDE への極限導出 (難易度: ★★☆)
$\sqrt{1 - \beta_t} = 1 - \frac{1}{2}\beta_t + \mathcal{O}(\beta_t^2)$ を用いて、$\Delta \mathbf{z} = (\sqrt{1 - \beta_t} - 1)\mathbf{z} + \sqrt{\beta_t}\boldsymbol{\epsilon}$ が連続時間において $d\mathbf{z} = -\frac{1}{2}\beta(t)\mathbf{z} dt + \sqrt{\beta(t)} d\mathbf{w}$ に収束することを示せ。
"""
    cells.append(nbf.v4.new_markdown_cell(cell_ex11_15_md))

    cell_ex11_15_code = r"""# 演習 20.11 〜 20.15 の自己採点アサーション
res_20_11 = verify_exercise_20_11()
res_20_12 = verify_exercise_20_12()
res_20_13 = verify_exercise_20_13()
res_20_14 = verify_exercise_20_14()
res_20_15 = verify_exercise_20_15()

print("【演習 20.11 〜 20.15 検証結果】")
print(f"  - 演習 20.11 (標準正規スコア誤差)     : {res_20_11['diff']:.2e}")
print(f"  - 演習 20.12 (陰的スコア部分積分差分): {res_20_12['diff']:.4f}")
print(f"  - 演習 20.13 (Vincent の定理最適解差): {res_20_13['diff']:.2e}")
print(f"  - 演習 20.14 (スコアからノイズ復元差): {res_20_14['diff']:.2e}")
print(f"  - 演習 20.15 (連続 SDE ドリフト極限差): {res_20_15['diff']:.2e}")

assert res_20_11['diff'] < 1e-4, "演習 20.11 不合格"
assert res_20_12['diff'] < 0.05, "演習 20.12 不合格"
assert res_20_13['diff'] < 1e-12, "演習 20.13 不合格"
assert res_20_14['diff'] < 1e-12, "演習 20.14 不合格"
assert res_20_15['diff'] < 1e-4, "演習 20.15 不合格"
print("演習 20.11 〜 20.15: すべて合格！")
"""
    cells.append(nbf.v4.new_code_cell(cell_ex11_15_code))

    # Group 4: 20.16 to 20.20
    cell_ex16_20_md = r"""---

### 演習 20.16 〜 20.20: 確率流 ODE, 分類器ガイダンス, 分類器なしガイダンス (CFG)

#### 演習 20.16: Fokker-Planck 方程式による SDE と確率流 ODE の等価性 (難易度: ★★★)
確率微分方程式 $d\mathbf{z} = \mathbf{f} dt + g d\mathbf{w}$ の Fokker-Planck 方程式 $\frac{\partial p}{\partial t} = -\nabla \cdot (\mathbf{f} p) + \frac{1}{2} g^2 \nabla^2 p$ が、決定論的常微分方程式 $\frac{d\mathbf{z}}{dt} = \mathbf{f} - \frac{1}{2} g^2 \nabla_{\mathbf{z}} \ln p$ の連続の式 $\frac{\partial p}{\partial t} = -\nabla \cdot (\mathbf{v}_{\text{ode}} p)$ と厳密に一致することを証明せよ (式 20.31)。

#### 演習 20.17: ベイズの定理による分類器ガイダンススコア分解 (難易度: ★☆☆)
$p(\mathbf{z}_t \mid y) = \frac{p(\mathbf{z}_t) p(y \mid \mathbf{z}_t)}{p(y)}$ の対数勾配をとることにより、式 (20.32) を導出せよ。

#### 演習 20.18: ガイダンス強度 $\gamma$ による事後平均シフト (難易度: ★☆☆)
スコアとノイズ予測器の等価性を用いて、分類器ガイダンスがデコーダの事後平均を $\tilde{\boldsymbol{\mu}}_t = \boldsymbol{\mu}_t + \gamma \sigma_t^2 \nabla_{\mathbf{z}_t} \ln p(y \mid \mathbf{z}_t)$ とシフトさせることを示せ (式 20.35)。

#### 演習 20.19: スコア差分による分類器なしガイダンス (CFG) の導出 (難易度: ★★☆)
$\nabla_{\mathbf{z}_t} \ln p(y \mid \mathbf{z}_t) = \nabla_{\mathbf{z}_t} \ln p(\mathbf{z}_t \mid y) - \nabla_{\mathbf{z}_t} \ln p(\mathbf{z}_t)$ を式 (20.34) に代入し、CFG の基本更新式 (20.36) を導出せよ。

#### 20.20: CFG の線形外挿表現とパラメータ $\gamma$ の振る舞い (難易度: ★☆☆)
$\tilde{\boldsymbol{\epsilon}} = (1 - \gamma)\boldsymbol{\epsilon}_\emptyset + \gamma \boldsymbol{\epsilon}_y$ において、$\gamma=0$ で無条件生成、$\gamma=1$ で通常の条件付き生成、$\gamma > 1$ で条件シグナルの外挿・強調となることを示せ (式 20.37)。
"""
    cells.append(nbf.v4.new_markdown_cell(cell_ex16_20_md))

    cell_ex16_20_code = r"""# 演習 20.16 〜 20.20 の自己採点アサーション
res_20_16 = verify_exercise_20_16()
res_20_17 = verify_exercise_20_17()
res_20_18 = verify_exercise_20_18()
res_20_19 = verify_exercise_20_19()
res_20_20 = verify_exercise_20_20()

print("【演習 20.16 〜 20.20 検証結果】")
print(f"  - 演習 20.16 (ODE ドリフト計算)       : drift={res_20_16['ode_drift']:.4f}")
print(f"  - 演習 20.17 (p(y) 勾配消去確認)      : grad={res_20_17['grad_ln_p_y']:.1f}")
print(f"  - 演習 20.18 (平均シフトベクトル誤差) : {res_20_18['diff']:.2e}")
print(f"  - 演習 20.19 (CFG スコア差分等価性差) : {res_20_19['diff']:.2e}")
print(f"  - 演習 20.20 (CFG 外挿特性の真偽)    : gamma=0一致={res_20_20['is_g0_uncond']}, gamma=1一致={res_20_20['is_g1_cond']}, gamma>1外挿={res_20_20['is_g2_extrap']}")

assert not np.isnan(res_20_16['ode_drift']), "演習 20.16 不合格"
assert res_20_17['grad_ln_p_y'] == 0.0, "演習 20.17 不合格"
assert res_20_18['diff'] < 1e-12, "演習 20.18 不合格"
assert res_20_19['diff'] < 1e-12, "演習 20.19 不合格"
assert res_20_20['is_g0_uncond'] and res_20_20['is_g1_cond'] and res_20_20['is_g2_extrap'], "演習 20.20 不合格"
print("演習 20.16 〜 20.20: すべて合格！")
"""
    cells.append(nbf.v4.new_code_cell(cell_ex16_20_code))

    # Cell: Chapter 20 Exercises Summary
    cell_summary_md = r"""---

## 第20章 演習問題 総括 (Chapter Summary)

第20章の全20問の演習問題を通じて、拡散モデル（Diffusion Models）の数理的基盤を完全に制覇しました：

1. **マルコフ拡散カーネルと閉形式周辺化 (演習 20.1 〜 20.6)**:
   - ガウス分布の再生性により、任意ステップへの直接サンプリング公式 $q(\mathbf{z}_t \mid \mathbf{x}) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}, (1 - \bar{\alpha}_t)\mathbf{I})$ が成立し、$\text{SNR}(t)$ は単調減少して $t=T$ で標準正規事前分布へ収束する。
   - 条件付き事後分布 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$ の平方完成により、理想的な逆遷移平均 $\tilde{\boldsymbol{\mu}}_t$ と分散 $\tilde{\beta}_t$ が導出された。

2. **変分下界の望遠鏡展開とノイズ予測 (演習 20.7 〜 20.10)**:
   - 連続する遷移確率の比が打ち消し合う望遠鏡積により、ELBO は各ステップの平均二乗誤差へ帰着される。
   - 事後平均 $\tilde{\boldsymbol{\mu}}_t$ をノイズ $\boldsymbol{\epsilon}$ で表現し直すことで、ノイズ予測器 $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)$ を用いた簡便で安定な目的関数 $L_{\text{simple}}(\theta)$ が導かれた。

3. **スコアマッチングと確率微分方程式 (演習 20.11 〜 20.16)**:
   - スコア関数 $\nabla \ln p(\mathbf{x})$ は正規化定数 $Z$ に依存せず、陰的・デノイジングスコアマッチングによってデータから直接推定可能である。
   - 連続時間極限において離散拡散は伊藤 SDE に帰着され、時間反転 SDE および決定論的確率流 ODE によって生成サンプリングと厳密な尤度評価が可能となる。

4. **条件付き生成とガイダンス技術 (演習 20.17 〜 20.20)**:
   - 分類器ガイダンスおよび分類器なしガイダンス（CFG）は、無条件スコアに条件付きスコアの差分をスケール $\gamma$ で外挿・増幅することで、驚異的な画像忠実度とクラス適合度を達成する。
"""
    cells.append(nbf.v4.new_markdown_cell(cell_summary_md))

    nb.cells = cells

    out_path = os.path.join(os.path.dirname(__file__), "..", "20", "20_Exercises.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated {out_path} with {len(cells)} cells.")


if __name__ == "__main__":
    build_notebook()

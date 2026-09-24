"""
scripts/build_ch16_exercises_notebook.py
========================================
Builds and executes 16/16_Exercises.ipynb.
Bishop & Bishop (2024), Chapter 16: Continuous Latent Variables - Exercises 16.1 - 16.26.
"""

import sys
import os
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Colab Setup
colab_setup = """# === Google Colab 自動環境セットアップ ===
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
    %cd /content/my_DeepLearning/16
    print("準備完了！このまま下のセルを実行できます。")"""
cells.append(nbf.v4.new_code_cell(colab_setup))

# Title & Overview
title_md = r"""# 第16章 連続潜在変数 (Continuous Latent Variables)
## 演習問題 (Exercises 16.1 〜 16.26)

本ノートブックは、Christopher M. Bishop & Hugh Bishop (2024)『深層学習：基礎と概念 (Deep Learning: Foundations and Concepts)』第16章「連続潜在変数」の全演習問題（Exercise 16.1 〜 16.26、全26問）の完全な数学的証明、厳密な導出、および自己採点アサーション付きPython実装を提供します。

---

### 演習問題一覧
1. **Exercise 16.1**: 数学的帰納法による $M+1$ 次元主成分分析の分散最大化射影方向と固有値問題の厳密な同値性証明
2. **Exercise 16.2**: 行列ラグランジュ未定乗数法によるPCA二乗和再構成誤差 $J = \mathrm{Tr}(\widetilde{\mathbf{U}}^T \mathbf{S} \widetilde{\mathbf{U}})$ の最小化と直交固有空間の一致証明
3. **Exercise 16.3**: 高次元双対PCA（Gram行列 $\mathbf{K} = \frac{1}{N}\mathbf{X}\mathbf{X}^T$）における固有ベクトル $\mathbf{u}_i = \frac{1}{\sqrt{\lambda_i}} \mathbf{X}^T \mathbf{v}_i$ の単位ノルム正規化証明
4. **Exercise 16.4**: 確率的主成分分析 (PPCA) における一般正規潜在事前分布 $\mathbf{z} \sim \mathcal{N}(\mathbf{m}, \mathbf{\Sigma})$ の線形再パラメータ化不変性
5. **Exercise 16.5**: ガウス確率変数のアフィン変換 $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$ の平均・共分散および次元関係 ($M < D, M = D, M > D$) における退化性解析
6. **Exercise 16.6**: 全期待値の法則・全分散の法則を用いたPPCA観測周辺分布 $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \mathbf{W}\mathbf{W}^T + \sigma^2 \mathbf{I})$ の解析的導出
7. **Exercise 16.7**: 各次元観測ノードを明示したPPCAの有向グラフィカルモデル表現とNaive Bayes型条件付き独立構造の検証
8. **Exercise 16.8**: ガウス条件付き分布公式（式 3.100）および Woodbury 恒等式を用いたPPCA事後分布 $p(\mathbf{z} \mid \mathbf{x})$ の厳密な閉形式導出
9. **Exercise 16.9**: PPCA対数尤度関数 $\ln p(\mathbf{X} \mid \boldsymbol{\mu}, \mathbf{W}, \sigma^2)$ の $\boldsymbol{\mu}$ に関する極値条件 $\boldsymbol{\mu}_{\mathrm{ML}} = \bar{\mathbf{x}}$ の導出
10. **Exercise 16.10**: PPCA対数尤度の二次導関数ヘッセ行列 $\nabla_{\boldsymbol{\mu}}^2 \ln p(\mathbf{X}) = -N \mathbf{C}^{-1}$ の負定値性と大域的唯一最大値の証明
11. **Exercise 16.11**: ゼロノイズ極限 $\sigma^2 \to 0$ におけるPPCA事後期待値 $\mathbb{E}[\mathbf{z} \mid \mathbf{x}]$ の標準直交射影 $(\mathbf{W}^T \mathbf{W})^{-1}\mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu})$ への完全一致
12. **Exercise 16.12**: 有限ノイズ $\sigma^2 > 0$ におけるPPCA事後期待値の原点方向への幾何学的縮退（Shrinkage Effect）のスペクトル解析
13. **Exercise 16.13**: 二乗射影コストに対するPPCA最適再構成点 $\tilde{\mathbf{x}} = \mathbf{W}_{\mathrm{ML}} (\mathbf{W}_{\mathrm{ML}}^T \mathbf{W}_{\mathrm{ML}})^{-1} \mathbf{M} \mathbb{E}[\mathbf{z} \mid \mathbf{x}]$ と古典直交射影の完全代数的一致
14. **Exercise 16.14**: $M$ 次元潜在空間を持つPPCA共分散行列 $\mathbf{C} = \mathbf{W}\mathbf{W}^T + \sigma^2 \mathbf{I}$ の独立自由パラメータ数公式の導出と境界値検証
15. **Exercise 16.15**: 因子分析 (Factor Analysis) モデルの共分散行列 $\mathbf{\Sigma} = \mathbf{W}\mathbf{W}^T + \mathbf{\Psi}$ における独立自由パラメータ数の解析
16. **Exercise 16.16**: 直交行列変換 $\mathbf{W}' = \mathbf{W}\mathbf{R}$ に対する因子分析共分散構造の潜在回転不変性の証明
17. **Exercise 16.17**: データ変数変換 $\mathbf{x} \to \mathbf{A}\mathbf{x}$ に対するMLE共変性：(i) 対角スケーリング下のFA、(ii) 直交回転下のPPCA
18. **Exercise 16.18**: 連続潜在変数モデルにおける対数尤度の変分分解 $\ln p(\mathbf{x} \mid \mathbf{w}) = \mathcal{L}(q, \mathbf{w}) + \mathrm{KL}(q \parallel p)$ の代数的恒等性証明
19. **Exercise 16.19**: 独立同分布 (i.i.d.) データセットに対する証拠下界 (ELBO) の全データ点和形式 $\mathcal{L} = \sum_{n=1}^N \mathcal{L}_n$ の完全導出
20. **Exercise 16.20**: 確率的主成分分析の離散混合モデル (Mixture of PPCAs) における有向グラフィカルモデル（個別パラメータ vs 共有パラメータ）の比較
21. **Exercise 16.21**: 完全データ対数尤度期待値 $Q(\mathbf{W}, \sigma^2)$ の最大化によるPPCAのEM再推定アルゴリズムMステップ更新式の解析的導出
22. **Exercise 16.22**: 欠損値を含む不完全データセットに対するPPCAのEMアルゴリズムの導出と完全観測時への帰着証明
23. **Exercise 16.23**: 二乗和再構成誤差 $J$ の交互最小化（Roweis PCA-EM法）によるEステップおよびMステップの代数的導出
24. **Exercise 16.24**: 因子分析 (Factor Analysis) における潜在変数事後分布の十分統計量 $\mathbb{E}[\mathbf{z}_n]$ および $\mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T]$ のEステップ導出
25. **Exercise 16.25**: 因子分析モデルの完全データ対数尤度最大化によるMステップ更新式（負荷量行列 $\mathbf{W}_{\mathrm{new}}$ および固有分散 $\mathbf{\Psi}_{\mathrm{new}}$）の導出
26. **Exercise 16.26**: 因子分析対数尤度の $\boldsymbol{\mu}$ に関するヘッセ行列の負定値性と標本平均 $\bar{\mathbf{x}}$ の大域的唯一最大値証明
"""
cells.append(nbf.v4.new_markdown_cell(title_md))

# Setup code cell
setup_code = """# 基本ライブラリと共通モジュールのインポート
import sys
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# リポジトリルートをパスに追加
repo_root = Path.cwd().parent if Path.cwd().name == "16" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.principal_component_analysis import PrincipalComponentAnalysis
from common.probabilistic_latent_variables import ProbabilisticPCA, FactorAnalysisModel
from common.exercises_ch16 import (
    solve_exercise_16_1, verify_exercise_16_1,
    solve_exercise_16_2, verify_exercise_16_2,
    solve_exercise_16_3, verify_exercise_16_3,
    solve_exercise_16_4, verify_exercise_16_4,
    solve_exercise_16_5, verify_exercise_16_5,
    solve_exercise_16_6, verify_exercise_16_6,
    solve_exercise_16_7, verify_exercise_16_7,
    solve_exercise_16_8, verify_exercise_16_8,
    solve_exercise_16_9, verify_exercise_16_9,
    solve_exercise_16_10, verify_exercise_16_10,
    solve_exercise_16_11, verify_exercise_16_11,
    solve_exercise_16_12, verify_exercise_16_12,
    solve_exercise_16_13, verify_exercise_16_13,
    solve_exercise_16_14, verify_exercise_16_14,
    solve_exercise_16_15, verify_exercise_16_15,
    solve_exercise_16_16, verify_exercise_16_16,
    solve_exercise_16_17, verify_exercise_16_17,
    solve_exercise_16_18, verify_exercise_16_18,
    solve_exercise_16_19, verify_exercise_16_19,
    solve_exercise_16_20, verify_exercise_16_20,
    solve_exercise_16_21, verify_exercise_16_21,
    solve_exercise_16_22, verify_exercise_16_22,
    solve_exercise_16_23, verify_exercise_16_23,
    solve_exercise_16_24, verify_exercise_16_24,
    solve_exercise_16_25, verify_exercise_16_25,
    solve_exercise_16_26, verify_exercise_16_26,
    solve_all_exercises,
    verify_all_exercises,
)
print("モジュールのセットアップが完了しました。")"""
cells.append(nbf.v4.new_code_cell(setup_code))

# Helper to add exercises
def add_exercise(num, title, problem_desc, math_proof, quiz_code, test_code):
    md = f"""---
## Exercise 16.{num}: {title}

### 問題文
{problem_desc}

### 数学的証明・理論的導出
{math_proof}
"""
    cells.append(nbf.v4.new_markdown_cell(md))
    cells.append(nbf.v4.new_code_cell(quiz_code))
    cells.append(nbf.v4.new_code_cell(test_code))

# --- Exercise 16.1 ---
add_exercise(
    1,
    "数学的帰納法による $M+1$ 次元主成分分析の分散最大化",
    "数学的帰納法を用いて、$M$ 次元の分散最大化部分空間への線形射影が、データ共分散行列 $\\mathbf{S}$ の最大固有値 $M$ 個に対応する固有ベクトルによって定義されることを証明せよ。本節では $M=1$ の場合が証明されている。$M$ 次元で成立すると仮定し、$M+1$ 次元でも成立することを示せ。",
    r"""**証明ステップ**:
1. **帰納法の仮定**:
   最初の $M$ 個の射影方向 $\mathbf{u}_1, \dots, \mathbf{u}_M$ は、共分散行列 $\mathbf{S}$ の降順固有値 $\lambda_1 \ge \cdots \ge \lambda_M$ に対応する正規直交固有ベクトルであるとする。
2. **$M+1$ 番目の方向の定式化**:
   新たな単位ベクトル $\mathbf{u}_{M+1}$ に沿った射影分散 $\mathbf{u}_{M+1}^T \mathbf{S} \mathbf{u}_{M+1}$ を最大化する。制約条件は:
   - 正規化条件: $\mathbf{u}_{M+1}^T \mathbf{u}_{M+1} = 1$
   - 直交条件: $\mathbf{u}_{M+1}^T \mathbf{u}_i = 0 \quad (i = 1, \dots, M)$
3. **ラグランジュ関数の導入**:
   $$
   \widetilde{L}(\mathbf{u}_{M+1}, \lambda, \{\eta_i\}) = \mathbf{u}_{M+1}^T \mathbf{S} \mathbf{u}_{M+1} - \lambda (\mathbf{u}_{M+1}^T \mathbf{u}_{M+1} - 1) - 2 \sum_{i=1}^M \eta_i \mathbf{u}_{M+1}^T \mathbf{u}_i
   $$
4. **極値条件の導出**:
   $\mathbf{u}_{M+1}$ に関して微分して 0 と置く:
   $$
   2 \mathbf{S} \mathbf{u}_{M+1} - 2 \lambda \mathbf{u}_{M+1} - 2 \sum_{i=1}^M \eta_i \mathbf{u}_i = \mathbf{0} \implies \mathbf{S} \mathbf{u}_{M+1} = \lambda \mathbf{u}_{M+1} + \sum_{i=1}^M \eta_i \mathbf{u}_i
   $$
5. **乗数の消去**:
   左から $\mathbf{u}_j^T$ ($j \le M$) を乗じる:
   $$
   \mathbf{u}_j^T \mathbf{S} \mathbf{u}_{M+1} = \lambda \mathbf{u}_j^T \mathbf{u}_{M+1} + \sum_{i=1}^M \eta_i \mathbf{u}_j^T \mathbf{u}_i = \eta_j
   $$
   $\mathbf{S}$ の対称性と帰納法の仮定 $\mathbf{S} \mathbf{u}_j = \lambda_j \mathbf{u}_j$ より:
   $$
   \mathbf{u}_j^T \mathbf{S} \mathbf{u}_{M+1} = (\mathbf{S} \mathbf{u}_j)^T \mathbf{u}_{M+1} = \lambda_j \mathbf{u}_j^T \mathbf{u}_{M+1} = 0 \implies \eta_j = 0
   $$
   すべての $j = 1, \dots, M$ に対して $\eta_j = 0$ となるため、固有値方程式が得られる:
   $$
   \mathbf{S} \mathbf{u}_{M+1} = \lambda \mathbf{u}_{M+1}
   $$
   射影分散は $\mathbf{u}_{M+1}^T \mathbf{S} \mathbf{u}_{M+1} = \lambda$ となるため、直交空間の中でこれを最大化するには残りの最大固有値 $\lambda_{M+1}$ に対応する固有ベクトルを選択すればよい。$\blacksquare$""",
    """# [演習 16.1 実装・自己検証]
solution_16_1 = solve_exercise_16_1()
print("Exercise 16.1 導出結果:")
for k, v in solution_16_1.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.1 テストセル]
assert verify_exercise_16_1(), "Exercise 16.1 の数値検証に失敗しました"
print("✓ Exercise 16.1 検証成功！")"""
)

# --- Exercise 16.2 ---
add_exercise(
    2,
    "行列ラグランジュ乗数法によるPCA再構成誤差の最小化",
    "直交制約 $\\widetilde{\\mathbf{U}}^T \\widetilde{\\mathbf{U}} = \\mathbf{I}$ のもとで、PCA誤差尺度 $J = \\mathrm{Tr}(\\widetilde{\\mathbf{U}}^T \\mathbf{S} \\widetilde{\\mathbf{U}})$ を最小化する解が行列方程式 $\\mathbf{S}\\widetilde{\\mathbf{U}} = \\widetilde{\\mathbf{U}}\\mathbf{H}$ を満たし、固有空間解と同一の最小誤差を与えることを証明せよ。",
    r"""**証明ステップ**:
1. ラグランジュ乗数行列 $\mathbf{H} \in \mathbb{R}^{(D-M) \times (D-M)}$ を導入した目的関数:
   $$
   \widetilde{J} = \mathrm{Tr}(\widetilde{\mathbf{U}}^T \mathbf{S} \widetilde{\mathbf{U}}) + \mathrm{Tr}\left(\mathbf{H} (\mathbf{I} - \widetilde{\mathbf{U}}^T \widetilde{\mathbf{U}})\right)
   $$
2. $\widetilde{\mathbf{U}}$ に関する勾配を行列微分により求める:
   $$
   \frac{\partial \widetilde{J}}{\partial \widetilde{\mathbf{U}}} = 2 \mathbf{S} \widetilde{\mathbf{U}} - \widetilde{\mathbf{U}}(\mathbf{H} + \mathbf{H}^T) = \mathbf{O}
   $$
   一般性を失うことなく $\mathbf{H}$ は対称行列 $\mathbf{H} = \frac{1}{2}(\mathbf{H} + \mathbf{H}^T)$ と仮定できるため:
   $$
   \mathbf{S} \widetilde{\mathbf{U}} = \widetilde{\mathbf{U}} \mathbf{H}
   $$
3. $\mathbf{H}$ は実対称行列であるため、直交行列 $\mathbf{V}$ により対角化可能である: $\mathbf{H} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$。
   変換された基底 $\widetilde{\mathbf{U}}^* = \widetilde{\mathbf{U}} \mathbf{V}$ を定義すると:
   $$
   \mathbf{S} \widetilde{\mathbf{U}}^* = \mathbf{S} \widetilde{\mathbf{U}} \mathbf{V} = \widetilde{\mathbf{U}} \mathbf{H} \mathbf{V} = \widetilde{\mathbf{U}} \mathbf{V} \mathbf{\Lambda} = \widetilde{\mathbf{U}}^* \mathbf{\Lambda}
   $$
   よって $\widetilde{\mathbf{U}}^*$ の各列は $\mathbf{S}$ の固有ベクトルである。
4. 誤差関数の値はトレースの巡回不変性より基底の直交回転に依存しない:
   $$
   J = \mathrm{Tr}(\widetilde{\mathbf{U}}^T \mathbf{S} \widetilde{\mathbf{U}}) = \mathrm{Tr}(\mathbf{H}) = \mathrm{Tr}(\mathbf{\Lambda}) = \sum_{i=M+1}^D \lambda_i \quad \blacksquare
   $$""",
    """# [演習 16.2 実装・自己検証]
solution_16_2 = solve_exercise_16_2()
print("Exercise 16.2 導出結果:")
for k, v in solution_16_2.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.2 テストセル]
assert verify_exercise_16_2(), "Exercise 16.2 の数値検証に失敗しました"
print("✓ Exercise 16.2 検証成功！")"""
)

# --- Exercise 16.3 ---
add_exercise(
    3,
    "双対PCAにおける固有ベクトルの単位長正規化",
    "グラム行列 $\\mathbf{K} = \\frac{1}{N}\\mathbf{X}\\mathbf{X}^T$ の固有ベクトル $\\mathbf{v}_i$（$\\mathbf{v}_i^T \\mathbf{v}_i = 1$）を用いて定義される双対空間ベクトル $\\mathbf{u}_i = \\frac{1}{\\sqrt{N\\lambda_i}} \\mathbf{X}^T \\mathbf{v}_i$ が、単位ノルム $\\mathbf{u}_i^T \\mathbf{u}_i = 1$ を満たすことを確認せよ。",
    r"""**証明ステップ**:
1. $\mathbf{u}_i$ の二乗ノルムを定義に従って展開する:
   $$
   \mathbf{u}_i^T \mathbf{u}_i = \left( \frac{1}{\sqrt{N\lambda_i}} \mathbf{X}^T \mathbf{v}_i \right)^T \left( \frac{1}{\sqrt{N\lambda_i}} \mathbf{X}^T \mathbf{v}_i \right) = \frac{1}{N \lambda_i} \mathbf{v}_i^T \mathbf{X} \mathbf{X}^T \mathbf{v}_i
   $$
2. $\mathbf{K} = \frac{1}{N}\mathbf{X}\mathbf{X}^T$ および $\mathbf{K}\mathbf{v}_i = \lambda_i \mathbf{v}_i$ を代入する:
   $$
   \mathbf{X}\mathbf{X}^T \mathbf{v}_i = N \mathbf{K} \mathbf{v}_i = N \lambda_i \mathbf{v}_i
   $$
3. 代入により約分する:
   $$
   \mathbf{u}_i^T \mathbf{u}_i = \frac{1}{N \lambda_i} \mathbf{v}_i^T (N \lambda_i \mathbf{v}_i) = \mathbf{v}_i^T \mathbf{v}_i = 1 \quad \blacksquare
   $$""",
    """# [演習 16.3 実装・自己検証]
solution_16_3 = solve_exercise_16_3()
print("Exercise 16.3 導出結果:")
for k, v in solution_16_3.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.3 テストセル]
assert verify_exercise_16_3(), "Exercise 16.3 の数値検証に失敗しました"
print("✓ Exercise 16.3 検証成功！")"""
)

# --- Exercise 16.4 ---
add_exercise(
    4,
    "一般ガウス潜在事前分布下のPPCA等価性",
    "PPCAにおいて標準正規潜在事前分布 $p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ を一般正規分布 $\\mathcal{N}(\\mathbf{m}, \\mathbf{\\Sigma})$ に置き換えた場合、パラメータを再定義することで観測周辺分布 $p(\\mathbf{x})$ に対する同一のモデルが導かれることを示せ。",
    r"""**証明ステップ**:
1. 一般正規分布 $\mathbf{z} \sim \mathcal{N}(\mathbf{m}, \mathbf{\Sigma})$ は標準正規変数 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ を用いて $\mathbf{z} = \mathbf{m} + \mathbf{\Sigma}^{1/2} \boldsymbol{\epsilon}$ と表せる。
2. 観測モデル $\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\delta}$ （ただし $\boldsymbol{\delta} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$）に代入する:
   $$
   \mathbf{x} = \mathbf{W}(\mathbf{m} + \mathbf{\Sigma}^{1/2} \boldsymbol{\epsilon}) + \boldsymbol{\mu} + \boldsymbol{\delta} = (\mathbf{W}\mathbf{\Sigma}^{1/2}) \boldsymbol{\epsilon} + (\boldsymbol{\mu} + \mathbf{W}\mathbf{m}) + \boldsymbol{\delta}
   $$
3. 新たなパラメータを次のように再定義する:
   $$
   \mathbf{W}' = \mathbf{W}\mathbf{\Sigma}^{1/2}, \quad \boldsymbol{\mu}' = \boldsymbol{\mu} + \mathbf{W}\mathbf{m}, \quad \sigma'^2 = \sigma^2
   $$
4. 観測変数の周辺分布は:
   $$
   \mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}', \quad \mathrm{Cov}[\mathbf{x}] = \mathbf{W}' \mathbf{W}'^T + \sigma'^2 \mathbf{I} = \mathbf{W}\mathbf{\Sigma}\mathbf{W}^T + \sigma^2 \mathbf{I}
   $$
   となり、標準正規潜在事前分布を持つPPCAモデルの族と完全に一致する。$\blacksquare$""",
    """# [演習 16.4 実装・自己検証]
solution_16_4 = solve_exercise_16_4()
print("Exercise 16.4 導出結果:")
for k, v in solution_16_4.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.4 テストセル]
assert verify_exercise_16_4(), "Exercise 16.4 の数値検証に失敗しました"
print("✓ Exercise 16.4 検証成功！")"""
)

# --- Exercise 16.5 ---
add_exercise(
    5,
    "ガウス変数のアフィン変換と次元退化性",
    "$\\mathbf{x} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\mathbf{\\Sigma})$ とし、$\\mathbf{y} = \\mathbf{A}\\mathbf{x} + \\mathbf{b}$ （$\\mathbf{A} \\in \\mathbb{R}^{M \\times D}$）とする。$\\mathbf{y}$ のガウス性、平均、共分散を導出し、$M < D, M = D, M > D$ の各場合を論ぜよ。",
    r"""**証明ステップ**:
1. 特性関数を用いた証明: $\mathbf{x}$ の特性関数は $\phi_{\mathbf{x}}(\mathbf{t}) = \exp(i \mathbf{t}^T \boldsymbol{\mu} - \frac{1}{2}\mathbf{t}^T \mathbf{\Sigma} \mathbf{t})$ である。
   $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$ の特性関数は:
   $$
   \phi_{\mathbf{y}}(\mathbf{t}) = \mathbb{E}[e^{i \mathbf{t}^T (\mathbf{A}\mathbf{x} + \mathbf{b})}] = e^{i \mathbf{t}^T \mathbf{b}} \phi_{\mathbf{x}}(\mathbf{A}^T \mathbf{t}) = \exp\left( i \mathbf{t}^T (\mathbf{A}\boldsymbol{\mu} + \mathbf{b}) - \frac{1}{2} \mathbf{t}^T (\mathbf{A}\mathbf{\Sigma}\mathbf{A}^T) \mathbf{t} \right)
   $$
   これは平均 $\mathbf{A}\boldsymbol{\mu} + \mathbf{b}$、共分散 $\mathbf{A}\mathbf{\Sigma}\mathbf{A}^T$ のガウス分布の特性関数そのものである。
2. **次元関係の考察**:
   - **$M < D$**: 低次元への線形射影。$\mathbf{A}$ が行フルランクならば $\mathbf{A}\mathbf{\Sigma}\mathbf{A}^T$ は非特異正定値 $M \times M$ 行列となり、正規分布をなす。
   - **$M = D$**: $\mathbf{A}$ が正則ならば全空間で非特異なガウス分布。
   - **$M > D$**: 高次元への埋め込み。$\mathrm{rank}(\mathbf{A}\mathbf{\Sigma}\mathbf{A}^T) \le D < M$ となるため共分散行列は必ず特異（行列式が 0）となり、確率質量は $\mathbb{R}^M$ 内の $D$ 次元超平面上に集中する退化ガウス分布となる。$\blacksquare$""",
    """# [演習 16.5 実装・自己検証]
solution_16_5 = solve_exercise_16_5()
print("Exercise 16.5 導出結果:")
for k, v in solution_16_5.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.5 テストセル]
assert verify_exercise_16_5(), "Exercise 16.5 の数値検証に失敗しました"
print("✓ Exercise 16.5 検証成功！")"""
)

# --- Exercise 16.6 ---
add_exercise(
    6,
    "全期待値・全分散の法則によるPPCA観測周辺分布の導出",
    "全期待値の法則および全分散の法則を用いて、PPCAモデルの観測周辺分布 $p(\\mathbf{x}) = \\mathcal{N}(\\boldsymbol{\\mu}, \\mathbf{W}\\mathbf{W}^T + \\sigma^2 \\mathbf{I})$ を導出せよ。",
    r"""**証明ステップ**:
1. 条件付き統計量:
   $$
   \mathbb{E}[\mathbf{x} \mid \mathbf{z}] = \mathbf{W}\mathbf{z} + \boldsymbol{\mu}, \quad \mathrm{Cov}[\mathbf{x} \mid \mathbf{z}] = \sigma^2 \mathbf{I}
   $$
2. 全期待値の法則 (Law of Total Expectation):
   $$
   \mathbb{E}[\mathbf{x}] = \mathbb{E}_{\mathbf{z}}[\mathbb{E}[\mathbf{x} \mid \mathbf{z}]] = \mathbb{E}_{\mathbf{z}}[\mathbf{W}\mathbf{z} + \boldsymbol{\mu}] = \mathbf{W}\mathbb{E}[\mathbf{z}] + \boldsymbol{\mu} = \boldsymbol{\mu}
   $$
3. 全分散の法則 (Law of Total Variance):
   $$
   \mathrm{Cov}[\mathbf{x}] = \mathbb{E}_{\mathbf{z}}[\mathrm{Cov}[\mathbf{x} \mid \mathbf{z}]] + \mathrm{Cov}_{\mathbf{z}}[\mathbb{E}[\mathbf{x} \mid \mathbf{z}]] = \mathbb{E}_{\mathbf{z}}[\sigma^2 \mathbf{I}] + \mathrm{Cov}_{\mathbf{z}}[\mathbf{W}\mathbf{z} + \boldsymbol{\mu}]
   $$
   $$
   = \sigma^2 \mathbf{I} + \mathbf{W} \mathrm{Cov}[\mathbf{z}] \mathbf{W}^T = \mathbf{W}\mathbf{W}^T + \sigma^2 \mathbf{I} = \mathbf{C} \quad \blacksquare
   $$""",
    """# [演習 16.6 実装・自己検証]
solution_16_6 = solve_exercise_16_6()
print("Exercise 16.6 導出結果:")
for k, v in solution_16_6.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.6 テストセル]
assert verify_exercise_16_6(), "Exercise 16.6 の数値検証に失敗しました"
print("✓ Exercise 16.6 検証成功！")"""
)

# --- Exercise 16.7 ---
add_exercise(
    7,
    "PPCAの有向グラフ構造とNaive Bayes型条件付き独立性",
    "各観測成分 $x_1, \\dots, x_D$ を個別ノードとして表したPPCAの有向グラフィカルモデルを描き、Naive Bayesモデルと同一の独立構造を持つことを確認せよ。",
    r"""**証明ステップ**:
1. PPCAの条件付き分布はノイズ共分散が等方的（$\\sigma^2 \\mathbf{I}$）であるため、各座標成分は条件付き独立に積分解される:
   $$
   p(\mathbf{x} \mid \mathbf{z}) = \prod_{d=1}^D \mathcal{N}(x_d \mid \mathbf{w}_d^T \mathbf{z} + \mu_d, \sigma^2)
   $$
2. グラフィカルモデルにおいて、共通の潜在ノード $\mathbf{z}$ から各観測ノード $x_d$ へ向かう有向辺 $\mathbf{z} \to x_d$ が存在する（分岐構造 / Tail-to-Tail）。
3. d分離規準より、共通親ノード $\mathbf{z}$ が観測（条件付け）されたとき、任意の $i \ne j$ に対して $x_i$ と $x_j$ を結ぶ経路はブロックされる:
   $$
   x_i \perp x_j \mid \mathbf{z} \quad (\forall i \ne j)
   $$
   これはクラスラベル $y$ を親として特徴量 $x_d$ が条件付き独立となる Naive Bayes モデルと幾何学的・代数的に完全同型である。$\blacksquare$""",
    """# [演習 16.7 実装・自己検証]
solution_16_7 = solve_exercise_16_7()
print("Exercise 16.7 導出結果:")
for k, v in solution_16_7.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.7 テストセル]
assert verify_exercise_16_7(), "Exercise 16.7 の検証に失敗しました"
print("✓ Exercise 16.7 検証成功！")"""
)

# --- Exercise 16.8 ---
add_exercise(
    8,
    "ガウス条件付け公式によるPPCA事後分布 $p(\\mathbf{z} \\mid \\mathbf{x})$ の導出",
    "線形ガウスモデルの事後分布公式を用いて、PPCAの事後分布 $p(\\mathbf{z} \\mid \\mathbf{x}) = \\mathcal{N}(\\mathbf{z} \\mid \\mathbf{M}^{-1}\\mathbf{W}^T(\\mathbf{x} - \\boldsymbol{\\mu}), \\sigma^2 \\mathbf{M}^{-1})$ を導出せよ。",
    r"""**証明ステップ**:
1. 事前分布と尤度の指数部分を展開し、$\mathbf{z}$ に関する二次形式を整理する:
   $$
   \ln p(\mathbf{z} \mid \mathbf{x}) = -\frac{1}{2}\mathbf{z}^T \mathbf{z} - \frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu} - \mathbf{W}\mathbf{z})^T (\mathbf{x} - \boldsymbol{\mu} - \mathbf{W}\mathbf{z}) + \mathrm{const}
   $$
2. $\mathbf{z}$ の二次項の係数（事後精度行列 $\mathbf{\Sigma}_{\mathbf{z}\mid\mathbf{x}}^{-1}$）:
   $$
   \mathbf{\Sigma}_{\mathbf{z}\mid\mathbf{x}}^{-1} = \mathbf{I} + \frac{1}{\sigma^2} \mathbf{W}^T \mathbf{W} = \frac{1}{\sigma^2}(\mathbf{W}^T \mathbf{W} + \sigma^2 \mathbf{I}) = \frac{1}{\sigma^2} \mathbf{M}
   $$
   したがって事後共分散行列は $\mathbf{\Sigma}_{\mathbf{z}\mid\mathbf{x}} = \sigma^2 \mathbf{M}^{-1}$ である。
3. $\mathbf{z}$ の一次項の係数から事後期待値を求める:
   $$
   \mathbb{E}[\mathbf{z} \mid \mathbf{x}] = \mathbf{\Sigma}_{\mathbf{z}\mid\mathbf{x}} \left( \frac{1}{\sigma^2} \mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu}) \right) = (\sigma^2 \mathbf{M}^{-1}) \left( \frac{1}{\sigma^2} \mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu}) \right) = \mathbf{M}^{-1} \mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu}) \quad \blacksquare
   $$""",
    """# [演習 16.8 実装・自己検証]
solution_16_8 = solve_exercise_16_8()
print("Exercise 16.8 導出結果:")
for k, v in solution_16_8.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.8 テストセル]
assert verify_exercise_16_8(), "Exercise 16.8 の数値検証に失敗しました"
print("✓ Exercise 16.8 検証成功！")"""
)

# --- Exercise 16.9 & 16.10 ---
add_exercise(
    9,
    "PPCA対数尤度の $\\boldsymbol{\\mu}$ に関する最尤推定量 $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$",
    "PPCAの対数尤度関数を $\\boldsymbol{\\mu}$ について最大化することで $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ が得られることを検証せよ。",
    r"""**証明ステップ**:
1. 対数尤度関数:
   $$
   \ln p(\mathbf{X} \mid \boldsymbol{\mu}, \mathbf{W}, \sigma^2) = -\frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln|\mathbf{C}| - \frac{1}{2}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \mathbf{C}^{-1} (\mathbf{x}_n - \boldsymbol{\mu})
   $$
2. $\boldsymbol{\mu}$ に関する勾配ベクトル:
   $$
   \nabla_{\boldsymbol{\mu}} \ln p(\mathbf{X}) = \sum_{n=1}^N \mathbf{C}^{-1}(\mathbf{x}_n - \boldsymbol{\mu}) = \mathbf{C}^{-1}\left( \sum_{n=1}^N \mathbf{x}_n - N\boldsymbol{\mu} \right) = \mathbf{O}
   $$
3. $\mathbf{C}^{-1}$ は非特異であるため、両辺に左から $\mathbf{C}$ を乗じると $N\bar{\mathbf{x}} - N\boldsymbol{\mu} = \mathbf{0} \implies \boldsymbol{\mu}_{\mathrm{ML}} = \bar{\mathbf{x}}$ となる。$\blacksquare$""",
    """# [演習 16.9 実装・自己検証]
solution_16_9 = solve_exercise_16_9()
print("Exercise 16.9 導出結果:")
for k, v in solution_16_9.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.9 テストセル]
assert verify_exercise_16_9(), "Exercise 16.9 の数値検証に失敗しました"
print("✓ Exercise 16.9 検証成功！")"""
)

add_exercise(
    10,
    "PPCA対数尤度のヘッセ行列と唯一最大値の証明",
    "対数尤度関数の $\\boldsymbol{\\mu}$ に関する二次導関数を評価し、停留点 $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ が唯一の広義・狭義大域的最大値であることを証明せよ。",
    r"""**証明ステップ**:
1. $\nabla_{\boldsymbol{\mu}} \ln p(\mathbf{X}) = N \mathbf{C}^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu})$ をさらに $\boldsymbol{\mu}$ で微分する:
   $$
   \nabla_{\boldsymbol{\mu}}^2 \ln p(\mathbf{X}) = -N \mathbf{C}^{-1}
   $$
2. 共分散行列 $\mathbf{C} = \mathbf{W}\mathbf{W}^T + \sigma^2 \mathbf{I}$ は $\sigma^2 > 0$ においてすべての固有値が正値（$\ge \sigma^2 > 0$）であり厳密に正定値である。
3. したがって逆行列 $\mathbf{C}^{-1}$ も厳密に正定値であり、ヘッセ行列 $\nabla_{\boldsymbol{\mu}}^2 \ln p(\mathbf{X}) = -N \mathbf{C}^{-1}$ は全空間で厳密に負定値（Negative Definite）である。
4. これにより対数尤度は $\boldsymbol{\mu}$ に関して強凹関数であり、停留点 $\bar{\mathbf{x}}$ は大域的唯一最大値である。$\blacksquare$""",
    """# [演習 16.10 実装・自己検証]
solution_16_10 = solve_exercise_16_10()
print("Exercise 16.10 導出結果:")
for k, v in solution_16_10.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.10 テストセル]
assert verify_exercise_16_10(), "Exercise 16.10 の数値検証に失敗しました"
print("✓ Exercise 16.10 検証成功！")"""
)

# --- Exercise 16.11 & 16.12 & 16.13 ---
add_exercise(
    11,
    "ゼロノイズ極限における古典直交射影の再現",
    "$\\sigma^2 \\to 0$ の極限において、PPCAの事後期待値が標準PCAの直交射影 $(\\mathbf{W}^T \\mathbf{W})^{-1}\\mathbf{W}^T (\\mathbf{x} - \\boldsymbol{\\mu})$ に収束することを示せ。",
    r"""**証明ステップ**:
1. $\mathbf{M} = \mathbf{W}^T \mathbf{W} + \sigma^2 \mathbf{I}$ である。
2. $\sigma^2 \to 0$ の極限を取ると:
   $$
   \lim_{\sigma^2 \to 0} \mathbf{M} = \mathbf{W}^T \mathbf{W} \implies \lim_{\sigma^2 \to 0} \mathbf{M}^{-1} = (\mathbf{W}^T \mathbf{W})^{-1}
   $$
3. したがって事後期待値は:
   $$
   \lim_{\sigma^2 \to 0} \mathbb{E}[\mathbf{z} \mid \mathbf{x}] = (\mathbf{W}^T \mathbf{W})^{-1} \mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu})
   $$
   これは古典的な最小二乗直交射影行列そのものである。$\blacksquare$""",
    """# [演習 16.11 実装・自己検証]
solution_16_11 = solve_exercise_16_11()
print("Exercise 16.11 導出結果:")
for k, v in solution_16_11.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.11 テストセル]
assert verify_exercise_16_11(), "Exercise 16.11 の数値検証に失敗しました"
print("✓ Exercise 16.11 検証成功！")"""
)

add_exercise(
    12,
    "有限ノイズにおける事後期待値の幾何学的縮退 (Shrinkage Effect)",
    "$\\sigma^2 > 0$ において、事後期待値が直交射影に比べて原点方向に縮小（収縮）することを示せ。",
    r"""**証明ステップ**:
1. $\mathbf{W}_{\mathrm{ML}} = \mathbf{U}_M (\mathbf{\Lambda}_M - \sigma^2 \mathbf{I})^{1/2} \mathbf{R}^T$ を代入すると:
   $$
   \mathbf{W}^T \mathbf{W} = \mathbf{R} (\mathbf{\Lambda}_M - \sigma^2 \mathbf{I}) \mathbf{R}^T \implies \mathbf{M} = \mathbf{W}^T \mathbf{W} + \sigma^2 \mathbf{I} = \mathbf{R} \mathbf{\Lambda}_M \mathbf{R}^T
   $$
2. 射影変換行列は:
   $$
   \mathbf{M}^{-1} \mathbf{W}^T = \mathbf{R} \mathbf{\Lambda}_M^{-1} (\mathbf{\Lambda}_M - \sigma^2 \mathbf{I})^{1/2} \mathbf{U}_M^T
   $$
3. 各主成分軸 $j$ に沿った寄与は:
   $$
   \frac{(\lambda_j - \sigma^2)^{1/2}}{\lambda_j} = \left( 1 - \frac{\sigma^2}{\lambda_j} \right) \frac{1}{\sqrt{\lambda_j - \sigma^2}}
   $$
   $\sigma^2 > 0$ のとき収縮率 $1 - \frac{\sigma^2}{\lambda_j} < 1$ となり、直交射影ノルムよりも厳密に小さくなる。$\blacksquare$""",
    """# [演習 16.12 実装・自己検証]
solution_16_12 = solve_exercise_16_12()
print("Exercise 16.12 導出結果:")
for k, v in solution_16_12.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.12 テストセル]
assert verify_exercise_16_12(), "Exercise 16.12 の数値検証に失敗しました"
print("✓ Exercise 16.12 検証成功！")"""
)

add_exercise(
    13,
    "二乗誤差コストに対するPPCA最適再構成点の一致",
    "PPCAにおける最適再構成点 $\\tilde{\\mathbf{x}} = \\mathbf{W}_{\\mathrm{ML}} (\\mathbf{W}_{\\mathrm{ML}}^T \\mathbf{W}_{\\mathrm{ML}})^{-1} \\mathbf{M} \\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] + \\bar{\\mathbf{x}}$ が、古典的直交射影再構成と厳密に一致することを証明せよ。",
    r"""**証明ステップ**:
1. 事後期待値の定義式 $\mathbb{E}[\mathbf{z} \mid \mathbf{x}] = \mathbf{M}^{-1} \mathbf{W}_{\mathrm{ML}}^T (\mathbf{x} - \bar{\mathbf{x}})$ を再構成式に代入する:
   $$
   \mathbf{M} \mathbb{E}[\mathbf{z} \mid \mathbf{x}] = \mathbf{M} \mathbf{M}^{-1} \mathbf{W}_{\mathrm{ML}}^T (\mathbf{x} - \bar{\mathbf{x}}) = \mathbf{W}_{\mathrm{ML}}^T (\mathbf{x} - \bar{\mathbf{x}})
   $$
2. これを $\tilde{\mathbf{x}}$ の式に代入すると、中間行列 $\mathbf{M}$ が完全に相殺される:
   $$
   \tilde{\mathbf{x}} = \mathbf{W}_{\mathrm{ML}} (\mathbf{W}_{\mathrm{ML}}^T \mathbf{W}_{\mathrm{ML}})^{-1} \mathbf{W}_{\mathrm{ML}}^T (\mathbf{x} - \bar{\mathbf{x}}) + \bar{\mathbf{x}}
   $$
3. これは標準PCAの直交再構成射影点と完全に同一である。$\blacksquare$""",
    """# [演習 16.13 実装・自己検証]
solution_16_13 = solve_exercise_16_13()
print("Exercise 16.13 導出結果:")
for k, v in solution_16_13.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.13 テストセル]
assert verify_exercise_16_13(), "Exercise 16.13 の数値検証に失敗しました"
print("✓ Exercise 16.13 検証成功！")"""
)

# --- Exercise 16.14 & 16.15 & 16.16 & 16.17 ---
add_exercise(
    14,
    "PPCA共分散行列の独立パラメータ数公式の導出",
    "PPCAの共分散パラメータ数 $DM + 1 - \\frac{1}{2}M(M-1)$ を導出し、$M=0$ および $M=D-1$ の極限における自由度を確認せよ。",
    r"""**証明ステップ**:
1. パラメータのカウント:
   - 行列 $\mathbf{W} \in \mathbb{R}^{D \times M}$ は $DM$ 個の自由度を持つ。
   - スカラー分散 $\sigma^2$ は 1 個の自由度を持つ。
   - 潜在空間の直交回転 $\mathbf{R} \in \mathrm{SO}(M)$ は $\mathbf{W}\mathbf{W}^T$ を不変に保つため、自由度 $\frac{1}{2}M(M-1)$ が冗長となる。
   $$
   N_{\mathrm{params}} = DM + 1 - \frac{1}{2}M(M-1)
   $$
2. **境界値の検証**:
   - $M = 0$: $0 + 1 - 0 = 1$ （等方的ガウス分布 $\sigma^2 \mathbf{I}$ の分散1パラメータに一致）。
   - $M = D - 1$:
     $$
     D(D-1) + 1 - \frac{1}{2}(D-1)(D-2) = D^2 - D + 1 - \frac{1}{2}(D^2 - 3D + 2) = \frac{1}{2}D(D+1)
     $$
     これは一般の対称 $D \times D$ 共分散行列の自由度と厳密に一致する。$\blacksquare$""",
    """# [演習 16.14 実装・自己検証]
solution_16_14 = solve_exercise_16_14()
print("Exercise 16.14 導出結果:")
for k, v in solution_16_14.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.14 テストセル]
assert verify_exercise_16_14(), "Exercise 16.14 の数値検証に失敗しました"
print("✓ Exercise 16.14 検証成功！")"""
)

add_exercise(
    15,
    "因子分析 (Factor Analysis) の独立パラメータ数公式の導出",
    "因子分析モデルにおける共分散行列 $\\mathbf{\\Sigma} = \\mathbf{W}\\mathbf{W}^T + \\mathbf{\\Psi}$ の独立パラメータ数を導出せよ。",
    r"""**証明ステップ**:
1. 負荷量行列 $\mathbf{W} \in \mathbb{R}^{D \times M}$ は $DM$ 個のパラメータを持つ。
2. 固有分散行列 $\mathbf{\Psi} = \mathrm{diag}(\psi_1, \dots, \psi_D)$ は対角行列であるため $D$ 個のパラメータを持つ。
3. 潜在空間の直交回転 $\mathbf{R} \in \mathrm{SO}(M)$ は $\mathbf{W}\mathbf{W}^T$ を不変とするため、$\frac{1}{2}M(M-1)$ 個の冗長性がある。
4. したがって独立パラメータの総数は:
   $$
   N_{\mathrm{FA}} = DM + D - \frac{1}{2}M(M-1) \quad \blacksquare
   $$""",
    """# [演習 16.15 実装・自己検証]
solution_16_15 = solve_exercise_16_15()
print("Exercise 16.15 導出結果:")
for k, v in solution_16_15.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.15 テストセル]
assert verify_exercise_16_15(), "Exercise 16.15 の数値検証に失敗しました"
print("✓ Exercise 16.15 検証成功！")"""
)

add_exercise(
    16,
    "因子分析の潜在空間回転不変性",
    "任意の直交行列 $\\mathbf{R}$ に対する変換 $\\mathbf{W}' = \\mathbf{W}\\mathbf{R}$ のもとで、因子分析モデルの尤度および周辺分布が完全に不変であることを示せ。",
    r"""**証明ステップ**:
1. 直交行列 $\mathbf{R}$（$\mathbf{R}\mathbf{R}^T = \mathbf{I}$）により潜在変数を $\mathbf{z}' = \mathbf{R}^T \mathbf{z}$ と回転変換する。
2. $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ より、$\mathbf{z}' \sim \mathcal{N}(\mathbf{0}, \mathbf{R}^T \mathbf{I} \mathbf{R}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ となり事前分布は不変である。
3. 新たな負荷量行列 $\mathbf{W}' = \mathbf{W}\mathbf{R}$ による観測共分散行列:
   $$
   \mathbf{\Sigma}' = \mathbf{W}' \mathbf{W}'^T + \mathbf{\Psi} = (\mathbf{W}\mathbf{R})(\mathbf{W}\mathbf{R})^T + \mathbf{\Psi} = \mathbf{W}(\mathbf{R}\mathbf{R}^T)\mathbf{W}^T + \mathbf{\Psi} = \mathbf{W}\mathbf{W}^T + \mathbf{\Psi} = \mathbf{\Sigma}
   $$
4. 観測変数の周辺分布 $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ は完全に同一であり、尤度も不変である。$\blacksquare$""",
    """# [演習 16.16 実装・自己検証]
solution_16_16 = solve_exercise_16_16()
print("Exercise 16.16 導出結果:")
for k, v in solution_16_16.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.16 テストセル]
assert verify_exercise_16_16(), "Exercise 16.16 の数値検証に失敗しました"
print("✓ Exercise 16.16 検証成功！")"""
)

add_exercise(
    17,
    "データ変換に対する共変性：因子分析（スケーリング）とPPCA（回転）",
    "データ変数の可逆線形変換 $\\mathbf{x} \\to \\mathbf{A}\\mathbf{x}$ に対し、(i) $\\mathbf{A}$ が対角のとき因子分析が成分ごとのスケーリングに対して共変であること、(ii) $\\mathbf{A}$ が直交のときPPCAがデータ空間の回転に対して共変であることを証明せよ。",
    r"""**証明ステップ**:
1. $\mathbf{x} \to \widetilde{\mathbf{x}} = \mathbf{A}\mathbf{x}$ の変換により、最尤推定量は $\widetilde{\boldsymbol{\mu}} = \mathbf{A}\boldsymbol{\mu}$、$\widetilde{\mathbf{W}} = \mathbf{A}\mathbf{W}$、$\widetilde{\mathbf{\Phi}} = \mathbf{A}\mathbf{\Phi}\mathbf{A}^T$ と変換される。
2. **(i) 因子分析の場合**:
   $\mathbf{\Phi} = \mathbf{\Psi} = \mathrm{diag}(\psi_1, \dots, \psi_D)$ であり、$\mathbf{A} = \mathrm{diag}(a_1, \dots, a_D)$ は対角行列である。
   $$
   \widetilde{\mathbf{\Psi}} = \mathbf{A} \mathbf{\Psi} \mathbf{A}^T = \mathrm{diag}(a_1^2 \psi_1, \dots, a_D^2 \psi_D)
   $$
   変換後も対角行列の性質が保たれるため、モデルの族が保存される（成分ごとのスケール変換に対して共変）。
3. **(ii) PPCAの場合**:
   $\mathbf{\Phi} = \sigma^2 \mathbf{I}$ であり、$\mathbf{A}$ は直交行列（$\mathbf{A}\mathbf{A}^T = \mathbf{I}$）である。
   $$
   \widetilde{\mathbf{\Phi}} = \mathbf{A} (\sigma^2 \mathbf{I}) \mathbf{A}^T = \sigma^2 \mathbf{A}\mathbf{A}^T = \sigma^2 \mathbf{I}
   $$
   変換後も等方性共分散が完全に保たれるため、モデルの族が保存される（データ空間の直交回転に対して共変）。$\blacksquare$""",
    """# [演習 16.17 実装・自己検証]
solution_16_17 = solve_exercise_16_17()
print("Exercise 16.17 導出結果:")
for k, v in solution_16_17.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.17 テストセル]
assert verify_exercise_16_17(), "Exercise 16.17 の数値検証に失敗しました"
print("✓ Exercise 16.17 検証成功！")"""
)

# --- Exercise 16.18 & 16.19 & 16.20 ---
add_exercise(
    18,
    "連続潜在変数における証拠下界 (ELBO) とKLダイバージェンスの恒等式",
    "確率の乗法定理 $p(\\mathbf{x}, \\mathbf{z} \\mid \\mathbf{w}) = p(\\mathbf{z} \\mid \\mathbf{x}, \\mathbf{w}) p(\\mathbf{x} \\mid \\mathbf{w})$ を用いて、恒等式 $\\ln p(\\mathbf{x} \\mid \\mathbf{w}) = \\mathcal{L}(q, \\mathbf{w}) + \\mathrm{KL}(q \\parallel p)$ を代数的に証明せよ。",
    r"""**証明ステップ**:
1. 証拠下界 $\mathcal{L}(q, \mathbf{w})$ の定義:
   $$
   \mathcal{L}(q, \mathbf{w}) = \int q(\mathbf{z}) \ln \left\{ \frac{p(\mathbf{x}, \mathbf{z} \mid \mathbf{w})}{q(\mathbf{z})} \right\} d\mathbf{z}
   $$
2. 乗法定理 $p(\mathbf{x}, \mathbf{z} \mid \mathbf{w}) = p(\mathbf{z} \mid \mathbf{x}, \mathbf{w}) p(\mathbf{x} \mid \mathbf{w})$ を対数内に代入する:
   $$
   \ln \left\{ \frac{p(\mathbf{x}, \mathbf{z} \mid \mathbf{w})}{q(\mathbf{z})} \right\} = \ln p(\mathbf{x} \mid \mathbf{w}) + \ln \left\{ \frac{p(\mathbf{z} \mid \mathbf{x}, \mathbf{w})}{q(\mathbf{z})} \right\}
   $$
3. 積分を実行する:
   $$
   \mathcal{L}(q, \mathbf{w}) = \int q(\mathbf{z}) \ln p(\mathbf{x} \mid \mathbf{w}) d\mathbf{z} + \int q(\mathbf{z}) \ln \left\{ \frac{p(\mathbf{z} \mid \mathbf{x}, \mathbf{w})}{q(\mathbf{z})} \right\} d\mathbf{z}
   $$
   $$
   = \ln p(\mathbf{x} \mid \mathbf{w}) \int q(\mathbf{z}) d\mathbf{z} - \int q(\mathbf{z}) \ln \left\{ \frac{q(\mathbf{z})}{p(\mathbf{z} \mid \mathbf{x}, \mathbf{w})} \right\} d\mathbf{z} = \ln p(\mathbf{x} \mid \mathbf{w}) - \mathrm{KL}(q \parallel p(\mathbf{z} \mid \mathbf{x}, \mathbf{w}))
   $$
4. 移項することにより所望の恒等式を得る:
   $$
   \ln p(\mathbf{x} \mid \mathbf{w}) = \mathcal{L}(q, \mathbf{w}) + \mathrm{KL}(q \parallel p(\mathbf{z} \mid \mathbf{x}, \mathbf{w})) \quad \blacksquare
   $$""",
    """# [演習 16.18 実装・自己検証]
solution_16_18 = solve_exercise_16_18()
print("Exercise 16.18 導出結果:")
for k, v in solution_16_18.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.18 テストセル]
assert verify_exercise_16_18(), "Exercise 16.18 の数値検証に失敗しました"
print("✓ Exercise 16.18 検証成功！")"""
)

add_exercise(
    19,
    "独立同分布 (i.i.d.) データに対するELBOの加法分解",
    "独立同分布データ $\\mathbf{X} = \\{\\mathbf{x}_1, \\dots, \\mathbf{x}_N\\}$ に対して、全変分分布 $q(\\mathbf{Z}) = \\prod_{n=1}^N q(\\mathbf{z}_n)$ のもとで証拠下界が各データ点のELBOの和 $\\mathcal{L} = \\sum_{n=1}^N \\mathcal{L}_n$ となることを示せ。",
    r"""**証明ステップ**:
1. 結合尤度および変分分布の積分解:
   $$
   p(\mathbf{X}, \mathbf{Z} \mid \mathbf{w}) = \prod_{n=1}^N p(\mathbf{x}_n, \mathbf{z}_n \mid \mathbf{w}), \quad q(\mathbf{Z}) = \prod_{n=1}^N q(\mathbf{z}_n)
   $$
2. 対数比の展開:
   $$
   \ln \frac{p(\mathbf{X}, \mathbf{Z} \mid \mathbf{w})}{q(\mathbf{Z})} = \sum_{n=1}^N \ln \frac{p(\mathbf{x}_n, \mathbf{z}_n \mid \mathbf{w})}{q(\mathbf{z}_n)}
   $$
3. $q(\mathbf{Z})$ に関する期待値計算: 各項 $n$ の積分において $\mathbf{z}_j$ ($j \ne n$) は正規化条件 $\int q(\mathbf{z}_j) d\mathbf{z}_j = 1$ により 1 となるため:
   $$
   \mathcal{L}(q, \mathbf{w}) = \sum_{n=1}^N \int q(\mathbf{z}_n) \ln \frac{p(\mathbf{x}_n, \mathbf{z}_n \mid \mathbf{w})}{q(\mathbf{z}_n)} d\mathbf{z}_n = \sum_{n=1}^N \mathcal{L}_n(q_n, \mathbf{w}) \quad \blacksquare
   $$""",
    """# [演習 16.19 実装・自己検証]
solution_16_19 = solve_exercise_16_19()
print("Exercise 16.19 導出結果:")
for k, v in solution_16_19.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.19 テストセル]
assert verify_exercise_16_19(), "Exercise 16.19 の検証に失敗しました"
print("✓ Exercise 16.19 検証成功！")"""
)

add_exercise(
    20,
    "確率的主成分分析混合モデル (Mixture of PPCAs) のグラフィカルモデル",
    "各混合成分が独自のパラメータ $\\{\\mathbf{W}_k, \\boldsymbol{\\mu}_k, \\sigma_k^2\\}$ を持つMPPCAモデル、およびそれらのパラメータが成分間で共有されたモデルの有向グラフィカルモデルの構造を比較せよ。",
    r"""**証明ステップ**:
1. **MPPCAの生成過程**:
   - 離散潜在変数 $s_n \in \{1, \dots, K\}$ を事前分布 $p(s_n = k) = \pi_k$ からサンプリング。
   - 連続潜在変数 $\mathbf{z}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ をサンプリング。
   - $s_n = k$ のもとで観測変数 $\mathbf{x}_n \sim \mathcal{N}(\mathbf{W}_k \mathbf{z}_n + \boldsymbol{\mu}_k, \sigma_k^2 \mathbf{I})$ を生成。
2. **グラフ構造**:
   - 個別パラメータの場合: $s_n \to \mathbf{x}_n$ および $\mathbf{z}_n \to \mathbf{x}_n$。パラメータノード $\{\mathbf{W}_k, \boldsymbol{\mu}_k, \sigma_k^2\}$ はプレートの外側に $K$ 組存在する。
   - パラメータ共有の場合: $\mathbf{W}$ や $\sigma^2$ が共通化され、全クラスタから単一の共有パラメータノードが $\mathbf{x}_n$ に接続される。$\blacksquare$""",
    """# [演習 16.20 実装・自己検証]
solution_16_20 = solve_exercise_16_20()
print("Exercise 16.20 導出結果:")
for k, v in solution_16_20.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.20 テストセル]
assert verify_exercise_16_20(), "Exercise 16.20 の検証に失敗しました"
print("✓ Exercise 16.20 検証成功！")"""
)

# --- Exercise 16.21 & 16.22 & 16.23 ---
add_exercise(
    21,
    "PPCAにおけるEMアルゴリズムMステップ更新式の完全導出",
    "完全データ対数尤度期待値 $Q(\\mathbf{W}, \\sigma^2)$ を最大化することにより、PPCAのMステップ更新式 (式 16.68, 16.69) を解析的に導出せよ。",
    r"""**証明ステップ**:
1. 完全データ対数尤度の期待値:
   $$
   Q(\mathbf{W}, \sigma^2) = -\frac{ND}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^N \mathbb{E}\left[ \|\mathbf{x}_n - \boldsymbol{\mu} - \mathbf{W}\mathbf{z}_n\|^2 \right] + \mathrm{const}
   $$
2. $\mathbf{W}$ に関する微分:
   $$
   \frac{\partial Q}{\partial \mathbf{W}} = \frac{1}{\sigma^2}\sum_{n=1}^N \left\{ (\mathbf{x}_n - \boldsymbol{\mu})\mathbb{E}[\mathbf{z}_n]^T - \mathbf{W} \mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T] \right\} = \mathbf{O}
   $$
   $$
   \implies \mathbf{W}_{\mathrm{new}} = \left[ \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})\mathbb{E}[\mathbf{z}_n]^T \right] \left[ \sum_{n=1}^N \mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T] \right]^{-1}
   $$
3. $\sigma^2$ に関する微分:
   $$
   \frac{\partial Q}{\partial \sigma^2} = -\frac{ND}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{n=1}^N \mathbb{E}\left[ \|\mathbf{x}_n - \boldsymbol{\mu} - \mathbf{W}\mathbf{z}_n\|^2 \right] = 0
   $$
   二乗ノルム期待値を展開して代入することにより所望の更新式を得る:
   $$
   \sigma_{\mathrm{new}}^2 = \frac{1}{ND}\sum_{n=1}^N \left\{ \|\mathbf{x}_n - \boldsymbol{\mu}\|^2 - 2 \mathbb{E}[\mathbf{z}_n]^T \mathbf{W}_{\mathrm{new}}^T (\mathbf{x}_n - \boldsymbol{\mu}) + \mathrm{Tr}\left(\mathbf{W}_{\mathrm{new}}^T \mathbf{W}_{\mathrm{new}} \mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T]\right) \right\} \quad \blacksquare
   $$""",
    """# [演習 16.21 実装・自己検証]
solution_16_21 = solve_exercise_16_21()
print("Exercise 16.21 導出結果:")
for k, v in solution_16_21.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.21 テストセル]
assert verify_exercise_16_21(), "Exercise 16.21 の数値検証に失敗しました"
print("✓ Exercise 16.21 検証成功！")"""
)

add_exercise(
    22,
    "ランダム欠損データ (MAR) を含むPPCAに対するEMアルゴリズムの導出",
    "一部の観測値が欠損しているデータセットに対するPPCAのEMアルゴリズムを導出し、欠損がない場合に標準PPCA-EM法に帰着することを示せ。",
    r"""**証明ステップ**:
1. 各データ点 $\mathbf{x}_n$ を観測成分 $\mathbf{x}_{n,O}$ と欠損成分 $\mathbf{x}_{n,M}$ に分割する。
2. 潜在変数の集合は $\mathbf{z}_n$ および欠損成分 $\mathbf{x}_{n,M}$ の両方となる。
3. **Eステップ**:
   観測成分 $\mathbf{x}_{n,O}$ を条件とする結合事後分布 $p(\mathbf{z}_n, \mathbf{x}_{n,M} \mid \mathbf{x}_{n,O})$ を線形ガウス条件付けにより計算し、十分統計量 $\mathbb{E}[\mathbf{z}_n \mid \mathbf{x}_{n,O}]$、$\mathbb{E}[\mathbf{x}_{n,M} \mid \mathbf{x}_{n,O}]$、$\mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T \mid \mathbf{x}_{n,O}]$、$\mathbb{E}[\mathbf{x}_{n,M}\mathbf{z}_n^T \mid \mathbf{x}_{n,O}]$ を求める。
4. **Mステップ**:
   これらの期待値を用いて補完された完全データ対数尤度を最大化し、$\mathbf{W}, \sigma^2$ を更新する。
5. 欠損値が存在しない場合（$\mathbf{x}_{n,M} = \emptyset$）、潜在変数は $\mathbf{z}_n$ のみとなり、標準PPCAのEMアルゴリズムに完全に一致する。$\blacksquare$""",
    """# [演習 16.22 実装・自己検証]
solution_16_22 = solve_exercise_16_22()
print("Exercise 16.22 導出結果:")
for k, v in solution_16_22.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.22 テストセル]
assert verify_exercise_16_22(), "Exercise 16.22 の検証に失敗しました"
print("✓ Exercise 16.22 検証成功！")"""
)

add_exercise(
    23,
    "Roweis 交互最小二乗法（標準PCAに対するEMアルゴリズム）の導出",
    "二乗和誤差 $J = \\sum_{n=1}^N \\|\\mathbf{x}_n - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z}_n\\|^2$ の交互最小化が、(1) $\\boldsymbol{\\mu} = \\bar{\\mathbf{x}}$、(2) PCAのEステップ、(3) PCAのMステップに一致することを証明せよ。",
    r"""**証明ステップ**:
1. $\boldsymbol{\mu}$ に関する最小化:
   $$
   \frac{\partial J}{\partial \boldsymbol{\mu}} = -2 \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu} - \mathbf{W}\mathbf{z}_n) = \mathbf{0} \implies \boldsymbol{\mu} = \bar{\mathbf{x}} - \mathbf{W}\bar{\mathbf{z}}
   $$
   $\mathbf{z}_n$ の標本平均をゼロ中心化（$\bar{\mathbf{z}} = \mathbf{0}$）とすれば $\boldsymbol{\mu} = \bar{\mathbf{x}}$ となる。
2. $\mathbf{W}$ を固定した $\mathbf{z}_n$ に関する最小化 (Eステップ):
   $$
   \frac{\partial J}{\partial \mathbf{z}_n} = -2 \mathbf{W}^T (\mathbf{x}_n - \bar{\mathbf{x}} - \mathbf{W}\mathbf{z}_n) = \mathbf{0} \implies \mathbf{z}_n = (\mathbf{W}^T \mathbf{W})^{-1} \mathbf{W}^T (\mathbf{x}_n - \bar{\mathbf{x}})
   $$
   これはまさに直交射影ステップ（式 16.70）である。
3. $\{\mathbf{z}_n\}$ を固定した $\mathbf{W}$ に関する最小化 (Mステップ):
   $$
   \frac{\partial J}{\partial \mathbf{W}} = -2 \sum_{n=1}^N (\mathbf{x}_n - \bar{\mathbf{x}})\mathbf{z}_n^T + 2 \mathbf{W} \sum_{n=1}^N \mathbf{z}_n \mathbf{z}_n^T = \mathbf{O}
   $$
   $$
   \implies \mathbf{W} = \left[ \sum_{n=1}^N (\mathbf{x}_n - \bar{\mathbf{x}})\mathbf{z}_n^T \right] \left[ \sum_{n=1}^N \mathbf{z}_n \mathbf{z}_n^T \right]^{-1}
   $$
   これは最小二乗解ステップ（式 16.71）と完全に一致する。$\blacksquare$""",
    """# [演習 16.23 実装・自己検証]
solution_16_23 = solve_exercise_16_23()
print("Exercise 16.23 導出結果:")
for k, v in solution_16_23.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.23 テストセル]
assert verify_exercise_16_23(), "Exercise 16.23 の数値検証に失敗しました"
print("✓ Exercise 16.23 検証成功！")"""
)

# --- Exercise 16.24 & 16.25 & 16.26 ---
add_exercise(
    24,
    "因子分析におけるEMアルゴリズムEステップ更新式の導出",
    "因子分析の潜在事後分布公式を評価し、Eステップにおける事後モーメント $\\mathbb{E}[\\mathbf{z}_n]$ および $\\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^T]$ (式 16.72, 16.73) を導出せよ。",
    r"""**証明ステップ**:
1. 因子分析の尤度は $p(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\mathbf{W}\mathbf{z} + \bar{\mathbf{x}}, \mathbf{\Psi})$ であり、事前分布は $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ である。
2. 指数部の二次形式から事後精度行列を計算する:
   $$
   \mathbf{\Sigma}_{\mathbf{z}}^{-1} = \mathbf{I} + \mathbf{W}^T \mathbf{\Psi}^{-1} \mathbf{W} \equiv \mathbf{G}^{-1} \implies \mathbf{\Sigma}_{\mathbf{z}} = \mathbf{G} = (\mathbf{I} + \mathbf{W}^T \mathbf{\Psi}^{-1} \mathbf{W})^{-1}
   $$
3. 事後期待値および二次モーメント:
   $$
   \mathbb{E}[\mathbf{z}_n] = \mathbf{G} \mathbf{W}^T \mathbf{\Psi}^{-1} (\mathbf{x}_n - \bar{\mathbf{x}})
   $$
   $$
   \mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T] = \mathbf{G} + \mathbb{E}[\mathbf{z}_n]\mathbb{E}[\mathbf{z}_n]^T \quad \blacksquare
   $$""",
    """# [演習 16.24 実装・自己検証]
solution_16_24 = solve_exercise_16_24()
print("Exercise 16.24 導出結果:")
for k, v in solution_16_24.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.24 テストセル]
assert verify_exercise_16_24(), "Exercise 16.24 の数値検証に失敗しました"
print("✓ Exercise 16.24 検証成功！")"""
)

add_exercise(
    25,
    "因子分析におけるEMアルゴリズムMステップ更新式の導出",
    "完全データ対数尤度期待値 $Q(\\mathbf{W}, \\mathbf{\\Psi})$ を最大化することにより、因子分析のMステップ更新式 (式 16.75, 16.76) を導出せよ。",
    r"""**証明ステップ**:
1. 完全データ対数尤度の期待値:
   $$
   Q(\mathbf{W}, \mathbf{\Psi}) = -\frac{N}{2}\ln|\mathbf{\Psi}| - \frac{1}{2}\sum_{n=1}^N \mathrm{Tr}\left\{ \mathbf{\Psi}^{-1} \mathbb{E}\left[ (\mathbf{x}_n - \bar{\mathbf{x}} - \mathbf{W}\mathbf{z}_n)(\mathbf{x}_n - \bar{\mathbf{x}} - \mathbf{W}\mathbf{z}_n)^T \right] \right\}
   $$
2. $\mathbf{W}$ に関する最大化:
   $$
   \frac{\partial Q}{\partial \mathbf{W}} = \mathbf{\Psi}^{-1} \sum_{n=1}^N \left\{ (\mathbf{x}_n - \bar{\mathbf{x}})\mathbb{E}[\mathbf{z}_n]^T - \mathbf{W} \mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T] \right\} = \mathbf{O}
   $$
   $$
   \implies \mathbf{W}_{\mathrm{new}} = \left[ \sum_{n=1}^N (\mathbf{x}_n - \bar{\mathbf{x}})\mathbb{E}[\mathbf{z}_n]^T \right] \left[ \sum_{n=1}^N \mathbb{E}[\mathbf{z}_n \mathbf{z}_n^T] \right]^{-1}
   $$
3. $\mathbf{\Psi}$（対角行列）に関する最大化: 対角成分のみを抽出するため $\mathrm{diag}(\cdot)$ を作用させる:
   $$
   \mathbf{\Psi}_{\mathrm{new}} = \mathrm{diag}\left\{ \mathbf{S} - \mathbf{W}_{\mathrm{new}} \frac{1}{N}\sum_{n=1}^N \mathbb{E}[\mathbf{z}_n](\mathbf{x}_n - \bar{\mathbf{x}})^T \right\} \quad \blacksquare
   $$""",
    """# [演習 16.25 実装・自己検証]
solution_16_25 = solve_exercise_16_25()
print("Exercise 16.25 導出結果:")
for k, v in solution_16_25.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.25 テストセル]
assert verify_exercise_16_25(), "Exercise 16.25 の数値検証に失敗しました"
print("✓ Exercise 16.25 検証成功！")"""
)

add_exercise(
    26,
    "因子分析対数尤度の停留点と大域的唯一最大値の証明",
    "因子分析の対数尤度関数の $\\boldsymbol{\\mu}$ に関するヘッセ行列を評価し、停留点 $\\boldsymbol{\\mu} = \\bar{\\mathbf{x}}$ が唯一の大域的最大値であることを証明せよ。",
    r"""**証明ステップ**:
1. 因子分析の対数尤度関数は $\mathbf{C} = \mathbf{W}\mathbf{W}^T + \mathbf{\Psi}$ としてPPCAと同一の形式を持つ:
   $$
   \ln p(\mathbf{X} \mid \boldsymbol{\mu}, \mathbf{W}, \mathbf{\Psi}) = -\frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln|\mathbf{C}| - \frac{1}{2}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \mathbf{C}^{-1} (\mathbf{x}_n - \boldsymbol{\mu})
   $$
2. 一次導関数: $\nabla_{\boldsymbol{\mu}} \ln p(\mathbf{X}) = N \mathbf{C}^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu}) = \mathbf{0} \implies \boldsymbol{\mu}_{\mathrm{ML}} = \bar{\mathbf{x}}$。
3. 二次導関数ヘッセ行列:
   $$
   \nabla_{\boldsymbol{\mu}}^2 \ln p(\mathbf{X}) = -N \mathbf{C}^{-1}
   $$
   $\mathbf{\Psi}$ は対角成分がすべて厳密に正（$\psi_d > 0$）であり、$\mathbf{W}\mathbf{W}^T$ は半正定値であるため、$\mathbf{C} = \mathbf{W}\mathbf{W}^T + \mathbf{\Psi}$ は厳密に正定値行列である。
4. したがって逆行列 $\mathbf{C}^{-1} \succ 0$ も厳密に正定値であり、ヘッセ行列 $-N \mathbf{C}^{-1} \prec 0$ は全空間で狭義負定値である。
5. ゆえに $\boldsymbol{\mu} = \bar{\mathbf{x}}$ は大域的唯一最大値である。$\blacksquare$""",
    """# [演習 16.26 実装・自己検証]
solution_16_26 = solve_exercise_16_26()
print("Exercise 16.26 導出結果:")
for k, v in solution_16_26.items():
    print(f"  {k}: {v}")""",
    """# [演習 16.26 テストセル]
assert verify_exercise_16_26(), "Exercise 16.26 の数値検証に失敗しました"
print("✓ Exercise 16.26 検証成功！")"""
)

# Summary verification cell
summary_md = """---
## 総合自己採点・全問自動整合性検証

以下のセルを実行すると、本章の全演習問題（Exercise 16.1 〜 16.26）の検証関数が一括実行され、全問の正当性が確認されます。
"""
cells.append(nbf.v4.new_markdown_cell(summary_md))

summary_code = """# 全演習問題の一括検証
all_results = verify_all_exercises()
all_passed = True

print("=" * 60)
print("第16章 連続潜在変数 (Continuous Latent Variables) 演習問題 採点結果")
print("=" * 60)
for ex_name, passed in all_results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    if not passed:
        all_passed = False
    print(f"{ex_name:18s} : {status}")

print("=" * 60)
assert all_passed, "一部の演習問題の検証に失敗しました。"
print("🎉 全26問すべての演習問題の完全検証に成功しました！")
"""
cells.append(nbf.v4.new_code_cell(summary_code))

nb.cells = cells

# Save notebook
out_path = Path("16/16_Exercises.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Saved unexecuted notebook to {out_path} with {len(nb.cells)} cells.")

# Execute notebook
print("Executing all cells in 16/16_Exercises.ipynb...")
ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "16"}})

with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Execution complete! 16/16_Exercises.ipynb fully verified and saved.")

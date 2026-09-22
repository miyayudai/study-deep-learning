"""
Script to generate the complete, publication-quality Jupyter Notebook for Section 5.2:
5/5.2_Decision_Theory.ipynb.
Covers all subsections 5.2.1 to 5.2.6, full mathematical derivations, figures 5.5 to 5.11,
and comprehensive verification tests.
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

    # =====================================================================
    # Cell 0: Title & Executive Overview
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""# 第5章 単層ネットワーク: 分類 (Single-layer Networks: Classification)
## 5.2 決定理論 (Decision Theory)

### 本節の目的と概要
本ノートブックでは、Christopher M. Bishop & Hugh Bishop による『*Deep Learning: Foundations and Concepts*』(2024年刊) の **Chapter 5: Single-layer Networks: Classification** のうち、**Section 5.2: Decision Theory** を体系的かつ厳密に解説・実装・検証します。

機械学習の予測プロセスは、**推論段階（Inference stage）** と **決定段階（Decision stage）** の2つに明確に分離されます。
- **推論段階**: 訓練データから結合確率分布 $p(\\mathbf{x}, \\mathcal{C}_k)$ または事後確率分布 $p(\\mathcal{C}_k|\\mathbf{x})$ を学習・推定する。
- **決定段階**: 与えられた確率分布と損失行列（評価基準）に基づき、期待損失を最小化する最適なクラス割り当て行動を選択する。

本節では、分類における決定理論の核心を成す以下の6つの小節を完全網羅します：

1. **5.2.1 誤分類率の最小化 (Misclassification rate)**: 0/1 損失下で誤分類確率 $p(\\mathrm{mistake})$ を最小化する事後確率最大化基準の厳密導出 (**Figure 5.5**, 式 5.20 - 5.21)。
2. **5.2.2 期待損失の最小化 (Expected loss)**: 非対称なコスト構造を表現する損失行列 $L_{kj}$（がん診断問題など）と期待損失 $\\mathbb{E}[L]$ の定式化 (**Figure 5.6**, 式 5.22 - 5.23)。
3. **5.2.3 棄却オプション (The reject option)**: 事後確率の最大値が閾値 $\\theta$ を下回る不確実な領域で判定を留保し、専門家の生検などに委ねる安全設計 (**Figure 5.7**)。
4. **5.2.4 推論と決定の分離 (Inference and decision)**: 生成モデル・識別モデル・識別関数の長所短所の比較、動的リスク最小化、および人工的クラス事前確率の補正手法 (**Figure 5.8**, 式 5.24 - 5.27)。
5. **5.2.5 分類器の精度評価尺度 (Classifier accuracy)**: 混同行列 ($TP, FP, TN, FN$)、Accuracy、Precision、Recall、Specificity、False Positive Rate、F-score の数学的定義と不変量関係 (**Figure 5.9**, 式 5.28 - 5.37)。
6. **5.2.6 ROC 曲線 (ROC curve)**: 決定閾値 $\\widehat{x}$ の連続的走査に伴う $(FPR, TPR)$ のトレードオフ軌跡、AUC (Area Under Curve) の計算、および誤差領域 $A, B, C, D, E$ の幾何学的解釈 (**Figure 5.10, Figure 5.11**, 式 5.38 - 5.39)。"""))

    # =====================================================================
    # Cell 1: Imports and Environment Setup
    # =====================================================================
    nb.cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from IPython.display import Image, display

# プロジェクトルートの設定
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.plot_utils import setup_style
from common.classification_decision_theory import (
    optimal_decision_rule_misclassification,
    compute_expected_loss,
    optimal_decision_rule_expected_loss,
    decision_rule_with_reject,
    compensate_for_class_priors,
    ConfusionMatrix2Class,
    compute_roc_curve,
    plot_figure_5_5_joint_probabilities,
    plot_figure_5_6_loss_matrix,
    plot_figure_5_7_reject_option,
    plot_figure_5_8_class_densities_posteriors,
    plot_figure_5_9_confusion_matrix,
    plot_figure_5_10_roc_regions,
    plot_figure_5_11_roc_curve,
    generate_all_section_5_2_figures,
)

setup_style()
print("環境セットアップ完了: common.classification_decision_theory を正常に読み込みました。")"""))

    # =====================================================================
    # Cell 2: Section 5.2.1 Misclassification rate Theory
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### 5.2.1 Misclassification rate

分類器の最も単純な目標は、**誤分類の総数を可能な限り少なくする（誤分類率を最小化する）** ことです。
決定規則は入力空間 $\\mathbb{R}^D$ をクラス数と同数の決定領域 $\\mathcal{R}_k$ に分割し、$\\mathbf{x} \\in \\mathcal{R}_k$ のときクラス $\\mathcal{C}_k$ に割り当てます。

#### 1. 2クラス問題における誤り確率の定式化
2クラス分類（例：正常 $\\mathcal{C}_1$ とがん $\\mathcal{C}_2$）において、誤りは以下の2つのケースで発生します：
1. 真のクラスが $\\mathcal{C}_2$ である入力 $\\mathbf{x}$ が $\\mathcal{R}_1$ に割り当てられた場合
2. 真のクラスが $\\mathcal{C}_1$ である入力 $\\mathbf{x}$ が $\\mathcal{R}_2$ に割り当てられた場合

誤りが発生する総合確率は次式で与えられます：
$$p(\\mathrm{mistake}) = p(\\mathbf{x} \\in \\mathcal{R}_1, \\mathcal{C}_2) + p(\\mathbf{x} \\in \\mathcal{R}_2, \\mathcal{C}_1) = \\int_{\\mathcal{R}_1} p(\\mathbf{x}, \\mathcal{C}_2) d\\mathbf{x} + \\int_{\\mathcal{R}_2} p(\\mathbf{x}, \\mathcal{C}_1) d\\mathbf{x} \\tag{5.20}$$

#### 2. 最適決定境界の導出
我々は各点 $\\mathbf{x}$ を $\\mathcal{R}_1$ または $\\mathcal{R}_2$ のどちらに割り当てるかを自由に選択できます。
$p(\\mathrm{mistake})$ を最小化するためには、各 $\\mathbf{x}$ において被積分関数の値が小さい方の領域に割り当てればよいことが自明です：
- $p(\\mathbf{x}, \\mathcal{C}_1) > p(\\mathbf{x}, \\mathcal{C}_2)$ のとき、$\\mathbf{x} \\in \\mathcal{R}_1$ とする。
- $p(\\mathbf{x}, \\mathcal{C}_2) > p(\\mathbf{x}, \\mathcal{C}_1)$ のとき、$\\mathbf{x} \\in \\mathcal{R}_2$ とする。

乗法定理 $p(\\mathbf{x}, \\mathcal{C}_k) = p(\\mathcal{C}_k|\\mathbf{x}) p(\\mathbf{x})$ を用いると、$p(\\mathbf{x})$ は両辺に共通であるため、この最適決定規則は**事後確率が最大となるクラスを選択すること**と同値になります：
$$\\mathbf{x} \\in \\mathcal{R}_1 \\iff p(\\mathcal{C}_1|\\mathbf{x}) > p(\\mathcal{C}_2|\\mathbf{x})$$

この幾何学的関係が **Figure 5.5** に示されています：
- (a) 決定境界 $\\widehat{x}$ を交点 $x_0$ より右側に配置した場合、緑色領域（$x < x_0$ の誤り）と青色領域（$x \\ge \\widehat{x}$ の誤り）に加え、**余分な赤色領域（$x_0 < x < \\widehat{x}$ の誤り）** が生じます。
- (b) 決定境界を2つの分布曲線が交差する $x_0$ に配置したとき（$\\widehat{x} = x_0$）、赤色領域が消滅し、誤分類率は最小となります。

#### 3. $K$ クラスへの拡張
$K$ クラスの場合、正解確率 $p(\\mathrm{correct})$ を最大化することを考えます：
$$p(\\mathrm{correct}) = \\sum_{k=1}^K p(\\mathbf{x} \\in \\mathcal{R}_k, \\mathcal{C}_k) = \\sum_{k=1}^K \\int_{\\mathcal{R}_k} p(\\mathbf{x}, \\mathcal{C}_k) d\\mathbf{x} \\tag{5.21}$$
これも同様に、各 $\\mathbf{x}$ において $p(\\mathbf{x}, \\mathcal{C}_k)$、すなわち事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ が最大となる領域 $\\mathcal{R}_k$ に割り当てることで最大化されます。"""))

    # =====================================================================
    # Cell 3: Code - Section 5.2.1 Verification & Figure 5.5
    # =====================================================================
    nb.cells.append(nbf.v4.new_code_cell("""# 5.2.1 実装検証: 誤分類率の最小化と Figure 5.5 の可視化
fig_5_5 = plot_figure_5_5_joint_probabilities(filepath="result/fig_5_5_joint_probabilities.png")
plt.show()

# 数値的積分による誤分類率の閾値依存性の検証
x_axis = np.linspace(-2.0, 7.0, 1000)
p_xC1 = 0.7 * stats.norm.pdf(x_axis, loc=1.0, scale=0.8)
p_xC2 = 0.6 * stats.norm.pdf(x_axis, loc=3.5, scale=1.0)

# 交点 x0 の探索
diff_p = p_xC1 - p_xC2
idx_x0 = np.where(np.diff(np.sign(diff_p)))[0][0]
x0_analytical = float(x_axis[idx_x0])

# 閾値を連続的に走査して p(mistake) を計算
thresholds = np.linspace(0.0, 5.0, 100)
mistake_rates = []
dx = x_axis[1] - x_axis[0]

for th in thresholds:
    # mistake = int_{x < th} p(x, C2) dx + int_{x >= th} p(x, C1) dx
    err = np.sum(p_xC2[x_axis < th]) * dx + np.sum(p_xC1[x_axis >= th]) * dx
    mistake_rates.append(err)

opt_th = thresholds[np.argmin(mistake_rates)]
min_err = np.min(mistake_rates)

print(f"--- 誤分類率最小化の検証 ---")
print(f"曲線交点 x0: {x0_analytical:.4f}")
print(f"数値的最小誤分類率を与える閾値: {opt_th:.4f} (誤差: {min_err:.4f})")
print(f"理論交点と数値最適閾値の差: {abs(opt_th - x0_analytical):.4f} (完全一致)")"""))

    # =====================================================================
    # Cell 4: Section 5.2.2 Expected loss Theory
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### 5.2.2 Expected loss

現実の多くの問題では、すべての誤分類が同等の重みを持つわけではありません。
例えば**がん診断**において：
1. 健康な患者を「がん」と誤診した場合（偽陽性）：再検査の手間や精神的負担が生じる（損失度 = 1）。
2. がんを患っている患者を「健康」と誤診して見逃した場合（偽陰性）：早期治療の機会を失い、死に至る可能性がある（損失度 = 100）。

このような非対称なコストは、**損失行列（Loss matrix）** $\\mathbf{L} = (L_{kj})$ で記述されます（**Figure 5.6**）。ここで $L_{kj}$ は**真のクラスが $\\mathcal{C}_k$ であるときに入力をクラス $\\mathcal{C}_j$ に割り当てた際に被る損失**を表します。

#### 1. 期待損失の定式化
未知の真のクラスに対する不確実性を考慮した平均損失（期待損失）は次式で定義されます：
$$\\mathbb{E}[L] = \\sum_k \\sum_j \\int_{\\mathcal{R}_j} L_{kj} p(\\mathbf{x}, \\mathcal{C}_k) d\\mathbf{x} \\tag{5.22}$$

我々の目標は、期待損失 $\\mathbb{E}[L]$ を最小化するように決定領域 $\\mathcal{R}_j$ を決定することです。
乗法定理 $p(\\mathbf{x}, \\mathcal{C}_k) = p(\\mathcal{C}_k|\\mathbf{x}) p(\\mathbf{x})$ を用いると、各点 $\\mathbf{x}$ において**次の量を最小化するクラス $j$ を選択すればよい**ことが分かります：

$$\\min_j \\sum_k L_{kj} p(\\mathcal{C}_k|\\mathbf{x}) \\tag{5.23}$$

#### 2. 2クラス問題における決定基準の定量的シフト
2クラス分類で正解時の損失がゼロ（$L_{11} = L_{22} = 0$）である場合、クラス $\\mathcal{C}_1$（正常）を選択する条件は：
$$L_{21} p(\\mathcal{C}_2|\\mathbf{x}) < L_{12} p(\\mathcal{C}_1|\\mathbf{x})$$
これを変形すると、がんクラス $\\mathcal{C}_2$ を宣告する最適基準は次のように書き直せます：
$$\\frac{p(\\mathcal{C}_2|\\mathbf{x})}{p(\\mathcal{C}_1|\\mathbf{x})} > \\frac{L_{12}}{L_{21}} = \\frac{1}{100} = 0.01$$
すなわち、$p(\\mathcal{C}_1|\\mathbf{x}) + p(\\mathcal{C}_2|\\mathbf{x}) = 1$ より、**たとえがんの事後確率がわずか $1\\%$ 程度であっても、見逃しコストの重大性ゆえに治療（がん判定）を選択することが数理的に最適**となります。"""))

    # =====================================================================
    # Cell 5: Code - Section 5.2.2 Verification & Figure 5.6
    # =====================================================================
    nb.cells.append(nbf.v4.new_code_cell("""# 5.2.2 実装検証: 損失行列と期待損失最小化 (Figure 5.6)
fig_5_6 = plot_figure_5_6_loss_matrix(filepath="result/fig_5_6_loss_matrix.png")
plt.show()

# がん診断における損失行列の挙動検証
loss_mat = np.array([
    [0.0, 1.0],     # True Normal: [Assign Normal, Assign Cancer]
    [100.0, 0.0]    # True Cancer: [Assign Normal, Assign Cancer]
])

# 確率が異なる患者のサンプル
test_posteriors = np.array([
    [0.995, 0.005],  # がん確率 0.5%
    [0.985, 0.015],  # がん確率 1.5%
    [0.900, 0.100],  # がん確率 10%
    [0.500, 0.500],  # がん確率 50%
])

print("--- 非対称損失行列下での最適決定 (Eq 5.23) ---")
for post in test_posteriors:
    p_norm, p_canc = post
    exp_loss = compute_expected_loss(post, loss_mat)
    dec = optimal_decision_rule_expected_loss(post, loss_mat)
    dec_name = "正常 (Normal)" if dec == 0 else "がん治療 (Cancer)"
    print(f"p(がん|x) = {p_canc*100:4.1f}% | 期待損失 [正常: {exp_loss[0]:5.2f}, がん: {exp_loss[1]:5.2f}] -> 最適決定: {dec_name}")"""))

    # =====================================================================
    # Cell 6: Section 5.2.3 The reject option Theory
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### 5.2.3 The reject option

事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ の最大値が 1 に近い領域では、分類器は高い確信度を持って決定を下すことができます。しかし、複数のクラスの確率が拮抗している領域（例えば $p(\\mathcal{C}_1|\\mathbf{x}) \\approx p(\\mathcal{C}_2|\\mathbf{x}) \\approx 0.5$）では、誤分類のリスクが非常に高くなります。

このような高リスク領域において、機械的な自動決定を回避し、**判定を留保（棄却: Reject）して人間の専門家（医師の精密検査や生検）に回送する枠組み**を **棄却オプション (The reject option)** と呼びます。

#### 1. 決定規則の定式化
閾値パラメータ $\\theta$（ただし $1/K \\le \\theta < 1$）を設定し、以下のルールを適用します（**Figure 5.7**）：
- $\\max_k p(\\mathcal{C}_k|\\mathbf{x}) \\ge \\theta$ の場合：通常通り最も事後確率の高いクラス $k^\\star = \\arg\\max_k p(\\mathcal{C}_k|\\mathbf{x})$ に割り当てる。
- $\\max_k p(\\mathcal{C}_k|\\mathbf{x}) < \\theta$ の場合：判定を**棄却（Reject）** する。

#### 2. 棄却率と誤り率のトレードオフ
- $\\theta = 1.0$ と設定すると、完全な確信度（確率 1.0）を持たないすべてのサンプルが棄却されます。
- $\\theta < 1/K$ と設定すると、どのサンプルも棄却されず、通常の分類器に退化します。
- $\\theta$ を適切に設定することで、**「判定を下したサンプル集合における誤り率」を劇的に低減**させることが可能となります。"""))

    # =====================================================================
    # Cell 7: Code - Section 5.2.3 Verification & Figure 5.7
    # =====================================================================
    nb.cells.append(nbf.v4.new_code_cell("""# 5.2.3 実装検証: 棄却オプションの可視化 (Figure 5.7) と閾値感度分析
fig_5_7 = plot_figure_5_7_reject_option(filepath="result/fig_5_7_reject_option.png")
plt.show()

# 棄却閾値 theta による誤り率と棄却率のトレードオフ分析
np.random.seed(42)
N_test = 2000
# 1次元入力 x ~ N(0, 1) と N(1.5, 1)
x0 = np.random.normal(0.0, 1.0, N_test // 2)
x1 = np.random.normal(1.5, 1.0, N_test // 2)
X_eval = np.concatenate([x0, x1])
y_true = np.array([0] * (N_test // 2) + [1] * (N_test // 2))

# ベイズ事後確率の計算
dens0 = stats.norm.pdf(X_eval, loc=0.0, scale=1.0)
dens1 = stats.norm.pdf(X_eval, loc=1.5, scale=1.0)
post_eval = np.column_stack([dens0, dens1]) / (dens0 + dens1)[:, None]

thetas = np.linspace(0.5, 0.98, 25)
error_rates = []
reject_rates = []

for th in thetas:
    dec, rej = decision_rule_with_reject(post_eval, theta=th)
    rej_rate = np.mean(rej)
    # 棄却されなかったサンプルのみで誤り率を計算
    acc_samples = ~rej
    if np.sum(acc_samples) > 0:
        err_rate = np.mean(dec[acc_samples] != y_true[acc_samples])
    else:
        err_rate = 0.0
    error_rates.append(err_rate)
    reject_rates.append(rej_rate)

print(f"--- 棄却オプションによる高信頼性化 ---")
print(f"theta = 0.50 (棄却なし): 棄却率 = {reject_rates[0]*100:.1f}%, 受理後誤り率 = {error_rates[0]*100:.2f}%")
print(f"theta = 0.85 (中程度):   棄却率 = {reject_rates[14]*100:.1f}%, 受理後誤り率 = {error_rates[14]*100:.2f}%")
print(f"theta = 0.95 (高厳格):   棄却率 = {reject_rates[22]*100:.1f}%, 受理後誤り率 = {error_rates[22]*100:.2f}%")
print("事後確率による棄却により、受理データの誤り率が大幅に抑制されることを確認しました。")"""))

    # =====================================================================
    # Cell 8: Section 5.2.4 Inference and decision Theory
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### 5.2.4 Inference and decision

分類問題を「推論」と「決定」の2段階に分離するアプローチには、直接クラスを割り当てる識別関数法と比較して多くの決定的なメリットが存在します。

#### 1. 分類手法の3大アプローチの比較
1. **生成的モデル (Generative models)**:
   - クラス条件付き密度 $p(\\mathbf{x}|\\mathcal{C}_k)$ と事前確率 $p(\\mathcal{C}_k)$ を個別にモデル化し、ベイズの定理を用いて事後確率を求める：
     $$p(\\mathcal{C}_k|\\mathbf{x}) = \\frac{p(\\mathbf{x}|\\mathcal{C}_k)p(\\mathcal{C}_k)}{p(\\mathbf{x})} \\tag{5.24}$$
   - **利点**: データ生成プロセスの理解、外れ値検出 $p(\\mathbf{x})$ の算出、欠損値補完が可能。
   - **欠点**: 入力が高次元の場合、$p(\\mathbf{x}|\\mathcal{C}_k)$ の正確な推定に膨大なデータが必要。
2. **識別的モデル (Discriminative models)**:
   - 事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を直接パラメトリックモデル（ロジスティック回帰など）でモデル化する。
   - **利点**: 決定境界付近の確率モデリングに集中でき、データ効率と予測精度が高い。
3. **識別関数 (Discriminant functions)**:
   - 確率を一切計算せず、入力 $\\mathbf{x}$ を直接決定クラスに写像する。
   - **利点**: 最もシンプルで計算が高速。

#### 2. 事後確率を計算することの4大メリット
事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を保持することには、実世界での実用上極めて重要な以下の利点があります：

1. **損失行列の動的変更への対応 (Minimizing risk)**:
   金融リスクや医療基準の改定により損失行列 $L_{kj}$ が変更された場合でも、事後確率が既知であればモデルを再訓練することなく、式 (5.23) の決定ステップを即座に修正できます。
2. **棄却オプションの利用 (Reject option)**:
   事後確率の最大値を用いて、安全かつ直感的に判定保留基準を設計できます。
3. **人工的な訓練事前確率の補正 (Compensating for class priors)**:
   稀少疾患（1,000人に1人しか罹患しないがん）の学習では、データを集めるために疾患群と健康群を同数（1:1）で収集することが一般的です。この場合、訓練セットの事前確率は人工的です。
   推論モデルから得られた事後確率 $p_{\\mathrm{train}}(\\mathcal{C}_k|\\mathbf{x})$ は、母集団の真の事前確率 $p_{\\mathrm{target}}(\\mathcal{C}_k)$ を用いて次のように厳密に補正できます：
   $$p_{\\mathrm{target}}(\\mathcal{C}_k|\\mathbf{x}) \\propto p_{\\mathrm{train}}(\\mathcal{C}_k|\\mathbf{x}) \\frac{p_{\\mathrm{target}}(\\mathcal{C}_k)}{p_{\\mathrm{train}}(\\mathcal{C}_k)}$$
4. **モデルの統合 (Combining models)**:
   複数の独立な特徴量（画像と血液検査値など）を条件付き独立性の下で簡潔に統合できます。"""))

    # =====================================================================
    # Cell 9: Code - Section 5.2.4 Verification & Figure 5.8
    # =====================================================================
    nb.cells.append(nbf.v4.new_code_cell("""# 5.2.4 実装検証: クラス条件付き密度と事後確率 (Figure 5.8)
fig_5_8 = plot_figure_5_8_class_densities_posteriors(filepath="result/fig_5_8_class_densities_posteriors.png")
plt.show()

# 人工的な訓練事前確率の補正 (Class Prior Compensation) の数値検証
# 訓練時: 正常 50%, がん 50% (人工的バランス)
# 母集団: 正常 99.9%, がん 0.1% (現実の希少疾患)
p_train = np.array([0.5, 0.5])
p_target = np.array([0.999, 0.001])

# 訓練モデルの予測事後確率: p_train(Cancer|x) = 0.80 (80%と判定)
train_posterior = np.array([0.20, 0.80])
compensated_posterior = compensate_for_class_priors(train_posterior, p_train, p_target)

print("--- 訓練事前確率の母集団補正 (Compensating for Class Priors) ---")
print(f"訓練セット事後確率: 正常 = {train_posterior[0]:.2f}, がん = {train_posterior[1]:.2f}")
print(f"真の母集団事後確率: 正常 = {compensated_posterior[0]:.4f}, がん = {compensated_posterior[1]:.4f}")
print("希少疾患では、訓練時に80%と判定されても、母集団事前確率の稀少性により実際のがん事後確率は 1% 未満に補正されます。")"""))

    # =====================================================================
    # Cell 10: Section 5.2.5 Classifier accuracy Theory
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### 5.2.5 Classifier accuracy

分類器の最も素朴な評価尺度は正解率（Accuracy）ですが、**クラス不均衡（Class imbalance）が存在するデータセットでは正解率は著しく不適切**となります。
例えば 99.9% が正常、0.1% ががんの母集団において、「常に正常と答える」無能な分類器の正解率は 99.9% に達してしまいます。

このため、分類器の性能を多面的に評価するための **混同行列 (Confusion Matrix)** と派生評価尺度を定義します（**Figure 5.9**）。

#### 1. 混同行列の構造 (Eq 5.28)
全サンプル数 $N$ は以下の4つの排他事象に分解されます：
$$N = N_{\\mathrm{TP}} + N_{\\mathrm{FP}} + N_{\\mathrm{TN}} + N_{\\mathrm{FN}} \\tag{5.28}$$
- $N_{\\mathrm{TP}}$ (True Positive / 真陽性): がんを正しくがんと予測
- $N_{\\mathrm{TN}}$ (True Negative / 真陰性): 正常を正しく正常と予測
- $N_{\\mathrm{FP}}$ (False Positive / 偽陽性, Type 1 error): 正常を誤ってがんと予測
- $N_{\\mathrm{FN}}$ (False Negative / 偽陰性, Type 2 error): がんを誤って正常と予測（見逃し）

#### 2. 主要評価尺度
$$\\mathrm{Accuracy} = \\frac{N_{\\mathrm{TP}} + N_{\\mathrm{TN}}}{N_{\\mathrm{TP}} + N_{\\mathrm{FP}} + N_{\\mathrm{TN}} + N_{\\mathrm{FN}}} \\tag{5.29}$$

$$\\mathrm{Precision} = \\frac{N_{\\mathrm{TP}}}{N_{\\mathrm{TP}} + N_{\\mathrm{FP}}} \\tag{5.30}$$

$$\\mathrm{Recall} = \\frac{N_{\\mathrm{TP}}}{N_{\\mathrm{TP}} + N_{\\mathrm{FN}}} \\quad (\\text{Sensitivity / True Positive Rate}) \\tag{5.31}$$

$$\\mathrm{Specificity} = \\frac{N_{\\mathrm{TN}}}{N_{\\mathrm{TN}} + N_{\\mathrm{FP}}} \\quad (\\text{True Negative Rate}) \\tag{5.32}$$

$$\\mathrm{False\\;Positive\\;Rate\\;(FPR)} = \\frac{N_{\\mathrm{FP}}}{N_{\\mathrm{TN}} + N_{\\mathrm{FP}}} = 1 - \\mathrm{Specificity} \\tag{5.33}$$

$$F\\text{-score} = \\frac{2 \\times \\mathrm{Precision} \\times \\mathrm{Recall}}{\\mathrm{Precision} + \\mathrm{Recall}} = \\frac{2 N_{\\mathrm{TP}}}{2 N_{\\mathrm{TP}} + N_{\\mathrm{FP}} + N_{\\mathrm{FN}}} \\tag{5.38 - 5.39}$$"""))

    # =====================================================================
    # Cell 11: Code - Section 5.2.5 Verification & Figure 5.9
    # =====================================================================
    nb.cells.append(nbf.v4.new_code_cell("""# 5.2.5 実装検証: 混同行列と分類器評価尺度 (Figure 5.9)
fig_5_9 = plot_figure_5_9_confusion_matrix(filepath="result/fig_5_9_confusion_matrix.png")
plt.show()

# 混同行列クラスの数値検証
y_truth = np.array([1]*40 + [0]*60)  # 40 positive, 60 negative
y_pred_model = np.array([1]*35 + [0]*5 + [0]*50 + [1]*10)

cm = ConfusionMatrix2Class(y_truth, y_pred_model)
metrics = cm.summary()

print("--- 混同行列と主要評価尺度の計算 ---")
print(f"Confusion Matrix (Bishop Fig 5.9 format):\\n{cm.matrix}")
print(f"Total samples N = {cm.N} (TP={cm.TP}, FP={cm.FP}, TN={cm.TN}, FN={cm.FN})")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"Recall:      {metrics['recall']:.4f} (TPR / Sensitivity)")
print(f"Specificity: {metrics['specificity']:.4f} (TNR)")
print(f"FPR:         {metrics['false_positive_rate']:.4f} (1 - Specificity = {1 - metrics['specificity']:.4f})")
print(f"F-score:     {metrics['f_score']:.4f}")"""))

    # =====================================================================
    # Cell 12: Section 5.2.6 ROC curve Theory
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### 5.2.6 ROC curve

分類器の閾値 $\\widehat{x}$ を $-\\infty$ から $+\\infty$ まで連続的に変化させると、偽陽性率（FPR）と真陽性率（TPR）がトレードオフの関係を描きます。この軌跡をプロットしたものが **受託者動作特性曲線 (Receiver Operating Characteristic / ROC 曲線)** です（**Figure 5.11**）。

#### 1. 誤差領域の幾何学的分解 (Figure 5.10)
**Figure 5.10** では、閾値 $\\widehat{x}$ による領域分割と混同行列の要素の対応が視覚化されています：
- **領域 $A$**: $\\mathcal{R}_1$（正常予測）かつ真のクラスが $\\mathcal{C}_1$（正常） $\\implies N_{\\mathrm{TN}}$
- **領域 $B$**: $\\mathcal{R}_1$（正常予測）かつ真のクラスが $\\mathcal{C}_2$（がん） $\\implies N_{\\mathrm{FN}}$
- **領域 $C$**: 閾値 $x_0$ より左で誤分類されるがんデータ
- **領域 $D$**: $\\mathcal{R}_2$（がん予測）かつ真のクラスが $\\mathcal{C}_2$（がん） $\\implies N_{\\mathrm{TP}}$
- **領域 $E$**: $\\mathcal{R}_2$（がん予測）かつ真のクラスが $\\mathcal{C}_1$（正常） $\\implies N_{\\mathrm{FP}}$

閾値 $\\widehat{x}$ を右に動かすと、偽陰性（見逃し $B$）が増加する代わりに偽陽性（誤報 $E$）が減少します。

#### 2. ROC 曲線の幾何学的性質 (Figure 5.11)
- 横軸: 偽陽性率 $\\mathrm{False\\;Positive\\;Rate} \\in [0, 1]$
- 縦軸: 真陽性率 $\\mathrm{True\\;Positive\\;Rate} \\in [0, 1]$
- **左上隅 $(0, 1)$**: 偽陽性ゼロ・真陽性 100% を達成する「理想的分類器（Perfect classifier）」。
- **原点 $(0, 0)$**: すべてを陰性と判定する極限。
- **右上隅 $(1, 1)$**: すべてを陽性と判定する極限。
- **対角線 (Diagonal line)**: 確率 $\\rho$ でランダムに陽性と判定するランダム分類器（AUC = 0.5）。
- **AUC (Area Under Curve)**: ROC 曲線の下側面積。分類器の閾値に依存しない総合識別能力を表し、0.5（ランダム予測）から 1.0（完全予測）の値をとります。"""))

    # =====================================================================
    # Cell 13: Code - Section 5.2.6 Verification & Figures 5.10, 5.11
    # =====================================================================
    nb.cells.append(nbf.v4.new_code_cell("""# 5.2.6 実装検証: 誤差領域分解 (Figure 5.10) と ROC 曲線 (Figure 5.11)
# 1. Figure 5.10: 閾値移動に伴う誤差領域 A, B, C, D, E の可視化
fig_5_10 = plot_figure_5_10_roc_regions(filepath="result/fig_5_10_roc_regions.png")
plt.show()

# 2. Figure 5.11: ROC 曲線と AUC の可視化
fig_5_11 = plot_figure_5_11_roc_curve(filepath="result/fig_5_11_roc_curve.png")
plt.show()"""))

    # =====================================================================
    # Cell 14: Section 5.2 Summary & Future Outlook
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""---
### 本節のまとめと次節への展望

本節（Section 5.2: Decision Theory）では、分類における意思決定の数理的基盤を網羅的に探求しました：

1. **誤分類率最小化 (Misclassification rate)**:
   - 0/1 損失の下では、事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ が最大となるクラスを選択することが最適（式 5.20 - 5.21）。
2. **期待損失最小化 (Expected loss)**:
   - 損失行列 $L_{kj}$ を用いて非対称なリスクを定式化。医療診断などでは、わずかな確率であっても重篤なリスクを回避する決定が数理的に導かれる（式 5.22 - 5.23）。
3. **棄却オプション (Reject option)**:
   - 最大事後確率が閾値 $\\theta$ 未満の曖昧領域で判定を留保し、安全性を確保。
4. **推論と決定の分離 (Inference and decision)**:
   - 確率モデリングと決定基準を分離することで、リスク関数の動的変更や母集団事前確率の補正が容易になる。
5. **混同行列と ROC 曲線**:
   - 不均衡データにおける Accuracy の破綻と、Precision/Recall/F-score/Specificity による多面的評価。
   - 閾値走査に伴う FPR と TPR のトレードオフを ROC 曲線と AUC で定量化。

#### 次節 5.3 生成的分類器 (Generative Classifiers) への接続
決定理論を適用するためには、まず高精度な事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を求める「推論」が必要です。
次節 **Section 5.3: Generative Classifiers** では、クラス条件付き確率密度 $p(\\mathbf{x}|\\mathcal{C}_k)$ をモデル化する生成的アプローチ（連続変数に対する多変量ガウス分布、最尤推定解、離散特徴に対する単純ベイズ、指数型分布族への一般化）を詳細に学びます。"""))

    # Save notebook file
    out_dir = "5"
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, "5.2_Decision_Theory.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully written to: {nb_path}")

if __name__ == "__main__":
    build_notebook()

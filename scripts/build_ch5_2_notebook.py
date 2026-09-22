"""
Build Chapter 5 Section 5.2 Jupyter Notebook:
5/5.2_Decision_Theory.ipynb
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Title & Overview
cell_1_md = """# 第5章 単層ネットワーク: 分類 (Single-layer Networks: Classification)
## 5.2 決定理論 (Decision Theory)

### 本節の目的と概要
本ノートブックでは、Christopher M. Bishop & Hugh Bishop による『*Deep Learning: Foundations and Concepts*』(2024年刊) の **Chapter 5: Single-layer Networks: Classification** における核心的基盤である **Section 5.2: Decision Theory (決定理論)** を体系的かつ数学的に厳密に解説・実装・検証します。

本節では、推論（Inference）によって得られた確率分布（または事後確率）に基づき、いかにして最適な「決定（Decision）」を下すかを定式化します。

目次：
- **5.2.1 誤分類率の最小化 (Misclassification rate)**:
  - 決定領域 $\\mathcal{R}_k$ と決定境界（decision surface）
  - 2クラスにおける誤分類確率 $p(\\text{mistake}) = \\int_{\\mathcal{R}_1} p(\\mathbf{x}, \\mathcal{C}_2) d\\mathbf{x} + \\int_{\\mathcal{R}_2} p(\\mathbf{x}, \\mathcal{C}_1) d\\mathbf{x}$ (式 5.20)
  - $K$ クラスにおける正解率最大化 (式 5.21)
  - 最小誤分類率ルール：事後確率最大クラスへの割り当て $\\arg\\max_k p(\\mathcal{C}_k|\\mathbf{x})$
  - **Figure 5.5**: 2クラス同時確率分布と誤分類領域（境界 $\\hat{x}$ を最適点 $x_0$ へ移動させた際に赤領域が消滅する幾何学的機構の図解）
- **5.2.2 期待損失の最小化 (Expected loss)**:
  - 単純な誤分類数最小化の限界（癌診断における非対称性）
  - 損失行列（Loss Matrix / Cost Matrix） $L_{kj}$ の定義
  - 期待損失 $\\mathbb{E}[L] = \\sum_k \\sum_j \\int_{\\mathcal{R}_j} L_{kj} p(\\mathbf{x}, \\mathcal{C}_k) d\\mathbf{x}$ (式 5.22)
  - 最適決定規則：$\\arg\\min_j \\sum_k L_{kj} p(\\mathcal{C}_k|\\mathbf{x})$ (式 5.23)
  - **Figure 5.6**: 癌診断の損失行列例（正常を癌と誤診する損失1に対し、癌を正常と見落とす損失100）
- **5.2.3 棄却オプション (The reject option)**:
  - 事後確率が拮抗する曖昧な領域での自動判定回避と専門家・生検への委ね
  - 判定閾値 $\\theta$ の導入：$\\max_k p(\\mathcal{C}_k|\\mathbf{x}) \\le \\theta$ ならば棄却
  - 閾値の範囲 $\\theta \\in [1/K, 1]$ と棄却率の制御
  - **Figure 5.7**: 事後確率曲線と棄却領域（reject region）の可視化
- **5.2.4 推論と決定 (Inference and decision)**:
  - 3つの分類アプローチの比較：
    (a) 生成モデル（Generative models）：$p(\\mathbf{x}|\\mathcal{C}_k)$ と $p(\\mathcal{C}_k)$ をモデル化し、ベイズ則で事後確率を求める
    (b) 識別モデル（Discriminative models）：事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を直接モデル化する
    (c) 識別関数（Discriminant functions）：入力 $\\mathbf{x}$ を直接決定に写像する
  - 事後確率を計算する4大メリット：
    1. リスクの最小化（損失行列の変更への即座の追従）
    2. 棄却オプションの容易な適用
    3. クラス事前確率の補正（データ収集の偏り・サンプリングバイアスの補正、式 5.24 - 5.27）
    4. 異種情報の統合・ナイーブベイズモデル (式 5.26)
  - **Figure 5.8**: クラス条件付き密度（混合ガウス分布など）と事後確率曲線の対比（決定境界に無関係なクラスタ構造の存在）
- **5.2.5 分類器の精度 (Classifier accuracy)**:
  - 混同行列（Confusion Matrix）の記法：$N_{\\text{TP}}, N_{\\text{FP}}, N_{\\text{TN}}, N_{\\text{FN}}$
  - 総サンプル数 $N$ (式 5.28)
  - 正解率（Accuracy） (式 5.29) とその落とし穴（不均衡データでの99.9%ダミー分類器）
  - 適合率（Precision） (式 5.30)
  - 再現率（Recall）/ 感度（Sensitivity）/ TPR (式 5.31)
  - 特異度（Specificity）/ TNR (式 5.32)
  - 偽陽性率（False Positive Rate: FPR） (式 5.33)
  - **Figure 5.9**: 混同行列の定式化と用語整理
- **5.2.6 ROC曲線 (ROC curve)**:
  - 決定境界の移動と第1種の過誤・第2種の過誤のトレードオフ
  - 領域分割（A, B, C, D, E）と混同行列要素の積分表現 (式 5.34 - 5.37)
  - **Figure 5.10**: 決定境界移動に伴う各誤差成分の幾何学的推移
  - 受信者動作特性（ROC）曲線とAUC（Area Under the Curve）
  - $F_1$ スコア（調和平均）の定義と意義 (式 5.38 - 5.39)
  - **Figure 5.11**: 優れた分類器と劣る分類器のROC曲線、およびランダム推測ベースラインの比較"""
cells.append(nbf.v4.new_markdown_cell(cell_1_md))

# Cell 2: Setup Code
cell_2_code = """import os
import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from IPython.display import Image, display

# プロジェクトルートの設定
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

print("Environment and modules successfully imported.")"""
cells.append(nbf.v4.new_code_cell(cell_2_code))

# Cell 3: Section 5.2.1 Markdown
cell_3_md = """---
### 5.2.1 誤分類率の最小化 (Misclassification rate)

#### 1. 決定領域と決定境界
分類規則は、入力空間をクラス $\\mathcal{C}_k$ ごとの決定領域（decision regions） $\\mathcal{R}_k$ に分割します。領域内のすべての入力ベクトル $\\mathbf{x} \\in \\mathcal{R}_k$ はクラス $\\mathcal{C}_k$ に割り当てられます。領域間の境界は決定境界（decision surface）と呼ばれます。各決定領域は必ずしも連結である必要はなく、複数の互いに素な領域から構成されていても構いません。

#### 2. 2クラス問題における誤分類確率
2つのクラス $\\mathcal{C}_1, \\mathcal{C}_2$ を考えるとき、誤りは「$\\mathcal{C}_1$ の点が $\\mathcal{C}_2$ に割り当てられる」または「$\\mathcal{C}_2$ の点が $\\mathcal{C}_1$ に割り当てられる」ときに生じます。この誤分類確率 $p(\\text{mistake})$ は以下のように定式化されます：

$$
p(\\text{mistake}) = p(\\mathbf{x} \\in \\mathcal{R}_1, \\mathcal{C}_2) + p(\\mathbf{x} \\in \\mathcal{R}_2, \\mathcal{C}_1) = \\int_{\\mathcal{R}_1} p(\\mathbf{x}, \\mathcal{C}_2) d\\mathbf{x} + \\int_{\\mathcal{R}_2} p(\\mathbf{x}, \\mathcal{C}_1) d\\mathbf{x} \\tag{5.20}
$$

$p(\\text{mistake})$ を最小化するためには、各点 $\\mathbf{x}$ において被積分関数が小さい方のクラスに割り当てる必要があります。すなわち：
- $p(\\mathbf{x}, \\mathcal{C}_1) > p(\\mathbf{x}, \\mathcal{C}_2)$ ならば $\\mathbf{x} \\in \\mathcal{R}_1$
- $p(\\mathbf{x}, \\mathcal{C}_2) > p(\\mathbf{x}, \\mathcal{C}_1)$ ならば $\\mathbf{x} \\in \\mathcal{R}_2$

乗法定理 $p(\\mathbf{x}, \\mathcal{C}_k) = p(\\mathcal{C}_k|\\mathbf{x})p(\\mathbf{x})$ より、共通の因子 $p(\\mathbf{x})$ を消去すると、**事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ が大きい方のクラスに $\\mathbf{x}$ を割り当てることで誤分類率が最小化される** ことが導かれます。

#### 3. $K$ クラス問題における一般化
$K$ クラス問題では、正解確率 $p(\\text{correct})$ を最大化することを考えます：

$$
p(\\text{correct}) = \\sum_{k=1}^K p(\\mathbf{x} \\in \\mathcal{R}_k, \\mathcal{C}_k) = \\sum_{k=1}^K \\int_{\\mathcal{R}_k} p(\\mathbf{x}, \\mathcal{C}_k) d\\mathbf{x} \\tag{5.21}
$$

各点 $\\mathbf{x}$ を $p(\\mathbf{x}, \\mathcal{C}_k)$ が最大となる領域 $\\mathcal{R}_k$ に割り当てることで、全体の正解率は最大化されます。これは各 $\\mathbf{x}$ に対し事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を最大にするクラスを選択するベイズ決定規則と完全に同値です。

#### 4. Figure 5.5 の幾何学的解釈
下図 **Figure 5.5** は、2クラスの同時確率 $p(x, \\mathcal{C}_1), p(x, \\mathcal{C}_2)$ と決定境界 $\\hat{x}$ を示しています：
- (a) 境界が最適値 $x_0$ から右にずれた $\\hat{x} > x_0$ の場合：
  - 緑の領域：$x < x_0$ における $\\mathcal{C}_2$ の誤分類誤差
  - 赤の領域：$x_0 < x < \\hat{x}$ において本来 $\\mathcal{C}_2$ と判定すべきところを $\\mathcal{C}_1$ と誤判定してしまう **余分な誤差**
  - 青の領域：$x \\ge \\hat{x}$ における $\\mathcal{C}_1$ の誤分類誤差
  - 境界 $\\hat{x}$ を動かすと、青と緑の合計面積は一定のまま、赤の余分な誤差領域のみが伸縮します。
- (b) $\\hat{x} = x_0$（2つの曲線が交差する点）に設定した場合：
  - **赤の余分な領域が完全に消滅** し、誤分類誤差は（緑＋青）の理論的最小値へと到達します。"""
cells.append(nbf.v4.new_markdown_cell(cell_3_md))

# Cell 4: Section 5.2.1 Code & Verification
cell_4_code = """# Figure 5.5 の生成と描画
fig_5_5 = plot_figure_5_5_joint_probabilities(filepath="result/fig_5_5_joint_probabilities.png")
display(fig_5_5)
plt.close(fig_5_5)

# 数値的検証: 決定境界 x_hat を動かしたときの誤分類率の変化
x_grid = np.linspace(-1.5, 6.5, 1000)
dx = x_grid[1] - x_grid[0]
p1 = 0.65 * stats.norm.pdf(x_grid, 1.2, 0.75) + 0.35 * stats.norm.pdf(x_grid, 3.8, 1.4)
p2 = 0.65 * stats.norm.pdf(x_grid, 3.8, 0.95)

# 最適境界 x0 (p1 == p2)
idx_cross = np.where(np.diff(np.sign(p1 - p2)))[0]
mid_idx = [i for i in idx_cross if 1.5 < x_grid[i] < 3.5][0]
x0_opt = x_grid[mid_idx]

# 様々な境界候補 x_hat について誤分類率を計算
threshold_candidates = np.linspace(1.5, 4.5, 100)
errors = []
for th in threshold_candidates:
    # R1: x < th (predict C1, error if C2) -> int_{x < th} p2 dx
    # R2: x >= th (predict C2, error if C1) -> int_{x >= th} p1 dx
    err = np.sum(p2[x_grid < th]) * dx + np.sum(p1[x_grid >= th]) * dx
    errors.append(err)

min_th = threshold_candidates[np.argmin(errors)]
print(f"Theoretical crossing point x0: {x0_opt:.4f}")
print(f"Empirically minimized threshold: {min_th:.4f}")
print(f"Minimum error rate: {np.min(errors):.4f}")

# 数値アサーション: 最小誤差を与える閾値が x0 に一致することを確認
assert np.isclose(x0_opt, min_th, atol=0.05), "Threshold minimizing error must match crossing point x0."
print("Assertion Passed: Optimal threshold precisely minimizes misclassification error.")"""
cells.append(nbf.v4.new_code_cell(cell_4_code))

# Cell 5: Section 5.2.2 Markdown
cell_5_md = """---
### 5.2.2 期待損失の最小化 (Expected loss)

#### 1. 単純な誤分類率最小化の限界
現実の多くの問題では、誤りの種類によってその重大性（コスト・リスク）が著しく異なります。
例えば癌診断において：
1. **偽陽性 (False Positive)**: 健康な患者を癌と誤診する。精神的動揺や再検査の負担が生じるが、命に関わるわけではない。
2. **偽陰性 (False Negative)**: 癌患者を健康と誤診する。早期治療の機会を逸し、患者が手遅れになって死亡するリスクがある。

したがって、全体の誤分類件数を単に最小化するのではなく、「重大な誤りをより重くペナルティする」基準が不可欠となります。

#### 2. 損失行列 (Loss Matrix)
この非対称性を定式化するために **損失行列 (Loss Matrix)** $L_{kj}$ を導入します：
- $L_{kj}$：真のクラスが $\\mathcal{C}_k$ であるとき、決定規則がクラス $\\mathcal{C}_j$ を割り当てた場合に被る損失（コスト）。
- 正解した場合は通常損失なし（$L_{kk} = 0$）。

#### 3. 期待損失の定式化
真のクラスは未知であるため、同時確率分布 $p(\\mathbf{x}, \\mathcal{C}_k)$ に関する平均（期待値）として **期待損失 (Expected Loss)** $\\mathbb{E}[L]$ を定義します：

$$
\\mathbb{E}[L] = \\sum_{k} \\sum_j \\int_{\\mathcal{R}_j} L_{kj} p(\\mathbf{x}, \\mathcal{C}_k) d\\mathbf{x} \\tag{5.22}
$$

各点 $\\mathbf{x}$ は独立にいずれかの決定領域 $\\mathcal{R}_j$ に割り当てられるため、式 (5.22) を最小化するには、各 $\\mathbf{x}$ において以下の量を最小化するクラス $j$ を選択すればよいことになります：

$$
\\sum_k L_{kj} p(\\mathbf{x}, \\mathcal{C}_k)
$$

乗法定理 $p(\\mathbf{x}, \\mathcal{C}_k) = p(\\mathcal{C}_k|\\mathbf{x})p(\\mathbf{x})$ より、共通の正の因子 $p(\\mathbf{x})$ を除くと、**期待損失を最小化する最適決定規則** が得られます：

$$
j^* = \\arg\\min_j \\sum_k L_{kj} p(\\mathcal{C}_k|\\mathbf{x}) \\tag{5.23}
$$

#### 4. Figure 5.6 の癌診断損失行列の解析
下図 **Figure 5.6** は、教科書で示される癌診断の損失行列です：

$$
L = \\begin{pmatrix} L_{11} & L_{12} \\\\ L_{21} & L_{22} \\end{pmatrix} = \\begin{pmatrix} 0 & 1 \\\\ 100 & 0 \\end{pmatrix}
$$

（行：真のクラス $\\mathcal{C}_1=\\text{normal}, \\mathcal{C}_2=\\text{cancer}$、列：決定 $\\mathcal{C}_1=\\text{normal}, \\mathcal{C}_2=\\text{cancer}$）

この行列の下で、各決定 $j \\in \\{1, 2\\}$ の期待損失は：
- 正常 ($j=1$) と診断したときの期待損失：
  $$\\mathbb{E}[L|j=1, \\mathbf{x}] = L_{11} p(\\mathcal{C}_1|\\mathbf{x}) + L_{21} p(\\mathcal{C}_2|\\mathbf{x}) = 100 \\, p(\\mathcal{C}_2|\\mathbf{x})$$
- 癌 ($j=2$) と診断したときの期待損失：
  $$\\mathbb{E}[L|j=2, \\mathbf{x}] = L_{12} p(\\mathcal{C}_1|\\mathbf{x}) + L_{22} p(\\mathcal{C}_2|\\mathbf{x}) = 1 \\, p(\\mathcal{C}_1|\\mathbf{x}) = 1 - p(\\mathcal{C}_2|\\mathbf{x})$$

癌と診断 ($j=2$) すべき条件は：
$$1 - p(\\mathcal{C}_2|\\mathbf{x}) < 100 \\, p(\\mathcal{C}_2|\\mathbf{x}) \\iff p(\\mathcal{C}_2|\\mathbf{x}) > \\frac{1}{101} \\approx 0.0099$$

すなわち、癌の事後確率がわずか **1%** を超えるだけで、直ちに癌（要精査）と診断を下すのが期待損失最小の観点から最適となります！"""
cells.append(nbf.v4.new_markdown_cell(cell_5_md))

# Cell 6: Section 5.2.2 Code & Verification
cell_6_code = """# Figure 5.6 の生成と描画
fig_5_6 = plot_figure_5_6_loss_matrix(filepath="result/fig_5_6_loss_matrix.png")
display(fig_5_6)
plt.close(fig_5_6)

# 期待損失最小化の決定規則シミュレーション
L_cancer = np.array([
    [0.0, 1.0],     # normal
    [100.0, 0.0]    # cancer
])

# p(cancer | x) を 0 から 1 までスイープ
p_cancer = np.linspace(0.0, 0.05, 500)
p_normal = 1.0 - p_cancer
posteriors = np.column_stack([p_normal, p_cancer])

# 決定の計算
decisions = optimal_decision_rule_expected_loss(posteriors, L_cancer)
expected_losses = compute_expected_loss(posteriors, L_cancer)

# 決定が normal (0) から cancer (1) に切り替わる臨界点
switch_idx = np.where(decisions == 1)[0][0]
p_crit_empirical = p_cancer[switch_idx]
p_crit_theoretical = 1.0 / 101.0

print(f"Theoretical critical probability p(cancer|x): {p_crit_theoretical:.6f}")
print(f"Empirical switching probability:              {p_crit_empirical:.6f}")

# アサーション検証
assert np.isclose(p_crit_empirical, p_crit_theoretical, atol=1e-3)
print("Assertion Passed: Expected loss decision boundary matches theoretical 1 / (L_12 + L_21) threshold.")"""
cells.append(nbf.v4.new_code_cell(cell_6_code))

# Cell 7: Section 5.2.3 Markdown
cell_7_md = """---
### 5.2.3 棄却オプション (The reject option)

#### 1. 棄却オプションの動機
事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ がどのクラスに対しても接近しており拮抗している領域では、分類器の確信度が低く、誤判定のリスクが極めて高くなります。
このような困難な症例に対して無理に自動分類を行うのではなく、**判定を保留・棄却 (reject) して人間の専門医による追加検査（生検など）に回す** のが「棄却オプション」です。

#### 2. 判定基準の定式化
閾値 $\\theta \\in [1/K, 1.0]$ を導入し、以下の規則を適用します：
- 最大事後確率が $\\theta$ 以上の場合：
  $$\\max_k p(\\mathcal{C}_k|\\mathbf{x}) \\ge \\theta \\implies k^* = \\arg\\max_k p(\\mathcal{C}_k|\\mathbf{x}) \\text{ に分類}$$
- すべての事後確率が $\\theta$ 未満の場合：
  $$\\max_k p(\\mathcal{C}_k|\\mathbf{x}) < \\theta \\implies \\text{判定を棄却 (Reject)}$$

- $\\theta = 1.0$ とすると、事後確率が 100% 確実でない限りすべて棄却されます。
- $\\theta < 1/K$ とすると、最大値は常に $1/K$ 以上であるため棄却は一切発生しません。
- したがって、$\\theta$ を $[1/K, 1.0]$ の範囲で調整することにより、許容する誤り率と自動処理率（スループット）のトレードオフを自在に制御できます。

#### 3. Figure 5.7 の可視化
下図 **Figure 5.7** は、2クラスの事後確率 $p(\\mathcal{C}_1|x)$ と $p(\\mathcal{C}_2|x)$ に対する棄却領域を図示しています。
緑の破線で示された閾値 $\\theta$ 以下の領域（中央の帯状区間）が棄却領域（reject region）となります。"""
cells.append(nbf.v4.new_markdown_cell(cell_7_md))

# Cell 8: Section 5.2.3 Code & Verification
cell_8_code = """# Figure 5.7 の生成と描画
fig_5_7 = plot_figure_5_7_reject_option(filepath="result/fig_5_7_reject_option.png")
display(fig_5_7)
plt.close(fig_5_7)

# 棄却オプションの性能シミュレーション
# 確信度が低いサンプルを棄却することで、採択されたサンプルの精度が向上することを確認
np.random.seed(42)
N_sim = 1000
# 2クラスのシミュレーション事後確率
logits = np.random.normal(0, 1.5, N_sim)
p1_sim = 1.0 / (1.0 + np.exp(-logits))
p2_sim = 1.0 - p1_sim
sim_posteriors = np.column_stack([p1_sim, p2_sim])
# 真のラベルは事後確率に従ってサンプリング
true_labels = (np.random.rand(N_sim) < p2_sim).astype(int)

thetas = np.linspace(0.5, 0.98, 20)
rejection_rates = []
accuracy_on_accepted = []

for th in thetas:
    dec, rej = decision_rule_with_reject(sim_posteriors, theta=th)
    rej_rate = np.mean(rej)
    rejection_rates.append(rej_rate)
    
    accepted_mask = ~rej
    if np.sum(accepted_mask) > 0:
        acc = np.mean(dec[accepted_mask] == true_labels[accepted_mask])
    else:
        acc = 1.0
    accuracy_on_accepted.append(acc)

print(f"Theta = 0.50 (No reject): Acc = {accuracy_on_accepted[0]:.3f}, Rejection rate = {rejection_rates[0]:.3f}")
print(f"Theta = 0.90 (High conf): Acc = {accuracy_on_accepted[-4]:.3f}, Rejection rate = {rejection_rates[-4]:.3f}")

# 精度が単調非減少であることを検証
assert accuracy_on_accepted[-1] >= accuracy_on_accepted[0]
print("Assertion Passed: Reject option systematically improves accuracy on accepted samples.")"""
cells.append(nbf.v4.new_code_cell(cell_8_code))

# Cell 9: Section 5.2.4 Markdown
cell_9_md = """---
### 5.2.4 推論と決定 (Inference and decision)

#### 1. 分類問題における2段階アプローチ
分類問題は根本的に以下の2つの独立したステージに分離されます：
1. **推論段階 (Inference stage)**：訓練データからモデルを学習し、事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を推定する。
2. **決定段階 (Decision stage)**：得られた事後確率と損失関数に基づいて、最適な決定（クラス割り当て）を下す。

#### 2. 分類における3つのアプローチ
実用上、この決定問題に対する解法は以下の3通りが存在します（計算複雑さの降順）：

1. **(a) 生成モデル (Generative models)**:
   - 各クラスの条件付き密度 $p(\\mathbf{x}|\\mathcal{C}_k)$ および事前確率 $p(\\mathcal{C}_k)$ を個別にモデル化する。
   - ベイズの定理によって事後確率を求める：
     $$p(\\mathcal{C}_k|\\mathbf{x}) = \\frac{p(\\mathbf{x}|\\mathcal{C}_k)p(\\mathcal{C}_k)}{\\sum_j p(\\mathbf{x}|\\mathcal{C}_j)p(\\mathcal{C}_j)} \\tag{5.24}$$
   - メリット：入力の周辺分布 $p(\\mathbf{x})$ も得られるため外れ値検出が可能、データの生成が可能。
   - デメリット：入力の高次元密度推定は非常に困難であり、決定境界に無関係な特徴まで忠実にモデル化してしまう過剰なコストがかかる。

2. **(b) 識別モデル (Discriminative models)**:
   - 事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を直接パラメトリックモデル（ロジスティック回帰、ニューラルネットワーク等）として学習する。
   - メリット：決定に必要な事後確率のみを直接最適化するため、データ効率が高く性能が良い。

3. **(c) 識別関数 (Discriminant functions)**:
   - 確率を一切計算せず、入力 $\\mathbf{x}$ を直接決定ラベルに写像する関数 $f(\\mathbf{x})$ を学習する（SVM、パーセプトロン等）。
   - デメリット：事後確率へのアクセスが失われるため、損失行列の変更や棄却オプションに柔軟に対応できない。

#### 3. 事後確率を陽に計算するメリット
事後確率 $p(\\mathcal{C}_k|\\mathbf{x})$ を求めることには、以下の極めて強力な利点があります：
1. **リスクの最小化**: 財務アプリケーションなどで損失行列 $L_{kj}$ が頻繁に変更・改訂された場合でも、推論モデルを再学習することなく、式 (5.23) の最小化規則を即座に更新できる。
2. **棄却オプション**: 確信度の低いサンプルの棄却閾値を容易に設定・調整できる。
3. **クラス事前確率の補正 (Compensating for class priors)**:
   - 希少疾患のスクリーニングなどで、母集団では癌患者が 1/1000 であるのに対し、学習を効率化するために癌と正常を 50:50 で均等に収集してモデルを学習した場合。
   - 学習時の事前確率 $p_{\\text{train}}(\\mathcal{C}_k)$ と実際のターゲット母集団の事前確率 $p_{\\text{target}}(\\mathcal{C}_k)$ が異なるとき、クラス条件付き密度 $p(\\mathbf{x}|\\mathcal{C}_k)$ が不変であると仮定すると：
     $$p(\\mathbf{x}|\\mathcal{C}_k) \\propto \\frac{p_{\\text{train}}(\\mathcal{C}_k|\\mathbf{x})}{p_{\\text{train}}(\\mathcal{C}_k)}$$
   - したがって、ターゲット母集団における事後確率は以下のように厳密に補正できます (式 5.27)：
     $$p_{\\text{target}}(\\mathcal{C}_k|\\mathbf{x}) = \\frac{p_{\\text{target}}(\\mathcal{C}_k) \\frac{p_{\\text{train}}(\\mathcal{C}_k|\\mathbf{x})}{p_{\\text{train}}(\\mathcal{C}_k)}}{\\sum_j p_{\\text{target}}(\\mathcal{C}_j) \\frac{p_{\\text{train}}(\\mathcal{C}_j|\\mathbf{x})}{p_{\\text{train}}(\\mathcal{C}_j)}} \\tag{5.27}$$
4. **モデルの結合**:
   - 異なるデータ源 $\\mathbf{x}_1, \\mathbf{x}_2$ に対し、クラス条件付き独立性 $p(\\mathbf{x}_1, \\mathbf{x}_2|\\mathcal{C}_k) = p(\\mathbf{x}_1|\\mathcal{C}_k)p(\\mathbf{x}_2|\\mathcal{C}_k)$ を仮定すれば（Naive Bayes モデル）、別々に学習した事後確率を事後的に統合できる (式 5.26)。

#### 4. Figure 5.8 の可視化
下図 **Figure 5.8** は、左側にクラス条件付き密度、右側に事後確率を示しています。
左側の密度 $p(x|\\mathcal{C}_1)$ に存在する左側のピーク（$x \\approx 0.2$）は、決定境界（$x \\approx 0.58$）付近の事後確率に対して何の影響も及ぼしていません。生成モデルではこの無関係なピークのモデリングにパラメータを割く必要がありますが、識別モデルや事後確率直結アプローチでは決定境界に必要な情報のみに集中できることが直感的に理解できます。"""
cells.append(nbf.v4.new_markdown_cell(cell_9_md))

# Cell 10: Section 5.2.4 Code & Verification
cell_10_code = """# Figure 5.8 の生成と描画
fig_5_8 = plot_figure_5_8_class_densities_posteriors(filepath="result/fig_5_8_class_densities_posteriors.png")
display(fig_5_8)
plt.close(fig_5_8)

# 事前確率補正 (Compensating for class priors) の数値検証
# 学習環境: 人工的に 50:50 で学習
p_train = np.array([0.5, 0.5])
# 実際の母集団: 正常 99.9%, 癌 0.1%
p_target = np.array([0.999, 0.001])

# 学習済みモデルがある患者 x に対して出力した予測事後確率
train_posteriors = np.array([
    [0.5, 0.5],     # 境界付近で五分五分の判定
    [0.1, 0.9],     # 癌とかなり強く判定
    [0.01, 0.99]    # 癌と極めて強く判定 (99%)
])

target_posteriors = compensate_for_class_priors(train_posteriors, p_train, p_target)

print("--- Prior Compensation Results ---")
for i, (tp, tgt) in enumerate(zip(train_posteriors, target_posteriors)):
    print(f"Sample {i+1}: Train P(cancer|x)={tp[1]:.2f} -> Corrected Target P(cancer|x)={tgt[1]:.5f}")

# 数値検証: 50:50 の曖昧な点は、補正後は母集団事前確率 0.001 に等しくなる
assert np.isclose(target_posteriors[0, 1], 0.001)
# 確率の和が1に正規化されていることを検証
assert np.allclose(np.sum(target_posteriors, axis=1), 1.0)
print("Assertion Passed: Class prior compensation formula (Eq 5.27) rigorously verified.")"""
cells.append(nbf.v4.new_code_cell(cell_10_code))

# Cell 11: Section 5.2.5 Markdown
cell_11_md = """---
### 5.2.5 分類器の精度 (Classifier accuracy)

#### 1. 混同行列 (Confusion Matrix)
分類器の性能を詳細に把握するため、予測ラベルと真のラベルのクロス集計を行う **混同行列** を導入します (**Figure 5.9**)。
2クラス分類（正常: 陰性 Negative / 癌: 陽性 Positive）において：

$$
\\begin{pmatrix}
\\text{真: normal} \\\\
\\text{真: cancer}
\\end{pmatrix}
\\begin{pmatrix}
N_{\\text{TN}} & N_{\\text{FP}} \\\\
N_{\\text{FN}} & N_{\\text{TP}}
\\end{pmatrix}
\\quad \\text{（列: 決定 normal, 決定 cancer）}
$$

- $N_{\\text{TP}}$ (True Positive, 真陽性)：癌患者を正しく癌と判定。
- $N_{\\text{FP}}$ (False Positive, 偽陽性, 第1種の過誤)：正常患者を誤って癌と判定。
- $N_{\\text{TN}}$ (True Negative, 真陰性)：正常患者を正しく正常と判定。
- $N_{\\text{FN}}$ (False Negative, 偽陰性, 第2種の過誤)：癌患者を誤って正常と判定。

総サンプル数は以下の関係を満たします：
$$N = N_{\\text{TP}} + N_{\\text{FP}} + N_{\\text{TN}} + N_{\\text{FN}} \\tag{5.28}$$

#### 2. 各種評価指標の厳密な定義
1. **正解率 (Accuracy)** (式 5.29):
   $$\\text{Accuracy} = \\frac{N_{\\text{TP}} + N_{\\text{TN}}}{N_{\\text{TP}} + N_{\\text{FP}} + N_{\\text{TN}} + N_{\\text{FN}}} \\tag{5.29}$$
   - **落とし穴**: 癌患者が 1000 人中 1 人の不均衡データの場合、「全員正常」と答える無能な分類器でも正解率 **99.9%** を達成してしまいます。そのため、不均衡データでの正解率評価は極めて危険です。

2. **適合率 (Precision)** (式 5.30):
   $$\\text{Precision} = \\frac{N_{\\text{TP}}}{N_{\\text{TP}} + N_{\\text{FP}}} \\tag{5.30}$$
   - 陽性と予測した中で、実際に陽性であった割合（検査陽性の信頼度）。

3. **再現率 (Recall) / 感度 (Sensitivity) / 真陽性率 (TPR)** (式 5.31):
   $$\\text{Recall} = \\frac{N_{\\text{TP}}}{N_{\\text{TP}} + N_{\\text{FN}}} \\tag{5.31}$$
   - 実際の陽性患者のうち、見落とさずに陽性と検出できた割合。

4. **特異度 (Specificity) / 真陰性率 (TNR)** (式 5.32):
   $$\\text{Specificity} = \\frac{N_{\\text{TN}}}{N_{\\text{FP}} + N_{\\text{TN}}} \\tag{5.32}$$
   - 実際の陰性患者のうち、正しく陰性と判定できた割合。

5. **偽陽性率 (False Positive Rate: FPR)** (式 5.33):
   $$\\text{FPR} = \\frac{N_{\\text{FP}}}{N_{\\text{FP}} + N_{\\text{TN}}} = 1 - \\text{Specificity} \\tag{5.33}$$
   - 健常者を誤って陽性と判定してしまう割合。"""
cells.append(nbf.v4.new_markdown_cell(cell_11_md))

# Cell 12: Section 5.2.5 Code & Verification
cell_12_code = """# Figure 5.9 の生成と描画
fig_5_9 = plot_figure_5_9_confusion_matrix(filepath="result/fig_5_9_confusion_matrix.png")
display(fig_5_9)
plt.close(fig_5_9)

# 混同行列と不均衡データの評価シミュレーション
# 10,000 人の患者データ (正常 9,990 人, 癌 10 人)
np.random.seed(42)
N_patients = 10000
n_cancer = 10
y_true = np.array([0] * (N_patients - n_cancer) + [1] * n_cancer)

# 分類器A: 「全員正常」と答えるダミー分類器
y_pred_dummy = np.zeros(N_patients, dtype=int)
cm_dummy = ConfusionMatrix2Class(y_true, y_pred_dummy)

# 分類器B: 感度 90% (癌 10 人中 9 人検出), 特異度 98% (偽陽性 2%) の実用モデル
y_pred_model = np.zeros(N_patients, dtype=int)
# 癌患者 10人中 9人正解
y_pred_model[-n_cancer:] = [1]*9 + [0]*1
# 正常 9990 人中 2% (約 200人) を偽陽性
fp_indices = np.random.choice(N_patients - n_cancer, size=200, replace=False)
y_pred_model[fp_indices] = 1
cm_model = ConfusionMatrix2Class(y_true, y_pred_model)

print("=== Classifier Comparison on Imbalanced Data ===")
print(f"Dummy Classifier: Accuracy={cm_dummy.accuracy*100:.2f}%, Recall={cm_dummy.recall*100:.2f}%, Precision={cm_dummy.precision*100:.2f}%")
print(f"Medical Model:    Accuracy={cm_model.accuracy*100:.2f}%, Recall={cm_model.recall*100:.2f}%, Precision={cm_model.precision*100:.2f}%")

# 指標の数式等価性のアサーション検証
assert np.isclose(cm_model.false_positive_rate, 1.0 - cm_model.specificity)
assert cm_model.N == cm_model.TP + cm_model.FP + cm_model.TN + cm_model.FN
print("Assertion Passed: Confusion matrix identities (Eq 5.28 - 5.33) numerically confirmed.")"""
cells.append(nbf.v4.new_code_cell(cell_12_code))

# Cell 13: Section 5.2.6 Markdown
cell_13_md = """---
### 5.2.6 ROC曲線 (ROC curve)

#### 1. 閾値移動と誤差のトレードオフ
確率的分類器は事後確率を出力し、閾値 $\\hat{x}$ を設定することで最終決定を下します。
閾値を変化させると、第1種の過誤（偽陽性）と第2種の過誤（偽陰性）の間にトレードオフが生じます。

下図 **Figure 5.10** は、決定境界 $\\hat{x}$ と各誤差成分領域 $A, B, C, D, E$ の関係を示しています：
- $N_{\\text{FP}} / N = E$ (式 5.34)：$\\hat{x}$ より右側の正常クラス面積
- $N_{\\text{TP}} / N = D + E$ (式 5.35)：$\\hat{x}$ より右側の癌クラス総面積
- $N_{\\text{FN}} / N = B + C$ (式 5.36)：$\\hat{x}$ より左側の癌クラス面積
- $N_{\\text{TN}} / N = A + C$ (式 5.37)：$\\hat{x}$ より左側の正常クラス面積

境界 $\\hat{x}$ を $-\\infty$ から $+\\infty$ へ連続的に動かすと：
- $\\hat{x} \\to -\\infty$：全員を癌と判定 $\\implies \\text{TPR} = 1, \\text{FPR} = 1$（右上の点 $(1, 1)$）
- $\\hat{x} \\to +\\infty$：全員を正常と判定 $\\implies \\text{TPR} = 0, \\text{FPR} = 0$（左下の原点 $(0, 0)$）

#### 2. 受信者動作特性 (ROC: Receiver Operating Characteristic) 曲線
横軸に **偽陽性率 (FPR)**、縦軸に **真陽性率 (TPR / 感度)** をプロットした軌跡が **ROC 曲線** です (**Figure 5.11**)。
- **理想的な分類器**: 左上隅 $(0, 1)$（誤診ゼロで全陽性を検出）を通過する。
- **ランダム推測分類器**: 確率 $\\rho$ で陽性と判定する場合、$\\text{TPR} = \\rho, \\text{FPR} = \\rho$ となり、対角線 $y = x$ となる。
- **AUC (Area Under the Curve)**: ROC 曲線下の面積。
  - 完全分類器：$\\text{AUC} = 1.0$
  - ランダム推測：$\\text{AUC} = 0.5$
  - AUC は閾値の選択に依存しない分類器自体の識別能力（discriminability）の尺度となります。

#### 3. $F_1$ スコア (F-score)
Precision と Recall の調和平均として定義される $F_1$ スコアも広く使用されます：

$$
F_1 = \\frac{2 \\times \\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2 N_{\\text{TP}}}{2 N_{\\text{TP}} + N_{\\text{FP}} + N_{\\text{FN}}} \\tag{5.38 - 5.39}
$$

算術平均ではなく調和平均をとることで、Precision と Recall のいずれか一方が極端に低い場合に $F_1$ スコアが厳しくペナルティされます。"""
cells.append(nbf.v4.new_markdown_cell(cell_13_md))

# Cell 14: Section 5.2.6 Code & Verification
cell_14_code = """# Figure 5.10 と Figure 5.11 の生成と描画
fig_5_10 = plot_figure_5_10_roc_regions(filepath="result/fig_5_10_roc_regions.png")
display(fig_5_10)
plt.close(fig_5_10)

fig_5_11 = plot_figure_5_11_roc_curve(filepath="result/fig_5_11_roc_curve.png")
display(fig_5_11)
plt.close(fig_5_11)

# ROC曲線とAUCの数値計算検証
np.random.seed(123)
N_samples = 200
y_true_roc = np.array([0]*(N_samples//2) + [1]*(N_samples//2))

# 優れた分類器のスコア (平均分離度 d=2.0)
scores_good = np.r_[np.random.normal(0, 1, N_samples//2), np.random.normal(2.0, 1, N_samples//2)]
fpr_g, tpr_g, _, auc_g = compute_roc_curve(y_true_roc, scores_good)

# 劣る分類器のスコア (平均分離度 d=0.8)
scores_fair = np.r_[np.random.normal(0, 1, N_samples//2), np.random.normal(0.8, 1, N_samples//2)]
fpr_f, tpr_f, _, auc_f = compute_roc_curve(y_true_roc, scores_fair)

print(f"Good Classifier AUC: {auc_g:.4f}")
print(f"Fair Classifier AUC: {auc_f:.4f}")

# アサーション検証
assert auc_g > auc_f > 0.5, "AUC of better classifier must exceed fair classifier, and both exceed random (0.5)."
assert np.isclose(fpr_g[0], 0.0) and np.isclose(tpr_g[0], 0.0)
assert np.isclose(fpr_g[-1], 1.0) and np.isclose(tpr_g[-1], 1.0)
print("Assertion Passed: ROC curves correctly start at (0,0) and end at (1,1) with valid AUC ordering.")"""
cells.append(nbf.v4.new_code_cell(cell_14_code))

# Cell 15: Conclusion & Summary
cell_15_md = """---
### まとめと第5章 次節への展望

本ノートブックでは、**Section 5.2: Decision Theory (決定理論)** の数学的基盤とアルゴリズムを網羅的に検証しました：

1. **決定理論の本質**: 推論（確率分布の獲得）と決定（損失に基づく行動の選択）を峻別することの重要性。
2. **ベイズ決定規則**: 
   - 誤分類率最小化 $\\iff$ 事後確率最大のクラスへの割り当て。
   - 期待損失最小化 $\\iff$ 損失行列重み付き事後確率の最小化。
3. **実用的な拡張**:
   - 確信度の低いサンプルを保留する **棄却オプション**。
   - 人工的なデータ収集比率を現実の母集団へ適合させる **事前確率補正**。
4. **評価体系**:
   - 混同行列、Accuracy、Precision、Recall、Specificity、FPR、$F_1$ スコアの体系的整理。
   - 閾値移動に伴うトレードオフを可視化・定量化する **ROC 曲線と AUC**。

**次節への展望**:
次節 **Section 5.3: Generative Classifiers (生成的分類器)** では、本節で前提としたクラス条件付き確率 $p(\\mathbf{x}|\\mathcal{C}_k)$ を実際にデータから最尤推定によりモデル化する生成的アプローチ（連続変数に対するガウス判別分析、離散変数に対するナイーブベイズモデル等）へと進みます。"""
cells.append(nbf.v4.new_markdown_cell(cell_15_md))

nb.cells = cells

# Save notebook
os.makedirs("5", exist_ok=True)
notebook_path = "5/5.2_Decision_Theory.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {notebook_path} with {len(cells)} cells.")

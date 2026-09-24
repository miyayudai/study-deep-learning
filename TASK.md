# Deep Learning: Foundations and Concepts (Bishop & Bishop 2024) AI 実装・学習タスクリスト

このドキュメントは、教科書『深層学習：基礎と概念 (Deep Learning: Foundations and Concepts, Christopher M. Bishop & Hugh Bishop 著, Springer 2024)』の理論解説、数式厳密導出、Python実装、図版（Figure）再現、および演習問題（Exercises）の作成・検証を自動化・管理するためのタスク定義ファイルです。

AIエージェントはこのファイルを読み込み、以下の「AIエージェント向け厳格実行ルール（疎化防止・高品質保証）」に従ってタスクを1つずつ進行してください。

---

## タスク状態管理と排他制御（3段階ステータス制・重複防止機構）

本プロジェクトでは、タスクの重複着手やユーザー確認中の意図しない先回り進行を防止するため、**3段階ステータス管理**および**排他制御（Active Lock）**を採用しています。

### 3段階ステータス定義
- `- [ ]` : **未着手 (Pending)** - まだ作業が開始されていない状態。
- `- [-]` : **実行中 (In Progress / ロック中)** - 現在AIエージェントが作業中、またはユーザーの確認・レビュー待ちの状態。**このタスク以外の新規タスク着手は厳禁。**
- `- [x]` : **完了 (Completed)** - Definition of Done (DoD) をすべて満たし、テスト・全セル実行・ユーザー検証が完了した状態。

---

## AIエージェント向け厳格実行ルール（内容の疎化防止・排他制御12原則）

1. **タスクの選択と排他ロック取得（ロック取得の絶対原則）**:
   - 作業開始前に必ず後述の「現在の実行ステータス (Active Lock)」を確認してください。
   - Active Lock が「なし」の場合のみ、「Task Queue」から未着手（`- [ ]`）の最も番号の若いタスクを**1つだけ**選択してください。
   - **他の作業（リサーチやコード生成）を行う前に、まず `TASK.md` の対象タスクを `- [ ]` から `- [-]`（実行中）に更新し、Active Lock欄に対象タスク名を記録して保存してください（排他ロックの取得）**。
   - 既に `[-]` になっているタスクが存在する場合、またはユーザーへの確認待ちの間は、絶対に新規タスクに着手してはなりません。
2. **実行単位の極小化と自律パイプライン進行（1節ずつの確実なDoD完了と先回り進行原則）**【最重要】:
   - **1回の実行ステップで進めるのは、原則「1つの節（Section）」または「その章の演習問題（Exercises）」**としてください。「1章まるごと」を一気にまとめて処理することは厳禁です（内容が疎化するため）。
   - **先回り進行・連続自律実行の許可**: ユーザーからの明示的な停止指示がない限り、1つの節がDoD（全小節網羅、数式導出、図版生成、全テストパス、ノートブック全セルエラーゼロ実行）をすべて満たして完了（`[x]`）したら、Active Lockを更新し、**自動的に次の未着手タスク（`- [ ]`）を `- [-]` にロックして自律的に次の節の実装・検証に進んでよい**ものとします。
   - **重複着手の絶対禁止**: 既に `- [-]`（実行中）となっているタスクや、他のプロセスが着手中のタスクに二重に着手すること、および同じタスクを重複して実行することは一切禁止します。
3. **小節（Subsection）の完全網羅と個別見出し**:
   - 対象の節に含まれる**すべての小節（例: 4.1.1, 4.1.2...）を必ずJupyter Notebook内の見出し（Markdownセル `###`）として独立して設けてください**。
   - 複数の小節を1つに統合したり、内容を省略してスキップすることは禁止します。
4. **数式展開の途中計算（Derivation Steps）の徹底記述**:
   - 公式の結論（最終式）だけを天下り式に書くことを禁止します。
   - 教科書の式番号（例: 式(4.12)）を明記し、「なぜこの式から次の式に変形できるのか」「どのような微分の連鎖律や代数的変形、統計的仮定を用いているのか」を、数式（LaTeX）と日本語でステップ・バイ・ステップで解説してください。
5. **教科書の図（Figure）の完全再現・可視化（必須の完了条件 / DoD）**:
   - 対象の節に登場する**教科書の図（Figure X.X）は、matplotlib/seaborn等を用いて必ず忠実に再現・プロットしてください**。
   - 再現したプロットは各章の `result/` ディレクトリに `fig{chapter}_{figure}_{name}.png` 形式で高解像度保存し、ノートブック上にも表示させてください。
   - **「図を再現する」には背後の数式やサンプリングを正しく実装しなければならないため、これが内容の疎化を防ぐ最大の防御壁となります**。
6. **実装と理論解説の統合 (Jupyter Notebook)**:
   - 節ごとに1つの独立したJupyter Notebook（例: `4/4.1_Linear_Regression.ipynb`）を作成してください。
   - 理論解説と軽量な検証コードをノートブック内に統合し、学習ループや再利用クラス・関数は `common/` ディレクトリにモジュール化してNotebookからimportしてください。
7. **共通モジュール化 (`common/`) と依存関係管理**:
   - プロットスタイル設定（`common/plot_utils.py`）、数学ユーティリティ、モデル定義などは `common/` に集約し、重複のないコードベースを維持してください。
   - 外部ライブラリを追加した場合はルートの `requirements.txt` を更新してください。
8. **演習問題（Exercises）の完全網羅（穴埋め・選択形式）**:
   - 章末の `Exercises` は専用ノートブック（例: `4/4_Exercises.ipynb`）として作成してください。
   - **理論・証明問題**: 証明の道筋・ロジックステップをあらかじめ提示した上で、重要な数式変形やキーワードを「穴埋め」または「選択問題」にして出題してください。
   - **実装・数値検証問題**: 核となるコードを適度に穴埋め（`None` や `# YOUR CODE HERE`）にし、直後に自己検証可能なテストセル（`assert` による検証）を配置してください。
   - 全問について解説と解答、数値検証コードを完備してください。
9. **ユニットテストの作成と検証 (pytest)**:
   - コアアルゴリズムや数学的性質（勾配の一致、直交性、損失の単調減少、確率分布の正規化など）を検証するユニットテストを `tests/` に作成してください。
   - タスク完了のチェックを入れる前に、必ず `pytest` を実行し、全件パスすることを確認してください。
10. **ノートブックの全セル実行・エラーゼロ確認**:
    - 作成したJupyter Notebookは未実行セルを残さず、全セルが正常に実行され、エラー（`output_type == "error"`）が一切ないことを確認してください。
11. **タスク完了の定義（Definition of Done: DoD）とロック解除**:
    - 以下の条件をすべて満たした場合にのみ、`TASK.md` の該当タスクを `- [-]` から `- [x]` に更新し、Active Lockを「なし」に戻してください：
      1. [ ] 該当節の全小節（Subsections）の解説と数式導出がNotebookに記載されている
      2. [ ] 該当節のすべての教科書図版（Figure）が忠実に再現され、`result/` に保存されている
      3. [ ] コアアルゴリズムのPythonコードが動作し、可視化されている
      4. [ ] ユニットテストが `tests/` に追加され、`pytest` でパスしている
      5. [ ] Notebookの全セルが実行済みであり、エラーがない
12. **バージョン管理 (Git Commit)**:
    - 1つの節または演習問題が完了したら、`git add .`、`git commit -m "feat(chX): complete Section X.X with figures and tests"` を実行して記録してください。

---

## 全章構成・進捗概要 (Progress Summary)

| Chapter | タイトル | 節数 | 小節数 | 教科書図版 | 演習問題 (Exercises) | 状態 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **第1章** | 深層学習革命 (The Deep Learning Revolution) | 3 | 13 | Figure 1.1 〜 1.16 (全16枚) | なし (Tutorial) | 完了 (`[x]`) |
| **第2章** | 確率の基礎 (Probabilities) | 6 | 23 | Figure 2.1 〜 2.16 (全16枚) | 全41問 | 完了 (`[x]`) |
| **第3章** | 基本分布 (Standard Distributions) | 5 | 17 | Figure 3.1 〜 3.16 (全16枚) | 全38問 | 完了 (`[x]`) |
| **第4章** | 単層ネットワーク: 回帰 (Single-layer Networks: Regression) | 3 | 7 | Figure 4.1 〜 4.8 (全8枚) | 全12問 | 完了 (`[x]`) |
| **第5章** | 単層ネットワーク: 分類 (Single-layer Networks: Classification) | 4 | 20 | Figure 5.1 〜 5.17 (全17枚) | 全24問 | 完了 (`[x]`) |
| **第6章** | 深層ニューラルネットワーク (Deep Neural Networks) | 5 | 22 | Figure 6.1 〜 6.19 (全19枚) | 全21問 | 完了 (`[x]`) |
| **第7章** | 勾配降下法 (Gradient Descent) | 4 | 15 | Figure 7.1 〜 7.10 (全10枚) | 全14問 | 完了 (`[x]`) |
| **第8章** | 誤差逆伝播法 (Backpropagation) | 2 | 8 | Figure 8.1 〜 8.5 (全5枚) | 全18問 | 完了 (`[x]`) |
| **第9章** | 正則化 (Regularization) | 6 | 10 | Figure 9.1 〜 9.17 (全17枚) | 全18問 | 完了 (`[x]`) |
| **第10章** | 畳み込みネットワーク (Convolutional Networks) | 6 | 24 | Figure 10.1 〜 10.32 (全32枚) | 全13問 | 完了 (`[x]`) |
| **第11章** | 構造化分布 (Structured Distributions) | 3 | 15 | Figure 11.1 〜 11.32 (全32枚) | 全20問 | 完了 (`[x]`) |
| **第12章** | トランスフォーマー (Transformers) | 4 | 25 | Figure 12.1 〜 12.27 (全27枚) | 全16問 | 進行中 (`[-]`) |
| **第13章** | グラフニューラルネットワーク (Graph Neural Networks) | 3 | 16 | Figure 13.1 〜 13.5 (全5枚) | 全10問 | 未着手 |
| **第14章** | サンプリング (Sampling) | 4 | 12 | Figure 14.1 〜 14.13 (全13枚) | 全17問 | 未着手 |
| **第15章** | 離散潜在変数 (Discrete Latent Variables) | 3 | 17 | Figure 15.1 〜 15.19 (全19枚) | 全18問 | 未着手 |
| **第16章** | 連続潜在変数 (Continuous Latent Variables) | 4 | 16 | Figure 16.1 〜 16.29 (全29枚) | 全23問 | 未着手 |
| **第17章** | 生成対抗ネットワーク (Generative Adversarial Networks) | 4 | 12 | Figure 17.1 〜 17.13 (全13枚) | 全12問 | 未着手 |
| **第18章** | 正規化フロー (Normalizing Flows) | 3 | 3 | Figure 18.1 〜 18.7 (全7枚) | 全11問 | 未着手 |
| **第19章** | 自己符号化器 (Autoencoders) | 2 | 7 | Figure 19.1 〜 19.11 (全11枚) | 全6問 | 未着手 |
| **第20章** | 拡散モデル (Diffusion Models) | 4 | 13 | Figure 20.1 〜 20.9 (全9枚) | 全20問 | 未着手 |
| **合計** | 全20章 + 付録 | **76節** | **281小節** | **約300枚** | **全355問** | **63 / 96 ユニット完了** |

---

## 現在の実行ステータス (Active Lock)
- **現在実行中のタスク**: 13.2 Neural Message-Passing
- **担当セクション**: 第13章 13.2 ニューラル・メッセージパッシング (Neural Message-Passing)
- **ステータス**: 実行中 (`[-]`)
- **獲得ロックタイムスタンプ**: 2026-09-24T11:17:00+09:00
- **直前完了タスク**: 第13章 13.1 グラフ上の機械学習 (Machine Learning on Graphs) (2026-09-24 完了)
※上記が「なし」以外の場合、新しいタスクの開始・割り込みは厳禁（排他ロック中）。

---

## Task Queue

### 第1章 深層学習革命 (The Deep Learning Revolution)

- ディレクトリ: `1/`
- 再現図版目標: Figure 1.1 〜 1.16 (全16枚)

- [x] **1.1 The Impact of Deep Learning**
  - ノートブック: `1/1.1_The_Impact_of_Deep_Learning.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 1.1.1 Medical diagnosis
    - [x] 1.1.2 Protein structure
    - [x] 1.1.3 Image synthesis
    - [x] 1.1.4 Large language models
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 1.1, 1.2, 1.3)
  - テスト: `tests/test_ch1_the_impact_of_d.py`

- [x] **1.2 A Tutorial Example**
  - ノートブック: `1/1.2_A_Tutorial_Example.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 1.2.1 Synthetic data
    - [x] 1.2.2 Linear models
    - [x] 1.2.3 Error function
    - [x] 1.2.4 Model complexity
    - [x] 1.2.5 Regularization
    - [x] 1.2.6 Model selection
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, Table 1.1, 1.2)
  - テスト: `tests/test_ch1_a_tutorial_exam.py`

- [x] **1.3 A Brief History of Machine Learning**
  - ノートブック: `1/1.3_A_Brief_History_of_Machine_Learning.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 1.3.1 Single-layer networks
    - [x] 1.3.2 Backpropagation
    - [x] 1.3.3 Deep networks
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 1.12, 1.13, 1.14, 1.15, 1.16)
  - テスト: `tests/test_ch1_a_brief_history.py`

---

### 第2章 確率の基礎 (Probabilities)

- ディレクトリ: `2/`
- 再現図版目標: Figure 2.1 〜 2.16 (全16枚)

- [x] **2.1 The Rules of Probability**
  - ノートブック: `2/2.1_The_Rules_of_Probability.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 2.1.1 A medical screening example
    - [x] 2.1.2 The sum and product rules
    - [x] 2.1.3 Bayes’ theorem
    - [x] 2.1.4 Medical screening revisited
    - [x] 2.1.5 Prior and posterior probabilities
    - [x] 2.1.6 Independent variables
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 2.1, 2.2, 2.3, 2.4, 2.5)
  - テスト: `tests/test_ch2_the_rules_of_pr.py`

- [x] **2.2 Probability Densities**
  - ノートブック: `2/2.2_Probability_Densities.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 2.2.1 Example distributions
    - [x] 2.2.2 Expectations and covariances
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 2.6, 2.7)
  - テスト: `tests/test_ch2_probability_den.py`

- [x] **2.3 The Gaussian Distribution**
  - ノートブック: `2/2.3_The_Gaussian_Distribution.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 2.3.1 Mean and variance
    - [x] 2.3.2 Likelihood function
    - [x] 2.3.3 Bias of maximum likelihood
    - [x] 2.3.4 Linear regression
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 2.8, 2.9, 2.10, 2.11)
  - テスト: `tests/test_ch2_the_gaussian_di.py`

- [x] **2.4 Transformation of Densities**
  - ノートブック: `2/2.4_Transformation_of_Densities.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 2.4.1 Multivariate distributions
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 2.12, 2.13)
  - テスト: `tests/test_ch2_transformation_.py`

- [x] **2.5 Information Theory**
  - ノートブック: `2/2.5_Information_Theory.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 2.5.1 Entropy
    - [x] 2.5.2 Physics perspective
    - [x] 2.5.3 Differential entropy
    - [x] 2.5.4 Maximum entropy
    - [x] 2.5.5 Kullback–Leibler divergence
    - [x] 2.5.6 Conditional entropy
    - [x] 2.5.7 Mutual information
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 2.14, 2.15)
  - テスト: `tests/test_ch2_information_the.py`

- [x] **2.6 Bayesian Probabilities**
  - ノートブック: `2/2.6_Bayesian_Probabilities.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 2.6.1 Model parameters
    - [x] 2.6.2 Regularization
    - [x] 2.6.3 Bayesian machine learning
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 2.16)
  - テスト: `tests/test_ch2_bayesian_probab.py`

- [x] **第2章 演習問題 (Exercises 2.1 〜 2.41, 全41問)**
  - ノートブック: `2/2_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch2_exercises.py`

---

### 第3章 基本分布 (Standard Distributions)

- ディレクトリ: `3/`
- 再現図版目標: Figure 3.1 〜 3.16 (全16枚)

- [x] **3.1 Discrete Variables**
  - ノートブック: `3/3.1_Discrete_Variables.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 3.1.1 Bernoulli distribution
    - [x] 3.1.2 Binomial distribution
    - [x] 3.1.3 Multinomial distribution
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 3.1)
  - テスト: `tests/test_ch3_discrete_variab.py`

- [x] **3.2 The Multivariate Gaussian**
  - ノートブック: `3/3.2_The_Multivariate_Gaussian.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 3.2.1 Geometry of the Gaussian
    - [x] 3.2.2 Moments
    - [x] 3.2.3 Limitations
    - [x] 3.2.4 Conditional distribution
    - [x] 3.2.5 Marginal distribution
    - [x] 3.2.6 Bayes’ theorem
    - [x] 3.2.7 Maximum likelihood
    - [x] 3.2.8 Sequential estimation
    - [x] 3.2.9 Mixtures of Gaussians
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 3.2 〜 3.8)
  - テスト: `tests/test_ch3_the_multivariat.py`

- [x] **3.3 Periodic Variables**
  - ノートブック: `3/3.3_Periodic_Variables.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 3.3.1 Von Mises distribution
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 3.9 〜 3.12)
  - テスト: `tests/test_ch3_periodic_variab.py`

- [x] **3.4 The Exponential Family**
  - ノートブック: `3/3.4_The_Exponential_Family.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 3.4.1 Sufficient statistics
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 3.13 〜 3.15)
  - テスト: `tests/test_ch3_the_exponential.py`

- [x] **3.5 Nonparametric Methods**
  - ノートブック: `3/3.5_Nonparametric_Methods.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 3.5.1 Histograms
    - [x] 3.5.2 Kernel densities
    - [x] 3.5.3 Nearest-neighbours
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 3.13 〜 3.16)
  - テスト: `tests/test_ch3_nonparametric_m.py`

- [x] **第3章 演習問題 (Exercises 3.1 〜 3.38, 全38問)**
  - ノートブック: `3/3_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch3_exercises.py`

---

### 第4章 単層ネットワーク: 回帰 (Single-layer Networks: Regression)

- ディレクトリ: `4/`
- 再現図版目標: Figure 4.1 〜 4.8 (全8枚)

- [x] **4.1 Linear Regression**
  - ノートブック: `4/4.1_Linear_Regression.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 4.1.1 Basis functions
    - [x] 4.1.2 Likelihood function
    - [x] 4.1.3 Maximum likelihood
    - [x] 4.1.4 Geometry of least squares
    - [x] 4.1.5 Sequential learning
    - [x] 4.1.6 Regularized least squares
    - [x] 4.1.7 Multiple outputs
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 4.1 〜 4.4)
  - テスト: `tests/test_ch4_linear_regressi.py`

- [x] **4.2 Decision theory**
  - ノートブック: `4/4.2_Decision_theory.ipynb`
  - 小節: （単独構成）
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 4.5 〜 4.6)
  - テスト: `tests/test_ch4_decision_theory.py`

- [x] **4.3 The Bias–Variance Trade-off**
  - ノートブック: `4/4.3_The_Bias_Variance_Trade_off.ipynb`
  - 小節: （単独構成）
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 4.7 〜 4.8)
  - テスト: `tests/test_ch4_the_bias_varian.py`

- [x] **第4章 演習問題 (Exercises 4.1 〜 4.12, 全12問)**
  - ノートブック: `4/4_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch4_exercises.py`

---

### 第5章 単層ネットワーク: 分類 (Single-layer Networks: Classification)

- ディレクトリ: `5/`
- 再現図版目標: Figure 5.1 〜 5.17 (全17枚)

- [x] **5.1 Discriminant Functions**
  - ノートブック: `5/5.1_Discriminant_Functions.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 5.1.1 Two classes
    - [x] 5.1.2 Multiple classes
    - [x] 5.1.3 1-of-K coding
    - [x] 5.1.4 Least squares for classification
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 5.1 〜 5.4)
  - テスト: `tests/test_ch5_discriminant_fu.py`

- [x] **5.2 Decision Theory**
  - ノートブック: `5/5.2_Decision_Theory.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 5.2.1 Misclassification rate
    - [x] 5.2.2 Expected loss
    - [x] 5.2.3 The reject option
    - [x] 5.2.4 Inference and decision
    - [x] 5.2.5 Classifier accuracy
    - [x] 5.2.6 ROC curve
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 5.5 〜 5.11)
  - テスト: `tests/test_ch5_decision_theory.py`

- [x] **5.3 Generative Classifiers**
  - ノートブック: `5/5.3_Generative_Classifiers.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 5.3.1 Continuous inputs
    - [x] 5.3.2 Maximum likelihood solution
    - [x] 5.3.3 Discrete features
    - [x] 5.3.4 Exponential family
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 5.12 〜 5.14)
  - テスト: `tests/test_ch5_generative_clas.py`

- [x] **5.4 Discriminative Classifiers**
  - ノートブック: `5/5.4_Discriminative_Classifiers.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 5.4.1 Activation functions
    - [x] 5.4.2 Fixed basis functions
    - [x] 5.4.3 Logistic regression
    - [x] 5.4.4 Multi-class logistic regression
    - [x] 5.4.5 Probit regression
    - [x] 5.4.6 Canonical link functions
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 5.15 〜 5.17)
  - テスト: `tests/test_ch5_discriminative_classifiers.py`

- [x] **第5章 演習問題 (Exercises 5.1 〜 5.24, 全24問)**
  - ノートブック: `5/5_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch5_exercises.py`

---

### 第6章 深層ニューラルネットワーク (Deep Neural Networks)

- ディレクトリ: `6/`
- 再現図版目標: Figure 6.1 〜 6.19 (全19枚)

- [x] **6.1 Limitations of Fixed Basis Functions**
  - ノートブック: `6/6.1_Limitations_of_Fixed_Basis_Functions.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 6.1.1 The curse of dimensionality
    - [x] 6.1.2 High-dimensional spaces
    - [x] 6.1.3 Data manifolds
    - [x] 6.1.4 Data-dependent basis functions
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 6.1 〜 6.8)
  - テスト: `tests/test_ch6_limitations_of_fixed_basis_functions.py`

- [x] **6.2 Multilayer Networks**
  - ノートブック: `6/6.2_Multilayer_Networks.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 6.2.1 Parameter matrices
    - [x] 6.2.2 Universal approximation
    - [x] 6.2.3 Hidden unit activation functions
    - [x] 6.2.4 Weight-space symmetries
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 6.9 〜 6.12)
  - テスト: `tests/test_ch6_multilayer_netw.py`

- [x] **6.3 Deep Networks**
  - ノートブック: `6/6.3_Deep_Networks.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 6.3.1 Hierarchical representations
    - [x] 6.3.2 Distributed representations
    - [x] 6.3.3 Representation learning
    - [x] 6.3.4 Transfer learning
    - [x] 6.3.5 Contrastive learning
    - [x] 6.3.6 General network architectures
    - [x] 6.3.7 Tensors
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 6.13 〜 6.15)
  - テスト: `tests/test_ch6_deep_networks.py`

- [x] **6.4 Error Functions**
  - ノートブック: `6/6.4_Error_Functions.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 6.4.1 Regression
    - [x] 6.4.2 Binary classification
    - [x] 6.4.3 multiclass classification
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 6 Overview)
  - テスト: `tests/test_ch6_error_functions.py`

- [x] **6.5 Mixture Density Networks**
  - ノートブック: `6/6.5_Mixture_Density_Networks.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 6.5.1 Robot kinematics example
    - [x] 6.5.2 Conditional mixture distributions
    - [x] 6.5.3 Gradient optimization
    - [x] 6.5.4 Predictive distribution
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 6.16 〜 6.19)
  - テスト: `tests/test_ch6_mixture_density.py`

- [x] **第6章 演習問題 (Exercises 6.1 〜 6.21, 全21問)**
  - ノートブック: `6/6_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch6_exercises.py`

---

### 第7章 勾配降下法 (Gradient Descent)

- ディレクトリ: `7/`
- 再現図版目標: Figure 7.1 〜 7.8 (全8枚)

- [x] **7.1 Error Surfaces**
  - ノートブック: `7/7.1_Error_Surfaces.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 7.1.1 Local quadratic approximation
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 7.1, 7.2)
  - テスト: `tests/test_ch7_error_surfaces.py`

- [x] **7.2 Gradient Descent Optimization**
  - ノートブック: `7/7.2_Gradient_Descent_Optimization.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 7.2.1 Use of gradient information
    - [x] 7.2.2 Batch gradient descent
    - [x] 7.2.3 Stochastic gradient descent
    - [x] 7.2.4 Mini-batches
    - [x] 7.2.5 Parameter initialization
  - 図版再現: `result/` への該当Figureプロット保存 (fig_7_2_minibatch_noise, fig_7_2_variance_propagation, fig_7_2_gd_trajectories)
  - テスト: `tests/test_ch7_gradient_descen.py`

- [x] **7.3 Convergence**
  - ノートブック: `7/7.3_Convergence.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 7.3.1 Momentum
    - [x] 7.3.2 Learning rate schedule
    - [x] 7.3.3 RMSProp and Adam
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 7.3, 7.4, 7.5, 7.6)
  - テスト: `tests/test_ch7_convergence.py`

- [x] **7.4 Normalization**
  - ノートブック: `7/7.4_Normalization.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 7.4.1 Data normalization
    - [x] 7.4.2 Batch normalization
    - [x] 7.4.3 Layer normalization
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 7.7, 7.8)
  - テスト: `tests/test_ch7_normalization.py`

- [x] **第7章 演習問題 (Exercises 7.1 〜 7.14, 全14問)**
  - ノートブック: `7/7_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch7_exercises.py`

---

### 第8章 誤差逆伝播法 (Backpropagation)

- ディレクトリ: `8/`
- 再現図版目標: Figure 8.1 〜 8.5 (全5枚)

- [x] **8.1 Evaluation of Gradients**
  - ノートブック: `8/8.1_Evaluation_of_Gradients.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 8.1.1 Single-layer networks
    - [x] 8.1.2 General feed-forward networks
    - [x] 8.1.3 A simple example
    - [x] 8.1.4 Numerical differentiation
    - [x] 8.1.5 The Jacobian matrix
    - [x] 8.1.6 The Hessian matrix
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 8.1, 8.2, 8.3)
  - テスト: `tests/test_ch8_evaluation_of_gradients.py`

- [x] **8.2 Automatic Differentiation**
  - ノートブック: `8/8.2_Automatic_Differentiation.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 8.2.1 Forward-mode automatic differentiation
    - [x] 8.2.2 Reverse-mode automatic differentiation
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 8.4, 8.5)
  - テスト: `tests/test_ch8_automatic_differentiation.py`

- [x] **第8章 演習問題 (Exercises 8.1 〜 8.18, 全18問)**
  - ノートブック: `8/8_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch8_exercises.py`

---

### 第9章 正則化 (Regularization)

- ディレクトリ: `9/`
- 再現図版目標: Figure 9.1 〜 9.17 (全17枚)

- [x] **9.1 Inductive Bias**
  - ノートブック: `9/9.1_Inductive_Bias.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 9.1.1 Inverse problems
    - [x] 9.1.2 No free lunch theorem
    - [x] 9.1.3 Symmetry and invariance
    - [x] 9.1.4 Equivariance
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 9.1, 9.2)
  - テスト: `tests/test_ch9_inductive_bias.py`

- [x] **9.2 Weight Decay**
  - ノートブック: `9/9.2_Weight_Decay.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 9.2.1 Consistent regularizers
    - [x] 9.2.2 Generalized weight decay
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 9.3, 9.4, 9.5, 9.6)
  - テスト: `tests/test_ch9_weight_decay.py`

- [x] **9.3 Learning Curves**
  - ノートブック: `9/9.3_Learning_Curves.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 9.3.1 Early stopping
    - [x] 9.3.2 Double descent
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 9.7, 9.8, 9.9, 9.10, 9.11)
  - テスト: `tests/test_ch9_learning_curves.py`

- [x] **9.4 Parameter Sharing**
  - ノートブック: `9/9.4_Parameter_Sharing.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 9.4.1 Soft weight sharing
  - 図版再現: `result/` への該当Figureプロット保存 (fig_9_soft_weight_sharing_prior, fig_9_soft_weight_sharing_clustering, fig_9_hard_vs_soft_comparison)
  - テスト: `tests/test_ch9_parameter_shari.py`

- [x] **9.5 Residual Connections**
  - ノートブック: `9/9.5_Residual_Connections.ipynb`
  - 小節: （単独構成）
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 9.12, 9.13, 9.14, 9.15, 9.16)
  - テスト: `tests/test_ch9_residual_connec.py`

- [x] **9.6 Model Averaging**
  - ノートブック: `9/9.6_Model_Averaging.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 9.6.1 Dropout
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 9.17)
  - テスト: `tests/test_ch9_model_averaging.py`

- [x] **第9章 演習問題 (Exercises 9.1 〜 9.18, 全18問)**
  - ノートブック: `9/9_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch9_exercises.py`

---

### 第10章 畳み込みネットワーク (Convolutional Networks)

- ディレクトリ: `10/`
- 再現図版目標: Figure 10.1 〜 10.32 (全32枚)

- [x] **10.1 Computer Vision**
  - ノートブック: `10/10.1_Computer_Vision.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 10.1.1 Image data
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch10_computer_vision.py`

- [x] **10.2 Convolutional Filters**
  - ノートブック: `10/10.2_Convolutional_Filters.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 10.2.1 Feature detectors
    - [x] 10.2.2 Translation equivariance
    - [x] 10.2.3 Padding
    - [x] 10.2.4 Strided convolutions
    - [x] 10.2.5 Multi-dimensional convolutions
    - [x] 10.2.6 Pooling
    - [x] 10.2.7 Multilayer convolutions
    - [x] 10.2.8 Example network architectures
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 10.1 〜 10.10 全10枚)
  - テスト: `tests/test_ch10_convolutional_f.py`

- [x] **10.3 Visualizing Trained CNNs**
  - ノートブック: `10/10.3_Visualizing_Trained_CNNs.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 10.3.1 Visual cortex
    - [x] 10.3.2 Visualizing trained filters
    - [x] 10.3.3 Saliency maps
    - [x] 10.3.4 Adversarial attacks
    - [x] 10.3.5 Synthetic images
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 10.11 〜 10.18 全8枚)
  - テスト: `tests/test_ch10_visualizing_tra.py`

- [x] **10.4 Object Detection**
  - ノートブック: `10/10.4_Object_Detection.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 10.4.1 Bounding boxes
    - [x] 10.4.2 Intersection-over-union
    - [x] 10.4.3 Sliding windows
    - [x] 10.4.4 Detection across scales
    - [x] 10.4.5 Non-max suppression
    - [x] 10.4.6 Fast region CNNs
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 10.19 〜 10.25 全7枚)
  - テスト: `tests/test_ch10_object_detectio.py`

- [x] **10.5 Image Segmentation**
  - ノートブック: `10/10.5_Image_Segmentation.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 10.5.1 Convolutional segmentation
    - [x] 10.5.2 Up-sampling
    - [x] 10.5.3 Fully convolutional networks
    - [x] 10.5.4 The U-net architecture
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 10.26 〜 10.31 全6枚)
  - テスト: `tests/test_ch10_image_segmentat.py`

- [x] **10.6 Style Transfer**
  - ノートブック: `10/10.6_Style_Transfer.ipynb`
  - 小節: （単独構成）
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 10.32 全1枚)
  - テスト: `tests/test_ch10_style_transfer.py`

- [x] **第10章 演習問題 (Exercises 10.1 〜 10.13, 全13問)**
  - ノートブック: `10/10_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch10_exercises.py`

---

### 第11章 構造化分布 (Structured Distributions)

- ディレクトリ: `11/`
- 再現図版目標: Figure 11.1 〜 11.32 (全32枚)

- [x] **11.1 Graphical Models**
  - ノートブック: `11/11.1_Graphical_Models.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 11.1.1 Directed graphs
    - [x] 11.1.2 Factorization
    - [x] 11.1.3 Discrete variables
    - [x] 11.1.4 Gaussian variables
    - [x] 11.1.5 Binary classifier
    - [x] 11.1.6 Parameters and observations
    - [x] 11.1.7 Bayes’ theorem
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 11.1 〜 11.13 全13枚)
  - テスト: `tests/test_ch11_graphical_model.py`

- [x] **11.2 Conditional Independence**
  - ノートブック: `11/11.2_Conditional_Independence.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 11.2.1 Three example graphs
    - [x] 11.2.2 Explaining away
    - [x] 11.2.3 D-separation
    - [x] 11.2.4 Naive Bayes
    - [x] 11.2.5 Generative models
    - [x] 11.2.6 Markov blanket
    - [x] 11.2.7 Graphs as filters
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 11.14 〜 11.26 全13枚)
  - テスト: `tests/test_ch11_conditional_ind.py`

- [x] **11.3 Sequence Models**
  - ノートブック: `11/11.3_Sequence_Models.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 11.3.1 Hidden variables
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 11.27 〜 11.31 全5枚)
  - テスト: `tests/test_ch11_sequence_models.py`

- [x] **第11章 演習問題 (Exercises 11.1 〜 11.20, 全20問)**
  - ノートブック: `11/11_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 11.32)
  - テスト: `tests/test_ch11_exercises.py`

---

### 第12章 トランスフォーマー (Transformers)

- ディレクトリ: `12/`
- 再現図版目標: Figure 12.1 〜 12.27 (全27枚)

- [x] **12.1 Attention**
  - ノートブック: `12/12.1_Attention.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 12.1.1 Transformer processing
    - [x] 12.1.2 Attention coefficients
    - [x] 12.1.3 Self-attention
    - [x] 12.1.4 Network parameters
    - [x] 12.1.5 Scaled self-attention
    - [x] 12.1.6 Multi-head attention
    - [x] 12.1.7 Transformer layers
    - [x] 12.1.8 Computational complexity
    - [x] 12.1.9 Positional encoding
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 12.1 〜 12.10 全10枚)
  - テスト: `tests/test_ch12_attention.py`

- [x] **12.2 Natural Language**
  - ノートブック: `12/12.2_Natural_Language.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 12.2.1 Word embedding
    - [x] 12.2.2 Tokenization
    - [x] 12.2.3 Bag of words
    - [x] 12.2.4 Autoregressive models
    - [x] 12.2.5 Recurrent neural networks
    - [x] 12.2.6 Backpropagation through time
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 12.11 〜 12.14 全4枚)
  - テスト: `tests/test_ch12_natural_languag.py`

- [x] **12.3 Transformer Language Models**
  - ノートブック: `12/12.3_Transformer_Language_Models.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 12.3.1 Decoder transformers
    - [x] 12.3.2 Sampling strategies
    - [x] 12.3.3 Encoder transformers
    - [x] 12.3.4 Sequence-to-sequence transformers
    - [x] 12.3.5 Large language models
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 12.15 〜 12.21 全7枚)
  - テスト: `tests/test_ch12_transformer_lan.py`

- [x] **12.4 Multimodal Transformers**
  - ノートブック: `12/12.4_Multimodal_Transformers.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 12.4.1 Vision transformers
    - [x] 12.4.2 Generative image transformers
    - [x] 12.4.3 Audio data
    - [x] 12.4.4 Text-to-speech
    - [x] 12.4.5 Vision and language transformers
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 12.22 〜 12.27 全6枚)
  - テスト: `tests/test_ch12_multimodal_tran.py`

- [x] **第12章 演習問題 (Exercises 12.1 〜 12.16, 全16問)**
  - ノートブック: `12/12_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch12_exercises.py`

---

### 第13章 グラフニューラルネットワーク (Graph Neural Networks)

- ディレクトリ: `13/`
- 再現図版目標: Figure 13.1 〜 13.5 (全5枚)

- [x] **13.1 Machine Learning on Graphs**
  - ノートブック: `13/13.1_Machine_Learning_on_Graphs.ipynb`
  - 小節一覧（網羅必須）:
    - [x] 13.1.1 Graph properties
    - [x] 13.1.2 Adjacency matrix
    - [x] 13.1.3 Permutation equivariance
  - 図版再現: `result/` への該当Figureプロット保存 (Figure 13.1 〜 13.2 全2枚)
  - テスト: `tests/test_ch13_machine_learnin.py`

- [-] **13.2 Neural Message-Passing**
  - ノートブック: `13/13.2_Neural_Message_Passing.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 13.2.1 Convolutional filters
    - [ ] 13.2.2 Graph convolutional networks
    - [ ] 13.2.3 Aggregation operators
    - [ ] 13.2.4 Update operators
    - [ ] 13.2.5 Node classification
    - [ ] 13.2.6 Edge classification
    - [ ] 13.2.7 Graph classification
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch13_neural_message_.py`

- [ ] **13.3 General Graph Networks**
  - ノートブック: `13/13.3_General_Graph_Networks.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 13.3.1 Graph attention networks
    - [ ] 13.3.2 Edge embeddings
    - [ ] 13.3.3 Graph embeddings
    - [ ] 13.3.4 Over-smoothing
    - [ ] 13.3.5 Regularization
    - [ ] 13.3.6 Geometric deep learning
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch13_general_graph_n.py`

- [ ] **第13章 演習問題 (Exercises 13.1 〜 13.10, 全10問)**
  - ノートブック: `13/13_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch13_exercises.py`

---

### 第14章 サンプリング (Sampling)

- ディレクトリ: `14/`
- 再現図版目標: Figure 14.1 〜 14.14 (全14枚)

- [ ] **14.1 Basic Sampling Algorithms**
  - ノートブック: `14/14.1_Basic_Sampling_Algorithms.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 14.1.1 Expectations
    - [ ] 14.1.2 Standard distributions
    - [ ] 14.1.3 Rejection sampling
    - [ ] 14.1.4 Adaptive rejection sampling
    - [ ] 14.1.5 Importance sampling
    - [ ] 14.1.6 Sampling-importance-resampling
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch14_basic_sampling_.py`

- [ ] **14.2 Markov Chain Monte Carlo**
  - ノートブック: `14/14.2_Markov_Chain_Monte_Carlo.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 14.2.1 The Metropolis algorithm
    - [ ] 14.2.2 Markov chains
    - [ ] 14.2.3 The Metropolis–Hastings algorithm
    - [ ] 14.2.4 Gibbs sampling
    - [ ] 14.2.5 Ancestral sampling
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch14_markov_chain_mo.py`

- [ ] **14.3 Langevin Sampling**
  - ノートブック: `14/14.3_Langevin_Sampling.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 14.3.1 Energy-based models
    - [ ] 14.3.2 Maximizing the likelihood
    - [ ] 14.3.3 Langevin dynamics
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch14_langevin_sampli.py`

- [ ] **第14章 演習問題 (Exercises 14.1 〜 14.18, 全18問)**
  - ノートブック: `14/14_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch14_exercises.py`

---

### 第15章 離散潜在変数 (Discrete Latent Variables)

- ディレクトリ: `15/`
- 再現図版目標: Figure 15.1 〜 15.16 (全16枚)

- [ ] **15.1 K-means Clustering**
  - ノートブック: `15/15.1_K_means_Clustering.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 15.1.1 Image segmentation
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch15_k_means_cluster.py`

- [ ] **15.2 Mixtures of Gaussians**
  - ノートブック: `15/15.2_Mixtures_of_Gaussians.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 15.2.1 Likelihood function
    - [ ] 15.2.2 Maximum likelihood
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch15_mixtures_of_gau.py`

- [ ] **15.3 Expectation–Maximization Algorithm**
  - ノートブック: `15/15.3_Expectation_Maximization_Algorithm.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 15.3.1 Gaussian mixtures
    - [ ] 15.3.2 Relation to K-means
    - [ ] 15.3.3 Mixtures of Bernoulli distributions
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch15_expectation_max.py`

- [ ] **15.4 Evidence Lower Bound**
  - ノートブック: `15/15.4_Evidence_Lower_Bound.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 15.4.1 EM revisited
    - [ ] 15.4.2 Independent and identically distributed data
    - [ ] 15.4.3 Parameter priors
    - [ ] 15.4.4 Generalized EM
    - [ ] 15.4.5 Sequential EM
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch15_evidence_lower_.py`

- [ ] **第15章 演習問題 (Exercises 15.1 〜 15.24, 全24問)**
  - ノートブック: `15/15_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch15_exercises.py`

---

### 第16章 連続潜在変数 (Continuous Latent Variables)

- ディレクトリ: `16/`
- 再現図版目標: Figure 16.1 〜 16.15 (全15枚)

- [ ] **16.1 Principal Component Analysis**
  - ノートブック: `16/16.1_Principal_Component_Analysis.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 16.1.1 Maximum variance formulation
    - [ ] 16.1.2 Minimum-error formulation
    - [ ] 16.1.3 Data compression
    - [ ] 16.1.4 Data whitening
    - [ ] 16.1.5 High-dimensional data
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch16_principal_compo.py`

- [ ] **16.2 Probabilistic Latent Variables**
  - ノートブック: `16/16.2_Probabilistic_Latent_Variables.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 16.2.1 Generative model
    - [ ] 16.2.2 Likelihood function
    - [ ] 16.2.3 Maximum likelihood
    - [ ] 16.2.4 Factor analysis
    - [ ] 16.2.5 Independent component analysis
    - [ ] 16.2.6 Kalman filters
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch16_probabilistic_l.py`

- [ ] **16.3 Evidence Lower Bound**
  - ノートブック: `16/16.3_Evidence_Lower_Bound.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 16.3.1 Expectation maximization
    - [ ] 16.3.2 EM for PCA
    - [ ] 16.3.3 EM for factor analysis
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch16_evidence_lower_.py`

- [ ] **16.4 Nonlinear Latent Variable Models**
  - ノートブック: `16/16.4_Nonlinear_Latent_Variable_Models.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 16.4.1 Nonlinear manifolds
    - [ ] 16.4.2 Likelihood function
    - [ ] 16.4.3 Discrete data
    - [ ] 16.4.4 Four approaches to generative modelling
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch16_nonlinear_laten.py`

- [ ] **第16章 演習問題 (Exercises 16.1 〜 16.26, 全26問)**
  - ノートブック: `16/16_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch16_exercises.py`

---

### 第17章 敵対的生成ネットワーク (Generative Adversarial Networks: GAN)

- ディレクトリ: `17/`
- 再現図版目標: Figure 17.1 〜 17.10 (全10枚)

- [ ] **17.1 Adversarial Training**
  - ノートブック: `17/17.1_Adversarial_Training.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 17.1.1 Loss function
    - [ ] 17.1.2 GAN training in practice
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch17_adversarial_tra.py`

- [ ] **17.2 Image GANs**
  - ノートブック: `17/17.2_Image_GANs.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 17.2.1 CycleGAN
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch17_image_gans.py`

- [ ] **第17章 演習問題 (Exercises 17.1 〜 17.3, 全3問)**
  - ノートブック: `17/17_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch17_exercises.py`

---

### 第18章 正規化フロー (Normalizing Flows)

- ディレクトリ: `18/`
- 再現図版目標: Figure 18.1 〜 18.7 (全7枚)

- [ ] **18.1 Coupling Flows**
  - ノートブック: `18/18.1_Coupling_Flows.ipynb`
  - 小節: （単独構成）
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch18_coupling_flows.py`

- [ ] **18.2 Autoregressive Flows**
  - ノートブック: `18/18.2_Autoregressive_Flows.ipynb`
  - 小節: （単独構成）
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch18_autoregressive_.py`

- [ ] **18.3 Continuous Flows**
  - ノートブック: `18/18.3_Continuous_Flows.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 18.3.1 Neural differential equations
    - [ ] 18.3.2 Neural ODE backpropagation
    - [ ] 18.3.3 Neural ODE flows
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch18_continuous_flow.py`

- [ ] **第18章 演習問題 (Exercises 18.1 〜 18.11, 全11問)**
  - ノートブック: `18/18_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch18_exercises.py`

---

### 第19章 自己符号化器 (Autoencoders)

- ディレクトリ: `19/`
- 再現図版目標: Figure 19.1 〜 19.11 (全11枚)

- [ ] **19.1 Deterministic Autoencoders**
  - ノートブック: `19/19.1_Deterministic_Autoencoders.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 19.1.1 Linear autoencoders
    - [ ] 19.1.2 Deep autoencoders
    - [ ] 19.1.3 Sparse autoencoders
    - [ ] 19.1.4 Denoising autoencoders
    - [ ] 19.1.5 Masked autoencoders
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch19_deterministic_a.py`

- [ ] **19.2 Variational Autoencoders**
  - ノートブック: `19/19.2_Variational_Autoencoders.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 19.2.1 Amortized inference
    - [ ] 19.2.2 The reparameterization trick
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch19_variational_aut.py`

- [ ] **第19章 演習問題 (Exercises 19.1 〜 19.6, 全6問)**
  - ノートブック: `19/19_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch19_exercises.py`

---

### 第20章 拡散モデル (Diffusion Models)

- ディレクトリ: `20/`
- 再現図版目標: Figure 20.1 〜 20.9 (全9枚)

- [ ] **20.1 Forward Encoder**
  - ノートブック: `20/20.1_Forward_Encoder.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 20.1.1 Diffusion kernel
    - [ ] 20.1.2 Conditional distribution
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch20_forward_encoder.py`

- [ ] **20.2 Reverse Decoder**
  - ノートブック: `20/20.2_Reverse_Decoder.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 20.2.1 Training the decoder
    - [ ] 20.2.2 Evidence lower bound
    - [ ] 20.2.3 Rewriting the ELBO
    - [ ] 20.2.4 Predicting the noise
    - [ ] 20.2.5 Generating new samples
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch20_reverse_decoder.py`

- [ ] **20.3 Score Matching**
  - ノートブック: `20/20.3_Score_Matching.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 20.3.1 Score loss function
    - [ ] 20.3.2 Modified score loss
    - [ ] 20.3.3 Noise variance
    - [ ] 20.3.4 Stochastic differential equations
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch20_score_matching.py`

- [ ] **20.4 Guided Diffusion**
  - ノートブック: `20/20.4_Guided_Diffusion.ipynb`
  - 小節一覧（網羅必須）:
    - [ ] 20.4.1 Classifier guidance
    - [ ] 20.4.2 Classifier-free guidance
  - 図版再現: `result/` への該当Figureプロット保存
  - テスト: `tests/test_ch20_guided_diffusio.py`

- [ ] **第20章 演習問題 (Exercises 20.1 〜 20.20, 全20問)**
  - ノートブック: `20/20_Exercises.ipynb`
  - 形式: 理論問題は証明ロジック＋穴埋め・選択式、実装問題はコード穴埋め＋自己採点アサーション
  - テスト: `tests/test_ch20_exercises.py`

---

### 付録 (Appendices)

#### Appendix A: 線形代数 (Linear Algebra)

- [ ] **Appendix A 線形代数 (Linear Algebra)**
  - ノートブック: `appendix/appendix_a.ipynb`
    - [ ] A.1 Matrix Identities
    - [ ] A.2 Traces and Determinants
    - [ ] A.3 Matrix Derivatives
    - [ ] A.4 Eigenvectors

#### Appendix B: 変分法 (Calculus of Variations)

- [ ] **Appendix B 変分法 (Calculus of Variations)**
  - ノートブック: `appendix/appendix_b.ipynb`

#### Appendix C: ラグランジュの未定乗数法 (Lagrange Multipliers)

- [ ] **Appendix C ラグランジュの未定乗数法 (Lagrange Multipliers)**
  - ノートブック: `appendix/appendix_c.ipynb`

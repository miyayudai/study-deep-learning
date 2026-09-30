# Deep Learning: Foundations and Concepts (Bishop & Bishop 2024) Python Implementation

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-1320%20passed-success.svg)](tests/)
[![Status](https://img.shields.io/badge/Curriculum-100%25%20Completed-brightgreen.svg)](TASK.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Christopher M. Bishop & Hugh Bishop による深層学習の世界的名著『**Deep Learning: Foundations and Concepts** (Springer 2024)』を、理論解説・厳密な数式導出・Pythonスクラッチ実装・高解像度な教科書図版（Figure）再現・全章演習問題（Exercises・証明ロジック＆自己検証テスト付き）として完全実装したオープンソース教材リポジトリです。

全20章＋付録A〜Cの**全98ユニット（全98冊のJupyter Notebook）が100%完了**しており、すべてのコードセルがエラーゼロで実行検証済みです。

---

## 🌟 主な特徴

1. **原著の全節・全小節を網羅した詳細な対話型Jupyter Notebook（全98冊）**
   - 原著の全小節（例: 4.1.1, 4.1.2...）を省略することなく網羅した個別見出しと解説。
   - 天下り的な公式提示を排し、教科書の式番号に対応したステップ・バイ・ステップの途中式展開（Derivation Steps）を徹底記述。
2. **Google Colab 完全対応（1クリック即時起動・自動セットアップ）**
   - すべてのノートブックのセル0に Colab 自動環境セットアップセルを配備。GitHubのバッジから1クリックで起動し、依存パッケージのインストールから作業ディレクトリの設定まで全自動で完了します。
3. **教科書図版（約425枚超）の完全再現・高解像度保存**
   - 教科書に登場するほぼすべての図（Figure）を `matplotlib` / `scipy` を用いて忠実にスクラッチ再現し、高解像度（300 DPI）で `result/` および各章の `result/` に保存・ノートブック内に埋め込み表示。
4. **全章の演習問題（Exercises）の完全網羅（全355問）**
   - **理論・証明問題**: 証明の流れをステップ順に提示した上で、核心部分を穴埋め・選択式として出題。
   - **実装・数値問題**: 核心アルゴリズムを `None` / `# YOUR CODE HERE` とした穴埋めコードセルと、直後に自己検証可能な `assert` テストセルを完備。
5. **高品質な共通機械学習ライブラリ `deep_learning` (`common`)**
   - `pip install -e .` でインストール可能。PRML/Deep Learningの基盤アルゴリズム（最適化、分布族、サンプラー、可視化ユーティリティ）を統一APIで提供。
6. **包括的テストスイート (1,320 Tests Passing, 100% Pass Rate)**
   - 数学的整合性（勾配の一致、直交性、確率密度の正規化、不偏性、境界値）を検証する1,320件の単体テストが整備されており、`pytest tests` により高速・確実に全件パスします。

---

## 📂 ディレクトリ構成

```text
my_DeepLearning/
├── 1/               # 第1章: 深層学習革命 (多項式回帰、正則化、モデル選択、歴史)
├── 2/               # 第2章: 確率の基礎 (加法・乗法定理、ガウス分布、変数変換、情報理論)
├── 3/               # 第3章: 基本分布 (二値・カテゴリカル、多次元正規、フォン・ミーゼス、指数型分布族)
├── 4/               # 第4章: 単層ネットワーク: 回帰 (線形基底回帰、最小二乗解、決定理論、バイアス-分散)
├── 5/               # 第5章: 単層ネットワーク: 分類 (判別関数、Fisher LDA、ロジスティック回帰、IRLS)
├── 6/               # 第6章: 深層ニューラルネットワーク (MLP、活性化関数、万能近似定理、MDN)
├── 7/               # 第7章: 勾配降下法 (誤差曲面、SGD、モメンタム、Adam、BatchNorm)
├── 8/               # 第8章: 誤差逆伝播法 (誤差逆伝播、計算グラフ、ヤコビアン、自動微分)
├── 9/               # 第9章: 正則化 (重み減衰、データ拡張、学習曲線、ResNet、ドロップアウト)
├── 10/              # 第10章: 畳み込みネットワーク (畳み込み、VGG/ResNet、YOLO、画風変換)
├── 11/              # 第11章: 構造化分布 (ベイジアンネット、d-分離、マルコフモデル、自己回帰)
├── 12/              # 第12章: トランスフォーマー (Self-Attention、Multi-head、BERT/GPT、ViT)
├── 13/              # 第13章: グラフニューラルネットワーク (メッセージ伝播 MPNN、GCN、GAT)
├── 14/              # 第14章: サンプリング (棄却・重点サンプリング、MCMC、ランジュバン動力学)
├── 15/              # 第15章: 離散潜在変数 (K-means、GMM、EMアルゴリズム、証拠下界 ELBO)
├── 16/              # 第16章: 連続潜在変数 (主成分分析 PCA、確率的PCA、因子分析、非線形多様体)
├── 17/              # 第17章: 敵対的生成ネットワーク (GAN、DCGAN、Wasserstein GAN、CycleGAN)
├── 18/              # 第18章: 正規化フロー (RealNVP、カップリング層、連続フロー CNF)
├── 19/              # 第19章: 自己符号化器 (オートエンコーダ、変分自己符号化器 VAE、再パラメータ化)
├── 20/              # 第20章: 拡散モデル (前向き拡散過程、スコアベース生成、DDPM、確率流ODE)
├── appendix/        # 付録: A 線形代数とWoodbury公式 / B 変分法 / C ラグランジュ未定乗数法
├── common/          # 共通機械学習基盤パッケージ (アルゴリズム・可視化・最適化・データ)
├── result/          # 再現図版保存ディレクトリ (Figure 1.1 〜 20.9, 425+ images)
├── scripts/         # ノートブック自動生成・検証用スクリプト群
├── tests/           # 統合・単体テストスイート (1,320 tests passing)
├── pyproject.toml   # PEP 517/621 パッケージ定義ファイル
├── requirements.txt # 依存パッケージ一覧
└── TASK.md          # 全章・全小節の進捗・仕様管理タスク定義 (98/98 ユニット完了)
```

---

## 🚀 クイックスタート

### 1. リポジトリのクローンと環境構築

```bash
git clone git@github.com:miyayudai/my_DeepLearning.git
cd my_DeepLearning

# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# 共通ライブラリを開発モードでインストール
pip install -e .
```

### 2. 単体テストの実行

```bash
pytest tests/ -v
# 1,320 passed
```

---

## 📖 全章カリキュラムと進捗一覧

| Chapter | タイトル | 主なトピック | 演習問題 | 図版 | 状態 |
|---|---|---|---|:---:|:---:|
| **第1章** | **深層学習革命** | 人工データ生成、多項式曲線当てはめ、二乗和誤差、正則化、モデル選択、機械学習の歴史 | Tutorial | 16 枚 | 完了 (`[x]`) |
| **第2章** | **確率の基礎** | 加法・乗法定理、ベイズの定理、確率密度、多次元ガウス分布、変数変換、情報理論、エフロンのサイコロ | Ex 2.1〜2.41 (全41問) | 20 枚 | 完了 (`[x]`) |
| **第3章** | **基本分布** | 二値・カテゴリカル変数、多次元正規分布、フォン・ミーゼス分布、指数型分布族、ノンパラメトリック密度推定 | Ex 3.1〜3.38 (全38問) | 27 枚 | 完了 (`[x]`) |
| **第4章** | **単層ネットワーク: 回帰** | 線形基底関数回帰、最尤推定、最小二乗解、決定理論、損失関数、バイアス-バリアンス分解 | Ex 4.1〜4.12 (全12問) | 20 枚 | 完了 (`[x]`) |
| **第5章** | **単層ネットワーク: 分類** | 判別関数、Fisherの線形判別 (LDA)、ロジスティック回帰、多クラスソフトマックス、IRLS、クロスエントロピー | Ex 5.1〜5.24 (全24問) | 21 枚 | 完了 (`[x]`) |
| **第6章** | **深層ニューラルネットワーク** | 固定基底関数の限界、多層パーセプトロン (MLP)、活性化関数、万能近似定理、混合密度ネットワーク (MDN) | Ex 6.1〜6.21 (全21問) | 19 枚 | 完了 (`[x]`) |
| **第7章** | **勾配降下法** | 誤差曲面、バッチ勾配降下法、確率的勾配降下法 (SGD)、モメンタム、RMSprop、Adam、バッチ正規化 (BatchNorm) | Ex 7.1〜7.14 (全14問) | 11 枚 | 完了 (`[x]`) |
| **第8章** | **誤差逆伝播法** | 誤差逆伝播アルゴリズム (Backpropagation)、計算グラフ、連鎖律、ヤコビアン・ヘッセ行列、自動微分 (Autodiff) | Ex 8.1〜8.18 (全18問) | 10 枚 | 完了 (`[x]`) |
| **第9章** | **正則化** | 帰納バイアス、重み減衰 (L2正則化)、データ拡張、学習曲線、パラメータ共有、残差接続 (ResNet)、ドロップアウト | Ex 9.1〜9.18 (全18問) | 40 枚 | 完了 (`[x]`) |
| **第10章** | **畳み込みネットワーク** | 畳み込みフィルタ、受容野、CNNアーキテクチャ (VGG/ResNet)、特徴可視化、物体検出 (YOLO)、セグメンテーション、画風変換 | Ex 10.1〜10.13 (全13問) | 35 枚 | 完了 (`[x]`) |
| **第11章** | **構造化分布** | ベイジアンネットワーク、条件付き独立性 (d-分離)、マルコフ連鎖、自己回帰モデル、系列データモデリング | Ex 11.1〜11.20 (全20問) | 32 枚 | 完了 (`[x]`) |
| **第12章** | **トランスフォーマー** | 自己注意機構 (Self-Attention)、Multi-head Attention、トランスフォーマー層、BERT/GPT言語モデル、Vision Transformer (ViT) | Ex 12.1〜12.16 (全16問) | 28 枚 | 完了 (`[x]`) |
| **第13章** | **グラフニューラルネットワーク** | グラフ構造データ、順列同変性・不変性、ニューラルメッセージ伝播 (MPNN)、GCN、GAT、大域的グラフ属性結合 | Ex 13.1〜13.10 (全10問) | 9 枚 | 完了 (`[x]`) |
| **第14章** | **サンプリング** | 基本サンプリング (棄却・重点サンプリング)、マルコフ連鎖モンテカルロ (MCMC, Metropolis-Hastings, Gibbs)、ランジュバン動力学 | Ex 14.1〜14.18 (全18問) | 14 枚 | 完了 (`[x]`) |
| **第15章** | **離散潜在変数** | K-means クラスタリング、混合ガウスモデル (GMM)、EMアルゴリズムの厳密導出、証拠下界 (ELBO) とイェンセンの不等式 | Ex 15.1〜15.24 (全24問) | 16 枚 | 完了 (`[x]`) |
| **第16章** | **連続潜在変数** | 主成分分析 (PCA)、確率的PCA (PPCA)、因子分析 (FA)、EM-PCA、非線形多様体、脱量子化、現代生成モデルの俯瞰 | Ex 16.1〜16.26 (全26問) | 15 枚 | 完了 (`[x]`) |
| **第17章** | **敵対的生成ネットワーク** | GANのミニマックスゲーム、JSダイバージェンス、モード崩壊、Wasserstein GAN (WGAN-GP)、DCGAN、CycleGAN | Ex 17.1〜17.3 (全3問) | 17 枚 | 完了 (`[x]`) |
| **第18章** | **正規化フロー** | 可逆ニューラルネットワーク、ヤコビアン行列式、RealNVP、カップリング層、連続フロー (CNF)、厳密尤度計算 | Ex 18.1〜18.11 (全11問) | 11 枚 | 完了 (`[x]`) |
| **第19章** | **自己符号化器** | 決定論的AE、変分自己符号化器 (VAE)、再パラメータ化トリック、ELBO厳密分解、潜在空間補間、条件付きVAE | Ex 19.1〜19.6 (全6問) | 22 枚 | 完了 (`[x]`) |
| **第20章** | **拡散モデル** | 前向き拡散過程、スコアベース生成モデル、デノイジングスコアマッチング、逆時間SDE、DDPM、確率流ODE、ガイダンス | Ex 20.1〜20.20 (全20問) | 18 枚 | 完了 (`[x]`) |
| **付録** | **付録 (Appendices)** | 付録A: 線形代数とWoodbury公式、付録B: 変分法とオイラー・ラグランジュ方程式、付録C: ラグランジュ未定乗数法とKKT条件 | 証明・検証コード | 12 枚 | 完了 (`[x]`) |

---

## 📓 ノートブック一覧 & Google Colab リンク

各章のJupyter Notebookは、GitHub上で直接閲覧することも、バッジをクリックしてGoogle Colabで即座に対話実行することも可能です。

### 第1章 深層学習革命 (The Deep Learning Revolution)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`1.1_The_Impact_of_Deep_Learning.ipynb`](1/1.1_The_Impact_of_Deep_Learning.ipynb) | 1.1 The Impact of Deep Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/1/1.1_The_Impact_of_Deep_Learning.ipynb) |
| [`1.2_A_Tutorial_Example.ipynb`](1/1.2_A_Tutorial_Example.ipynb) | 1.2 A Tutorial Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/1/1.2_A_Tutorial_Example.ipynb) |
| [`1.3_A_Brief_History_of_Machine_Learning.ipynb`](1/1.3_A_Brief_History_of_Machine_Learning.ipynb) | 1.3 A Brief History of Machine Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/1/1.3_A_Brief_History_of_Machine_Learning.ipynb) |

### 第2章 確率の基礎 (Probabilities)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`2.1_The_Rules_of_Probability.ipynb`](2/2.1_The_Rules_of_Probability.ipynb) | 2.1 The Rules of Probability | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.1_The_Rules_of_Probability.ipynb) |
| [`2.2_Probability_Densities.ipynb`](2/2.2_Probability_Densities.ipynb) | 2.2 Probability Densities | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.2_Probability_Densities.ipynb) |
| [`2.3_The_Gaussian_Distribution.ipynb`](2/2.3_The_Gaussian_Distribution.ipynb) | 2.3 The Gaussian Distribution | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.3_The_Gaussian_Distribution.ipynb) |
| [`2.4_Transformation_of_Densities.ipynb`](2/2.4_Transformation_of_Densities.ipynb) | 2.4 Transformation of Densities | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.4_Transformation_of_Densities.ipynb) |
| [`2.5_Information_Theory.ipynb`](2/2.5_Information_Theory.ipynb) | 2.5 Information Theory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.5_Information_Theory.ipynb) |
| [`2.6_Bayesian_Probabilities.ipynb`](2/2.6_Bayesian_Probabilities.ipynb) | 2.6 Bayesian Probabilities | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.6_Bayesian_Probabilities.ipynb) |
| [`2_Exercises.ipynb`](2/2_Exercises.ipynb) | 2 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2_Exercises.ipynb) |

### 第3章 基本分布 (Standard Distributions)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`3.1_Discrete_Variables.ipynb`](3/3.1_Discrete_Variables.ipynb) | 3.1 Discrete Variables | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.1_Discrete_Variables.ipynb) |
| [`3.2_The_Multivariate_Gaussian.ipynb`](3/3.2_The_Multivariate_Gaussian.ipynb) | 3.2 The Multivariate Gaussian | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.2_The_Multivariate_Gaussian.ipynb) |
| [`3.3_Periodic_Variables.ipynb`](3/3.3_Periodic_Variables.ipynb) | 3.3 Periodic Variables | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.3_Periodic_Variables.ipynb) |
| [`3.4_The_Exponential_Family.ipynb`](3/3.4_The_Exponential_Family.ipynb) | 3.4 The Exponential Family | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.4_The_Exponential_Family.ipynb) |
| [`3.5_Nonparametric_Methods.ipynb`](3/3.5_Nonparametric_Methods.ipynb) | 3.5 Nonparametric Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.5_Nonparametric_Methods.ipynb) |
| [`3_Exercises.ipynb`](3/3_Exercises.ipynb) | 3 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3_Exercises.ipynb) |

### 第4章 単層ネットワーク: 回帰 (Single-layer Networks: Regression)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`4.1_Linear_Regression.ipynb`](4/4.1_Linear_Regression.ipynb) | 4.1 Linear Regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4.1_Linear_Regression.ipynb) |
| [`4.2_Decision_theory.ipynb`](4/4.2_Decision_theory.ipynb) | 4.2 Decision theory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4.2_Decision_theory.ipynb) |
| [`4.3_The_Bias_Variance_Trade_off.ipynb`](4/4.3_The_Bias_Variance_Trade_off.ipynb) | 4.3 The Bias Variance Trade off | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4.3_The_Bias_Variance_Trade_off.ipynb) |
| [`4_Exercises.ipynb`](4/4_Exercises.ipynb) | 4 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4_Exercises.ipynb) |

### 第5章 単層ネットワーク: 分類 (Single-layer Networks: Classification)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`5.1_Discriminant_Functions.ipynb`](5/5.1_Discriminant_Functions.ipynb) | 5.1 Discriminant Functions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.1_Discriminant_Functions.ipynb) |
| [`5.2_Decision_Theory.ipynb`](5/5.2_Decision_Theory.ipynb) | 5.2 Decision Theory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.2_Decision_Theory.ipynb) |
| [`5.3_Generative_Classifiers.ipynb`](5/5.3_Generative_Classifiers.ipynb) | 5.3 Generative Classifiers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.3_Generative_Classifiers.ipynb) |
| [`5.4_Discriminative_Classifiers.ipynb`](5/5.4_Discriminative_Classifiers.ipynb) | 5.4 Discriminative Classifiers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.4_Discriminative_Classifiers.ipynb) |
| [`5_Exercises.ipynb`](5/5_Exercises.ipynb) | 5 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5_Exercises.ipynb) |

### 第6章 深層ニューラルネットワーク (Deep Neural Networks)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`6.1_Limitations_of_Fixed_Basis_Functions.ipynb`](6/6.1_Limitations_of_Fixed_Basis_Functions.ipynb) | 6.1 Limitations of Fixed Basis Functions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.1_Limitations_of_Fixed_Basis_Functions.ipynb) |
| [`6.2_Multilayer_Networks.ipynb`](6/6.2_Multilayer_Networks.ipynb) | 6.2 Multilayer Networks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.2_Multilayer_Networks.ipynb) |
| [`6.3_Deep_Networks.ipynb`](6/6.3_Deep_Networks.ipynb) | 6.3 Deep Networks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.3_Deep_Networks.ipynb) |
| [`6.4_Error_Functions.ipynb`](6/6.4_Error_Functions.ipynb) | 6.4 Error Functions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.4_Error_Functions.ipynb) |
| [`6.5_Mixture_Density_Networks.ipynb`](6/6.5_Mixture_Density_Networks.ipynb) | 6.5 Mixture Density Networks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.5_Mixture_Density_Networks.ipynb) |
| [`6_Exercises.ipynb`](6/6_Exercises.ipynb) | 6 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6_Exercises.ipynb) |

### 第7章 勾配降下法 (Gradient Descent)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`7.1_Error_Surfaces.ipynb`](7/7.1_Error_Surfaces.ipynb) | 7.1 Error Surfaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.1_Error_Surfaces.ipynb) |
| [`7.2_Gradient_Descent_Optimization.ipynb`](7/7.2_Gradient_Descent_Optimization.ipynb) | 7.2 Gradient Descent Optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.2_Gradient_Descent_Optimization.ipynb) |
| [`7.3_Convergence.ipynb`](7/7.3_Convergence.ipynb) | 7.3 Convergence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.3_Convergence.ipynb) |
| [`7.4_Normalization.ipynb`](7/7.4_Normalization.ipynb) | 7.4 Normalization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.4_Normalization.ipynb) |
| [`7_Exercises.ipynb`](7/7_Exercises.ipynb) | 7 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7_Exercises.ipynb) |

### 第8章 誤差逆伝播法 (Backpropagation)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`8.1_Evaluation_of_Gradients.ipynb`](8/8.1_Evaluation_of_Gradients.ipynb) | 8.1 Evaluation of Gradients | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/8/8.1_Evaluation_of_Gradients.ipynb) |
| [`8.2_Automatic_Differentiation.ipynb`](8/8.2_Automatic_Differentiation.ipynb) | 8.2 Automatic Differentiation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/8/8.2_Automatic_Differentiation.ipynb) |
| [`8_Exercises.ipynb`](8/8_Exercises.ipynb) | 8 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/8/8_Exercises.ipynb) |

### 第9章 正則化 (Regularization)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`9.1_Inductive_Bias.ipynb`](9/9.1_Inductive_Bias.ipynb) | 9.1 Inductive Bias | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.1_Inductive_Bias.ipynb) |
| [`9.2_Weight_Decay.ipynb`](9/9.2_Weight_Decay.ipynb) | 9.2 Weight Decay | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.2_Weight_Decay.ipynb) |
| [`9.3_Learning_Curves.ipynb`](9/9.3_Learning_Curves.ipynb) | 9.3 Learning Curves | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.3_Learning_Curves.ipynb) |
| [`9.4_Parameter_Sharing.ipynb`](9/9.4_Parameter_Sharing.ipynb) | 9.4 Parameter Sharing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.4_Parameter_Sharing.ipynb) |
| [`9.5_Residual_Connections.ipynb`](9/9.5_Residual_Connections.ipynb) | 9.5 Residual Connections | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.5_Residual_Connections.ipynb) |
| [`9.6_Model_Averaging.ipynb`](9/9.6_Model_Averaging.ipynb) | 9.6 Model Averaging | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.6_Model_Averaging.ipynb) |
| [`9_Exercises.ipynb`](9/9_Exercises.ipynb) | 9 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9_Exercises.ipynb) |

### 第10章 畳み込みネットワーク (Convolutional Networks)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`10.1_Computer_Vision.ipynb`](10/10.1_Computer_Vision.ipynb) | 10.1 Computer Vision | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.1_Computer_Vision.ipynb) |
| [`10.2_Convolutional_Filters.ipynb`](10/10.2_Convolutional_Filters.ipynb) | 10.2 Convolutional Filters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.2_Convolutional_Filters.ipynb) |
| [`10.3_Visualizing_Trained_CNNs.ipynb`](10/10.3_Visualizing_Trained_CNNs.ipynb) | 10.3 Visualizing Trained CNNs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.3_Visualizing_Trained_CNNs.ipynb) |
| [`10.4_Object_Detection.ipynb`](10/10.4_Object_Detection.ipynb) | 10.4 Object Detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.4_Object_Detection.ipynb) |
| [`10.5_Image_Segmentation.ipynb`](10/10.5_Image_Segmentation.ipynb) | 10.5 Image Segmentation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.5_Image_Segmentation.ipynb) |
| [`10.6_Style_Transfer.ipynb`](10/10.6_Style_Transfer.ipynb) | 10.6 Style Transfer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.6_Style_Transfer.ipynb) |
| [`10_Exercises.ipynb`](10/10_Exercises.ipynb) | 10 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10_Exercises.ipynb) |

### 第11章 構造化分布 (Structured Distributions)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`11.1_Graphical_Models.ipynb`](11/11.1_Graphical_Models.ipynb) | 11.1 Graphical Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11.1_Graphical_Models.ipynb) |
| [`11.2_Conditional_Independence.ipynb`](11/11.2_Conditional_Independence.ipynb) | 11.2 Conditional Independence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11.2_Conditional_Independence.ipynb) |
| [`11.3_Sequence_Models.ipynb`](11/11.3_Sequence_Models.ipynb) | 11.3 Sequence Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11.3_Sequence_Models.ipynb) |
| [`11_Exercises.ipynb`](11/11_Exercises.ipynb) | 11 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11_Exercises.ipynb) |

### 第12章 トランスフォーマー (Transformers)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`12.1_Attention.ipynb`](12/12.1_Attention.ipynb) | 12.1 Attention | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.1_Attention.ipynb) |
| [`12.2_Natural_Language.ipynb`](12/12.2_Natural_Language.ipynb) | 12.2 Natural Language | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.2_Natural_Language.ipynb) |
| [`12.3_Transformer_Language_Models.ipynb`](12/12.3_Transformer_Language_Models.ipynb) | 12.3 Transformer Language Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.3_Transformer_Language_Models.ipynb) |
| [`12.4_Multimodal_Transformers.ipynb`](12/12.4_Multimodal_Transformers.ipynb) | 12.4 Multimodal Transformers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.4_Multimodal_Transformers.ipynb) |
| [`12_Exercises.ipynb`](12/12_Exercises.ipynb) | 12 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12_Exercises.ipynb) |

### 第13章 グラフニューラルネットワーク (Graph Neural Networks)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`13.1_Machine_Learning_on_Graphs.ipynb`](13/13.1_Machine_Learning_on_Graphs.ipynb) | 13.1 Machine Learning on Graphs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13.1_Machine_Learning_on_Graphs.ipynb) |
| [`13.2_Neural_Message_Passing.ipynb`](13/13.2_Neural_Message_Passing.ipynb) | 13.2 Neural Message Passing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13.2_Neural_Message_Passing.ipynb) |
| [`13.3_General_Graph_Networks.ipynb`](13/13.3_General_Graph_Networks.ipynb) | 13.3 General Graph Networks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13.3_General_Graph_Networks.ipynb) |
| [`13_Exercises.ipynb`](13/13_Exercises.ipynb) | 13 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13_Exercises.ipynb) |

### 第14章 サンプリング (Sampling)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`14.1_Basic_Sampling_Algorithms.ipynb`](14/14.1_Basic_Sampling_Algorithms.ipynb) | 14.1 Basic Sampling Algorithms | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14.1_Basic_Sampling_Algorithms.ipynb) |
| [`14.2_Markov_Chain_Monte_Carlo.ipynb`](14/14.2_Markov_Chain_Monte_Carlo.ipynb) | 14.2 Markov Chain Monte Carlo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14.2_Markov_Chain_Monte_Carlo.ipynb) |
| [`14.3_Langevin_Sampling.ipynb`](14/14.3_Langevin_Sampling.ipynb) | 14.3 Langevin Sampling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14.3_Langevin_Sampling.ipynb) |
| [`14_Exercises.ipynb`](14/14_Exercises.ipynb) | 14 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14_Exercises.ipynb) |

### 第15章 離散潜在変数 (Discrete Latent Variables)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`15.1_K_means_Clustering.ipynb`](15/15.1_K_means_Clustering.ipynb) | 15.1 K means Clustering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.1_K_means_Clustering.ipynb) |
| [`15.2_Mixtures_of_Gaussians.ipynb`](15/15.2_Mixtures_of_Gaussians.ipynb) | 15.2 Mixtures of Gaussians | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.2_Mixtures_of_Gaussians.ipynb) |
| [`15.3_Expectation_Maximization_Algorithm.ipynb`](15/15.3_Expectation_Maximization_Algorithm.ipynb) | 15.3 Expectation Maximization Algorithm | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.3_Expectation_Maximization_Algorithm.ipynb) |
| [`15.4_Evidence_Lower_Bound.ipynb`](15/15.4_Evidence_Lower_Bound.ipynb) | 15.4 Evidence Lower Bound | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.4_Evidence_Lower_Bound.ipynb) |
| [`15_Exercises.ipynb`](15/15_Exercises.ipynb) | 15 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15_Exercises.ipynb) |

### 第16章 連続潜在変数 (Continuous Latent Variables)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`16.1_Principal_Component_Analysis.ipynb`](16/16.1_Principal_Component_Analysis.ipynb) | 16.1 Principal Component Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.1_Principal_Component_Analysis.ipynb) |
| [`16.2_Probabilistic_Latent_Variables.ipynb`](16/16.2_Probabilistic_Latent_Variables.ipynb) | 16.2 Probabilistic Latent Variables | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.2_Probabilistic_Latent_Variables.ipynb) |
| [`16.3_Evidence_Lower_Bound.ipynb`](16/16.3_Evidence_Lower_Bound.ipynb) | 16.3 Evidence Lower Bound | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.3_Evidence_Lower_Bound.ipynb) |
| [`16.4_Nonlinear_Latent_Variable_Models.ipynb`](16/16.4_Nonlinear_Latent_Variable_Models.ipynb) | 16.4 Nonlinear Latent Variable Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.4_Nonlinear_Latent_Variable_Models.ipynb) |
| [`16_Exercises.ipynb`](16/16_Exercises.ipynb) | 16 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16_Exercises.ipynb) |

### 第17章 敵対的生成ネットワーク (Generative Adversarial Networks)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`17.1_Adversarial_Training.ipynb`](17/17.1_Adversarial_Training.ipynb) | 17.1 Adversarial Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/17/17.1_Adversarial_Training.ipynb) |
| [`17.2_Image_GANs.ipynb`](17/17.2_Image_GANs.ipynb) | 17.2 Image GANs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/17/17.2_Image_GANs.ipynb) |
| [`17_Exercises.ipynb`](17/17_Exercises.ipynb) | 17 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/17/17_Exercises.ipynb) |

### 第18章 正規化フロー (Normalizing Flows)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`18.1_Coupling_Flows.ipynb`](18/18.1_Coupling_Flows.ipynb) | 18.1 Coupling Flows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/18/18.1_Coupling_Flows.ipynb) |
| [`18.2_Autoregressive_Flows.ipynb`](18/18.2_Autoregressive_Flows.ipynb) | 18.2 Autoregressive Flows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/18/18.2_Autoregressive_Flows.ipynb) |
| [`18.3_Continuous_Flows.ipynb`](18/18.3_Continuous_Flows.ipynb) | 18.3 Continuous Flows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/18/18.3_Continuous_Flows.ipynb) |
| [`18_Exercises.ipynb`](18/18_Exercises.ipynb) | 18 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/18/18_Exercises.ipynb) |

### 第19章 自己符号化器 (Autoencoders)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`19.1_Deterministic_Autoencoders.ipynb`](19/19.1_Deterministic_Autoencoders.ipynb) | 19.1 Deterministic Autoencoders | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/19/19.1_Deterministic_Autoencoders.ipynb) |
| [`19.2_Variational_Autoencoders.ipynb`](19/19.2_Variational_Autoencoders.ipynb) | 19.2 Variational Autoencoders | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/19/19.2_Variational_Autoencoders.ipynb) |
| [`19_Exercises.ipynb`](19/19_Exercises.ipynb) | 19 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/19/19_Exercises.ipynb) |

### 第20章 拡散モデル (Diffusion Models)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`20.1_Forward_Encoder.ipynb`](20/20.1_Forward_Encoder.ipynb) | 20.1 Forward Encoder | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/20/20.1_Forward_Encoder.ipynb) |
| [`20.2_Reverse_Decoder.ipynb`](20/20.2_Reverse_Decoder.ipynb) | 20.2 Reverse Decoder | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/20/20.2_Reverse_Decoder.ipynb) |
| [`20.3_Score_Matching.ipynb`](20/20.3_Score_Matching.ipynb) | 20.3 Score Matching | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/20/20.3_Score_Matching.ipynb) |
| [`20.4_Guided_Diffusion.ipynb`](20/20.4_Guided_Diffusion.ipynb) | 20.4 Guided Diffusion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/20/20.4_Guided_Diffusion.ipynb) |
| [`20_Exercises.ipynb`](20/20_Exercises.ipynb) | 20 Exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/20/20_Exercises.ipynb) |

### 付録 (Appendices A-C)

| ノートブック | 内容 | Colab で開く |
|---|---|:---:|
| [`appendix_a.ipynb`](appendix/appendix_a.ipynb) | appendix a | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/appendix/appendix_a.ipynb) |
| [`appendix_b.ipynb`](appendix/appendix_b.ipynb) | appendix b | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/appendix/appendix_b.ipynb) |
| [`appendix_c.ipynb`](appendix/appendix_c.ipynb) | appendix c | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/appendix/appendix_c.ipynb) |

---

## 🛠️ 技術スタック

- **言語**: Python 3.9+
- **数値計算・科学技術**: `numpy`, `scipy`, `pandas`
- **可視化**: `matplotlib`, `seaborn`
- **機械学習・画像処理**: `scikit-learn`, `pillow`
- **テスト・品質管理**: `pytest` (1,320 tests)
- **環境構築**: Google Colab 1-click execution ready

---

## 📜 ライセンス

本リポジトリは [MIT License](LICENSE) の下で公開されています。

# Deep Learning: Foundations and Concepts (Bishop & Bishop 2024) Python Implementation

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-1135%20passed-success.svg)](tests/)
[![Deep Learning](https://img.shields.io/badge/Bishop%202024-Active%20Development-brightgreen.svg)](TASK.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Christopher M. Bishop & Hugh Bishop による深層学習の世界的名著『**Deep Learning: Foundations and Concepts** (Springer 2024)』を、理論解説・厳密な数式導出・Pythonスクラッチ実装・高解像度な教科書図版（Figure）再現・全章演習問題（Exercises・証明ロジック＆自己検証テスト付き）として完全実装したオープンソース教材リポジトリです。

---

## 🌟 主な特徴

1. **原著の全節・全小節を網羅した詳細な対話型Jupyter Notebook（全79冊）**
   - 原著の全小節（例: 4.1.1, 4.1.2...）を省略することなく網羅した個別見出しと解説。
   - 天下り的な公式提示を排し、教科書の式番号に対応したステップ・バイ・ステップの途中式展開（Derivation Steps）を徹底記述。
2. **Google Colab 完全対応（1クリック即時起動・自動セットアップ）**
   - すべてのノートブックのセル0に Colab 自動環境セットアップセルを配備。GitHubのバッジから1クリックで起動し、依存パッケージのインストールから作業ディレクトリの設定まで全自動で完了します。
3. **教科書図版（約300枚超）の完全再現・可視化**
   - 教科書に登場するほぼすべての図（Figure）を `matplotlib` / `scipy` を用いて忠実にスクラッチ再現し、高解像度（300 DPI）で `result/` に保存・ノートブック内に埋め込み表示。
4. **全章の演習問題（Exercises）の完全網羅（全355問）**
   - **理論・証明問題**: 証明の流れをステップ順に提示した上で、核心部分を穴埋め・選択式として出題。
   - **実装・数値問題**: 核心アルゴリズムを `None` / `# YOUR CODE HERE` とした穴埋めコードセルと、直後に自己検証可能な `assert` テストセルを完備。
5. **高品質な共通機械学習ライブラリ `deep_learning` (`common`)**
   - `pip install -e .` でインストール可能。PRML/Deep Learningの基盤アルゴリズム（最適化、分布族、サンプラー、可視化ユーティリティ）を統一APIで提供。
6. **包括的テストスイート (1,135 Tests Passing)**
   - 数学的整合性（勾配の一致、直交性、確率密度の正規化、不偏性、境界値）を検証する1,135件の単体テストが整備されており、`pytest tests` により高速・確実に全件パスします。

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
├── 17/              # 第17章: 敵対的生成ネットワーク (GAN、WGAN-GP) [準備中]
├── 18/              # 第18章: 正規化フロー (RealNVP、カップリング層) [準備中]
├── 19/              # 第19章: 自己符号化器 (オートエンコーダ、VAE) [準備中]
├── 20/              # 第20章: 拡散モデル (スコアベースモデル、DDPM) [準備中]
├── common/          # 共通機械学習基盤パッケージ (アルゴリズム・可視化・最適化・データ)
├── result/          # 再現図版保存ディレクトリ (Figure 1.1 〜 16.15, 300+ images)
├── scripts/         # ノートブック自動生成・検証用スクリプト群
├── tests/           # 統合・単体テストスイート (1,135 tests passing)
├── pyproject.toml   # PEP 517/621 パッケージ定義ファイル
├── requirements.txt # 依存パッケージ一覧
├── TASK.md          # 厳格な開発要件・品質基準・排他ロック進捗管理ドキュメント
└── README.md        # 本ドキュメント
```

---

## 🚀 クイックスタート

### 1. リポジトリのクローン & パッケージインストール

```bash
git clone https://github.com/miyayudai/my_DeepLearning.git
cd my_DeepLearning

# 開発モードでインストール (common パッケージが deep_learning として利用可能になります)
pip install -e .
```

### 2. ライブラリとしての利用例

Bishop の最新アルゴリズムは、統一された直感的なインターフェースで設計されています：

```python
import numpy as np
from common.principal_component_analysis import PrincipalComponentAnalysis
from common.probabilistic_latent_variables import ProbabilisticPCA
from common.expectation_maximization import GaussianMixtureModel

# 1. 主成分分析 (Ch 16)
X = np.random.randn(100, 5)
pca = PrincipalComponentAnalysis(n_components=2)
pca.fit(X)
X_proj = pca.transform(X)
print("PCA 射影後の形状:", X_proj.shape)

# 2. 確率的主成分分析 (PPCA, Ch 16)
ppca = ProbabilisticPCA(n_components=2)
ppca.fit(X)
mu_z, cov_z = ppca.infer_latent(X)
print("PPCA 潜在変数平均:", mu_z.shape)

# 3. 混合ガウスモデルとEMアルゴリズム (Ch 15)
gmm = GaussianMixtureModel(n_components=3)
gmm.fit(X)
responsibilities = gmm.predict_proba(X)
print("EM 負担率の形状:", responsibilities.shape)
```

### 3. テストスイートの実行

```bash
# 全章・全モジュールを網羅する1,135件のユニットテストを実行
pytest tests -v
```

---

## 📚 章別カリキュラムと主要トピック

| 章 | タイトル | 主なトピック・実装アルゴリズム | 演習問題 (Exercises) | 再現図版 | 進捗状態 |
|:---:|---|---|:---:|:---:|:---:|

| **1** | **深層学習革命** | 人工データ生成、多項式曲線当てはめ、二乗和誤差、正則化、モデル選択、機械学習の歴史 | Tutorial | 16 枚 | 完了 (`[x]`) |
| **2** | **確率の基礎** | 加法・乗法定理、ベイズの定理、確率密度、多次元ガウス分布、変数変換、情報理論、エフロンのサイコロ | Ex 2.1〜2.41 (全41問) | 16 枚 | 完了 (`[x]`) |
| **3** | **基本分布** | 二値・カテゴリカル変数、多次元正規分布、フォン・ミーゼス分布、指数型分布族、ノンパラメトリック密度推定 | Ex 3.1〜3.38 (全38問) | 16 枚 | 完了 (`[x]`) |
| **4** | **単層ネットワーク: 回帰** | 線形基底関数回帰、最尤推定、最小二乗解、決定理論、損失関数、バイアス-バリアンス分解 | Ex 4.1〜4.12 (全12問) | 8 枚 | 完了 (`[x]`) |
| **5** | **単層ネットワーク: 分類** | 判別関数、Fisherの線形判別 (LDA)、ロジスティック回帰、多クラスソフトマックス、IRLS、クロスエントロピー | Ex 5.1〜5.24 (全24問) | 17 枚 | 完了 (`[x]`) |
| **6** | **深層ニューラルネットワーク** | 固定基底関数の限界、多層パーセプトロン (MLP)、活性化関数、万能近似定理、混合密度ネットワーク (MDN) | Ex 6.1〜6.21 (全21問) | 19 枚 | 完了 (`[x]`) |
| **7** | **勾配降下法** | 誤差曲面、バッチ勾配降下法、確率的勾配降下法 (SGD)、モメンタム、RMSprop、Adam、バッチ正規化 (BatchNorm) | Ex 7.1〜7.14 (全14問) | 10 枚 | 完了 (`[x]`) |
| **8** | **誤差逆伝播法** | 誤差逆伝播アルゴリズム (Backpropagation)、計算グラフ、連鎖律、ヤコビアン・ヘッセ行列、自動微分 (Autodiff) | Ex 8.1〜8.18 (全18問) | 5 枚 | 完了 (`[x]`) |
| **9** | **正則化** | 帰納バイアス、重み減衰 (L2正則化)、データ拡張、学習曲線、パラメータ共有、残差接続 (ResNet)、ドロップアウト | Ex 9.1〜9.18 (全18問) | 17 枚 | 完了 (`[x]`) |
| **10** | **畳み込みネットワーク** | 畳み込みフィルタ、受容野、CNNアーキテクチャ (VGG/ResNet)、特徴可視化、物体検出 (YOLO)、セグメンテーション、画風変換 | Ex 10.1〜10.13 (全13問) | 32 枚 | 完了 (`[x]`) |
| **11** | **構造化分布** | ベイジアンネットワーク、条件付き独立性 (d-分離)、マルコフ連鎖、自己回帰モデル、系列データモデリング | Ex 11.1〜11.20 (全20問) | 32 枚 | 完了 (`[x]`) |
| **12** | **トランスフォーマー** | 自己注意機構 (Self-Attention)、Multi-head Attention、トランスフォーマー層、BERT/GPT言語モデル、Vision Transformer (ViT) | Ex 12.1〜12.16 (全16問) | 27 枚 | 完了 (`[x]`) |
| **13** | **グラフニューラルネットワーク** | グラフ構造データ、順列同変性・不変性、ニューラルメッセージ伝播 (MPNN)、GCN、GAT、大域的グラフ属性結合 | Ex 13.1〜13.10 (全10問) | 5 枚 | 完了 (`[x]`) |
| **14** | **サンプリング** | 基本サンプリング (棄却・重点サンプリング)、マルコフ連鎖モンテカルロ (MCMC, Metropolis-Hastings, Gibbs)、ランジュバン動力学 | Ex 14.1〜14.18 (全18問) | 14 枚 | 完了 (`[x]`) |
| **15** | **離散潜在変数** | K-means クラスタリング、混合ガウスモデル (GMM)、EMアルゴリズムの厳密導出、証拠下界 (ELBO) とイェンセンの不等式 | Ex 15.1〜15.24 (全24問) | 16 枚 | 完了 (`[x]`) |
| **16** | **連続潜在変数** | 主成分分析 (PCA)、確率的PCA (PPCA)、因子分析 (FA)、EM-PCA、非線形多様体、脱量子化、現代生成モデルの俯瞰 | Ex 16.1〜16.26 (全26問) | 29 枚 | 進行中 (`[-]`) |
| **17** | **敵対的生成ネットワーク** | GANのミニマックスゲーム、JSダイバージェンス、モード崩壊、Wasserstein GAN (WGAN-GP)、画像生成 | Ex 17.1〜17.12 (全12問) | 13 枚 | 待機中 (`[ ]`) |
| **18** | **正規化フロー** | 可逆ニューラルネットワーク、ヤコビアン行列式、RealNVP、カップリング層、厳密尤度計算 | Ex 18.1〜18.11 (全11問) | 7 枚 | 待機中 (`[ ]`) |
| **19** | **自己符号化器** | オートエンコーダ、変分オートエンコーダ (VAE)、再パラメータ化トリック、潜在空間正則化 | Ex 19.1〜19.6 (全6問) | 11 枚 | 待機中 (`[ ]`) |
| **20** | **拡散モデル** | 前方向拡散過程、スコアベース生成モデル、逆時間SDE、Denoising Diffusion Probabilistic Models (DDPM) | Ex 20.1〜20.20 (全20問) | 9 枚 | 待機中 (`[ ]`) |

**合計: 全20章中16章実装完了・全79冊ノートブック・演習問題全355問網羅・再現図版300枚超を収録！**

---

## 🔗 ノートブック一覧 (Colabで1クリック起動)

以下のバッジをクリックすると、各ノートブックをGoogle Colab上で直接開いて実行できます。先頭のセットアップセルを実行するだけで即座に学習を開始できます。


### 第1章 深層学習革命 (The Deep Learning Revolution)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/1/1.1_The_Impact_of_Deep_Learning.ipynb) `1.1_The_Impact_of_Deep_Learning.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/1/1.2_A_Tutorial_Example.ipynb) `1.2_A_Tutorial_Example.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/1/1.3_A_Brief_History_of_Machine_Learning.ipynb) `1.3_A_Brief_History_of_Machine_Learning.ipynb`

### 第2章 確率の基礎 (Probabilities)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.1_The_Rules_of_Probability.ipynb) `2.1_The_Rules_of_Probability.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.2_Probability_Densities.ipynb) `2.2_Probability_Densities.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.3_The_Gaussian_Distribution.ipynb) `2.3_The_Gaussian_Distribution.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.4_Transformation_of_Densities.ipynb) `2.4_Transformation_of_Densities.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.5_Information_Theory.ipynb) `2.5_Information_Theory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2.6_Bayesian_Probabilities.ipynb) `2.6_Bayesian_Probabilities.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/2/2_Exercises.ipynb) `2_Exercises.ipynb`

### 第3章 基本分布 (Standard Distributions)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.1_Discrete_Variables.ipynb) `3.1_Discrete_Variables.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.2_The_Multivariate_Gaussian.ipynb) `3.2_The_Multivariate_Gaussian.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.3_Periodic_Variables.ipynb) `3.3_Periodic_Variables.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.4_The_Exponential_Family.ipynb) `3.4_The_Exponential_Family.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3.5_Nonparametric_Methods.ipynb) `3.5_Nonparametric_Methods.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/3/3_Exercises.ipynb) `3_Exercises.ipynb`

### 第4章 単層ネットワーク: 回帰 (Single-layer Networks: Regression)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4.1_Linear_Regression.ipynb) `4.1_Linear_Regression.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4.2_Decision_theory.ipynb) `4.2_Decision_theory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4.3_The_Bias_Variance_Trade_off.ipynb) `4.3_The_Bias_Variance_Trade_off.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/4/4_Exercises.ipynb) `4_Exercises.ipynb`

### 第5章 単層ネットワーク: 分類 (Single-layer Networks: Classification)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.1_Discriminant_Functions.ipynb) `5.1_Discriminant_Functions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.2_Decision_Theory.ipynb) `5.2_Decision_Theory.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.3_Generative_Classifiers.ipynb) `5.3_Generative_Classifiers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5.4_Discriminative_Classifiers.ipynb) `5.4_Discriminative_Classifiers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/5/5_Exercises.ipynb) `5_Exercises.ipynb`

### 第6章 深層ニューラルネットワーク (Deep Neural Networks)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.1_Limitations_of_Fixed_Basis_Functions.ipynb) `6.1_Limitations_of_Fixed_Basis_Functions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.2_Multilayer_Networks.ipynb) `6.2_Multilayer_Networks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.3_Deep_Networks.ipynb) `6.3_Deep_Networks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.4_Error_Functions.ipynb) `6.4_Error_Functions.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6.5_Mixture_Density_Networks.ipynb) `6.5_Mixture_Density_Networks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/6/6_Exercises.ipynb) `6_Exercises.ipynb`

### 第7章 勾配降下法 (Gradient Descent)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.1_Error_Surfaces.ipynb) `7.1_Error_Surfaces.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.2_Gradient_Descent_Optimization.ipynb) `7.2_Gradient_Descent_Optimization.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.3_Convergence.ipynb) `7.3_Convergence.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7.4_Normalization.ipynb) `7.4_Normalization.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/7/7_Exercises.ipynb) `7_Exercises.ipynb`

### 第8章 誤差逆伝播法 (Backpropagation)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/8/8.1_Evaluation_of_Gradients.ipynb) `8.1_Evaluation_of_Gradients.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/8/8.2_Automatic_Differentiation.ipynb) `8.2_Automatic_Differentiation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/8/8_Exercises.ipynb) `8_Exercises.ipynb`

### 第9章 正則化 (Regularization)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.1_Inductive_Bias.ipynb) `9.1_Inductive_Bias.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.2_Weight_Decay.ipynb) `9.2_Weight_Decay.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.3_Learning_Curves.ipynb) `9.3_Learning_Curves.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.4_Parameter_Sharing.ipynb) `9.4_Parameter_Sharing.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.5_Residual_Connections.ipynb) `9.5_Residual_Connections.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9.6_Model_Averaging.ipynb) `9.6_Model_Averaging.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/9/9_Exercises.ipynb) `9_Exercises.ipynb`

### 第10章 畳み込みネットワーク (Convolutional Networks)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.1_Computer_Vision.ipynb) `10.1_Computer_Vision.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.2_Convolutional_Filters.ipynb) `10.2_Convolutional_Filters.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.3_Visualizing_Trained_CNNs.ipynb) `10.3_Visualizing_Trained_CNNs.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.4_Object_Detection.ipynb) `10.4_Object_Detection.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.5_Image_Segmentation.ipynb) `10.5_Image_Segmentation.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10.6_Style_Transfer.ipynb) `10.6_Style_Transfer.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/10/10_Exercises.ipynb) `10_Exercises.ipynb`

### 第11章 構造化分布 (Structured Distributions)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11.1_Graphical_Models.ipynb) `11.1_Graphical_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11.2_Conditional_Independence.ipynb) `11.2_Conditional_Independence.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11.3_Sequence_Models.ipynb) `11.3_Sequence_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/11/11_Exercises.ipynb) `11_Exercises.ipynb`

### 第12章 トランスフォーマー (Transformers)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.1_Attention.ipynb) `12.1_Attention.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.2_Natural_Language.ipynb) `12.2_Natural_Language.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.3_Transformer_Language_Models.ipynb) `12.3_Transformer_Language_Models.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12.4_Multimodal_Transformers.ipynb) `12.4_Multimodal_Transformers.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/12/12_Exercises.ipynb) `12_Exercises.ipynb`

### 第13章 グラフニューラルネットワーク (Graph Neural Networks)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13.1_Machine_Learning_on_Graphs.ipynb) `13.1_Machine_Learning_on_Graphs.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13.2_Neural_Message_Passing.ipynb) `13.2_Neural_Message_Passing.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13.3_General_Graph_Networks.ipynb) `13.3_General_Graph_Networks.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/13/13_Exercises.ipynb) `13_Exercises.ipynb`

### 第14章 サンプリング (Sampling)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14.1_Basic_Sampling_Algorithms.ipynb) `14.1_Basic_Sampling_Algorithms.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14.2_Markov_Chain_Monte_Carlo.ipynb) `14.2_Markov_Chain_Monte_Carlo.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14.3_Langevin_Sampling.ipynb) `14.3_Langevin_Sampling.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/14/14_Exercises.ipynb) `14_Exercises.ipynb`

### 第15章 離散潜在変数 (Discrete Latent Variables)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.1_K_means_Clustering.ipynb) `15.1_K_means_Clustering.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.2_Mixtures_of_Gaussians.ipynb) `15.2_Mixtures_of_Gaussians.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.3_Expectation_Maximization_Algorithm.ipynb) `15.3_Expectation_Maximization_Algorithm.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15.4_Evidence_Lower_Bound.ipynb) `15.4_Evidence_Lower_Bound.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/15/15_Exercises.ipynb) `15_Exercises.ipynb`

### 第16章 連続潜在変数 (Continuous Latent Variables)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.1_Principal_Component_Analysis.ipynb) `16.1_Principal_Component_Analysis.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.2_Probabilistic_Latent_Variables.ipynb) `16.2_Probabilistic_Latent_Variables.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.3_Evidence_Lower_Bound.ipynb) `16.3_Evidence_Lower_Bound.ipynb`
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miyayudai/my_DeepLearning/blob/main/16/16.4_Nonlinear_Latent_Variable_Models.ipynb) `16.4_Nonlinear_Latent_Variable_Models.ipynb`


---

## 🛠️ 開発者・メンテナー向け情報

- **Colab環境セットアップの一括反映**:
  ```bash
  python3 scripts/add_colab_setup.py
  ```
- **統一されたプロットスタイル**:
  すべての可視化スクリプトおよびノートブックは `common.plot_utils.setup_style()` を適用し、論文水準の統一されたフォント（TeX風 Serif）、高解像度（300 DPI）、およびカラーパレットで描画されています。

---

## 📜 ライセンス

本リポジトリのコードおよびノートブックは [MIT License](LICENSE) のもとで公開されています。
独学・研究・大学の講義・勉強会等の教材として自由にご活用ください。

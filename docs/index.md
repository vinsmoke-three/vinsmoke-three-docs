---
title: "vinsmoke-three 機械学習・深層学習の実践的チュートリアル"
description: "PyTorchを使った機械学習・深層学習の実践的なチュートリアル。初心者から上級者まで段階的に学習できる日本語の技術ドキュメント。"
---

# vinsmoke-three ドキュメントへようこそ

**機械学習・深層学習の実践的な学習リソース**

このサイトは、PyTorchを中心とした機械学習・深層学習の知識とノウハウを体系的にまとめた日本語の技術ドキュメントサイトです。

## 🎯 このサイトの特徴

- **実践重視**: 実際に動作するコードと詳細な解説
- **段階的学習**: 初心者から上級者まで、段階的に学習できる構成
- **日本語解説**: 複雑な技術概念をわかりやすい日本語で説明
- **最新技術**: TransformerやVision Transformerなど最新の研究成果を含む実装

---

## 技術スタック

### 機械学習・深層学習
- **PyTorch** - テンソル操作、ニューラルネットワーク、コンピュータビジョン、転移学習、実験追跡、モデルデプロイ

### 大規模言語モデル・自然言語処理
- **Transformers** - アーキテクチャ理解、ライブラリ活用、パイプライン操作、トークナイザー、実践的実装

### データサイエンス
*準備中*

### Web開発
*準備中*

### クラウド・インフラ
*準備中*

---

## 学習ガイド

### 初心者向け
1. [PyTorch fundamentals](PyTorch/01_pytorch_fundamentals.md) - テンソル操作とPyTorchの基本概念
2. [PyTorch workflow](PyTorch/02_pytorch_workflow.md) - 機械学習プロジェクトの基本的な流れ

### 中級者向け
1. [PyTorch classification](PyTorch/03_pytorch_classification.md) - 分類問題の実装
2. [PyTorch computer vision](PyTorch/04_pytorch_computer_vision.md) - コンピュータビジョンとFashionMNIST画像分類
3. [PyTorch custom datasets](PyTorch/05_pytorch_custom_datasets.md) - カスタムデータセットと画像分類モデルの構築

### 上級者向け
1. [PyTorch modular](PyTorch/06_pytorch_modular.md) - コードのモジュール化と再利用可能なMLパイプラインの構築
2. [PyTorch transfer learning](PyTorch/07_pytorch_transfer_learning.md) - 転移学習で事前学習済みモデルを活用した高精度画像分類
3. [PyTorch experiment tracking](PyTorch/08_pytorch_experiment_tracking.md) - TensorBoardを使った実験追跡と複数モデルの体系的比較
4. [PyTorch paper replicating](PyTorch/09_pytorch_paper_replicating.md) - Vision Transformerを一から実装してFoodVision Miniに適用する
5. [PyTorch model deployment](PyTorch/10_pytorch_model_deployment.md) - FoodVision Bigの構築とHugging Face Spacesへの公開

### 大規模言語モデル・自然言語処理

#### 基礎理論
1. [Transformerの図解](LLM/00_illustrated_transformer.md) - 「Attention is All You Need」論文のTransformerアーキテクチャを図解で詳しく解説

#### 実装・実践
2. [Transformer Models](LLM/01_transformer_models.md) - Transformerモデルの基本概念、NLPとLLMの違い、実装の基礎
3. [Using Transformers](LLM/02_using_transformers.md) - Hugging Face Transformersライブラリの実践的な使い方、pipelineの仕組み、モデルとトークナイザーの操作
4. [Fine-tuning a pretrained model](LLM/03_fine_tuning_a_pretrained_model.md) - 事前訓練済みモデルのファインチューニング、Trainer APIとカスタム訓練ループの実装
5. [Hugging Face Tokenizersライブラリの完全ガイド](LLM/04_the_huggingface_tokenizers_library.md) - 高速トークナイザーの仕組みと実装、BPE・WordPiece・Unigramアルゴリズムの詳細解説
6. [GPTをゼロから構築する完全ガイド](LLM/05_Let's_build_GPT_from_scratch.md) - PyTorchでGPTモデルを一から実装。Self-Attention、位置エンコーディング、重み初期化、テキスト生成アルゴリズムまで詳細解説
7. [Hugging Face Datasetsライブラリの完全ガイド](LLM/06_the_huggingface_datasets_library.md) - データセットの読み込み、前処理、保存、FAISSを使った意味検索システムの構築まで実践的に解説

#### Classical NLP Tasks
8. [Token Classification](LLM/ClassicalNLP/71_token_classification.md) - Transformersを使ったトークン分類：固有表現認識（NER）の実践ガイド
9. [Masked Language Modeling](LLM/ClassicalNLP/72_masked_language_modeling.md) - BERTライクなモデルのマスク言語モデルのファインチューニング
10. [Summarization](LLM/ClassicalNLP/73_summarization.md) - Transformersを使ったテキスト要約タスクの実装
11. [Translation](LLM/ClassicalNLP/74_translation.md) - ニューラル機械翻訳モデルの構築と評価
12. [Causal Language Modeling](LLM/ClassicalNLP/75_causal_language_modeling.md) - GPTライクなモデルの訓練とテキスト生成
13. [Question Answering](LLM/ClassicalNLP/76_question_answering.md) - 抽出型質問応答システムの実装

---
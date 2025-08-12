# vinsmoke-three ドキュメントサイト

PyTorch、機械学習、深層学習、大規模言語モデル（LLM）の実践的なチュートリアルとコード例を日本語で提供する技術ドキュメントサイトです。

🌐 **サイトURL**: https://vinsmoke-three.com

## 📚 コンテンツ

### PyTorchチュートリアル

- **[PyTorch fundamentals](docs/PyTorch/01_pytorch_fundamentals.md)** - テンソル操作とPyTorchの基本概念
- **[PyTorch workflow](docs/PyTorch/02_pytorch_workflow.md)** - 機械学習プロジェクトの基本的な流れ
- **[PyTorch classification](docs/PyTorch/03_pytorch_classification.md)** - 分類問題の実装
- **[PyTorch computer vision](docs/PyTorch/04_pytorch_computer_vision.md)** - コンピュータビジョンとFashionMNIST画像分類
- **[PyTorch custom datasets](docs/PyTorch/05_pytorch_custom_datasets.md)** - カスタムデータセットと画像分類モデルの構築
- **[PyTorch modular](docs/PyTorch/06_pytorch_modular.md)** - コードのモジュール化と再利用可能なMLパイプラインの構築
- **[PyTorch transfer learning](docs/PyTorch/07_pytorch_transfer_learning.md)** - 転移学習で事前学習済みモデルを活用した高精度画像分類
- **[PyTorch experiment tracking](docs/PyTorch/08_pytorch_experiment_tracking.md)** - TensorBoardを使った実験追跡と複数モデルの体系的比較
- **[PyTorch paper replicating](docs/PyTorch/09_pytorch_paper_replicating.md)** - Vision Transformerを一から実装してFoodVision Miniに適用する
- **[PyTorch model deployment](docs/PyTorch/10_pytorch_model_deployment.md)** - PyTorchモデルのデプロイメント - FoodVision Bigの構築とHugging Face Spacesへの公開

### 大規模言語モデル・自然言語処理

- **[Transformer Models](docs/LLM/01_transformer_models.md)** - Transformerモデルの基本概念から実装まで、NLPとLLMの違いを理解する
- **[Using Transformers](docs/LLM/02_using_transformers.md)** - Transformersライブラリの使い方 - モデルとトークナイザーの基本
- **[Fine-tuning a pretrained model](docs/LLM/03_fine_tuning_a_pretrained_model.md)** - 事前訓練済みモデルのファインチューニング、Trainer APIとカスタム訓練ループの実装

## 🛠️ 技術スタック

- **ドキュメント生成**: [MkDocs](https://www.mkdocs.org/)
- **テーマ**: [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- **デプロイ**: GitHub Pages
- **CI/CD**: GitHub Actions

## 🚀 ローカル開発

### 前提条件

- Python 3.9+
- Conda または pip

### セットアップ

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/vinsmoke-three/vinsmoke-three-docs.git
   cd vinsmoke-three-docs
   ```

2. **Conda環境の作成とアクティベート**
   ```bash
   conda create -n vinsmoke-three-docs python=3.9
   conda activate vinsmoke-three-docs
   ```

3. **依存関係のインストール**
   ```bash
   pip install mkdocs mkdocs-material
   pip install mkdocs-minify-plugin mkdocs-git-revision-date-localized-plugin
   ```

4. **開発サーバーの起動**
   ```bash
   mkdocs serve
   ```

   ブラウザで http://127.0.0.1:8000 にアクセス

### 利用可能なコマンド

- `mkdocs serve` - 開発サーバーを起動（ホットリロード対応）
- `mkdocs build` - 静的サイトを生成
- `mkdocs gh-deploy` - GitHub Pagesにデプロイ

## 📝 コンテンツの追加方法

1. **新しいMarkdownファイルを作成**
   ```
   docs/カテゴリ名/ファイル名.md
   ```

2. **Front Matterを追加**
   ```yaml
   ---
   title: "ページタイトル"
   description: "ページの説明"
   date: "2025-01-01"
   tags: ["タグ1", "タグ2"]
   ---
   ```

3. **ナビゲーションに追加**
   `mkdocs.yml`の`nav`セクションにページを追加

4. **インデックスページを更新**
   `docs/index.md`に新しいコンテンツへのリンクを追加

## 🎨 SEO最適化

### 実装済みの機能

- **メタデータ**: 各ページに適切なタイトル、説明、キーワードを設定
- **OGP対応**: Open Graph Protocol対応でSNSシェア最適化
- **Twitter Card**: Twitter での表示最適化
- **構造化データ**: JSON-LD形式でサイト情報を構造化
- **サイトマップ**: 自動生成でクローラビリティ向上
- **HTML圧縮**: ページ読み込み速度の最適化
- **日本語検索**: 日本語コンテンツに最適化された検索機能
- **数式表示**: MathJax 3によるLaTeX数式レンダリング

### Google Analytics設定

1. Google AnalyticsでトラッキングIDを取得
2. GitHub Secretsに`GOOGLE_ANALYTICS_KEY`を設定
3. 自動的にサイトに埋め込まれます

## 🚀 デプロイ

### 自動デプロイ

- `main`ブランチへのpushで自動的にGitHub Pagesにデプロイ
- GitHub Actionsを使用した自動化

### 手動デプロイ

```bash
mkdocs gh-deploy
```

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

---

**Note**: このプロジェクトは継続的に更新されています。新しいチュートリアルやコンテンツが定期的に追加されます。

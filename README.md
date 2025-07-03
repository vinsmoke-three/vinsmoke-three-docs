# vinsmoke-three ドキュメントサイト

PyTorch、機械学習、深層学習の実践的なチュートリアルとコード例を日本語で提供する技術ドキュメントサイトです。

🌐 **サイトURL**: https://vinsmoke-three.com

## 📚 コンテンツ

### PyTorchチュートリアル

- **[PyTorch fundamentals](docs/PyTorch/00_pytorch_fundamentals.md)** - テンソル操作とPyTorchの基本概念
- **[PyTorch workflow](docs/PyTorch/01_pytorch_workflow.md)** - 機械学習プロジェクトの基本的な流れ
- **[PyTorch classification](docs/PyTorch/02_pytorch_classification.md)** - 分類問題の実装
- **[PyTorch computer vision](docs/PyTorch/03_pytorch_computer_vision.md)** - コンピュータビジョンとFashionMNIST画像分類
- **[PyTorch custom datasets](docs/PyTorch/04_pytorch_custom_datasets.md)** - カスタムデータセットと画像分類モデルの構築
- **[PyTorch modular](docs/PyTorch/06_pytorch_modular.md)** - コードのモジュール化と再利用可能なMLパイプラインの構築
- **[PyTorch transfer learning](docs/PyTorch/07_pytorch_transfer_learning.md)** - 転移学習で事前学習済みモデルを活用した高精度画像分類

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
- **サイトマップ**: 自動生成でクローラビリティ向上
- **HTML圧縮**: ページ読み込み速度の最適化
- **日本語検索**: 日本語コンテンツに最適化された検索機能
- **構造化データ**: 見出しとナビゲーションの適切な階層化

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

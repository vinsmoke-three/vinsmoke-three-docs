---
name: update-docs
description: 新しい記事を作成・追加した後に、関連ファイル（mkdocs.yml, index.md, README.md）を自動更新する。記事の追加、チュートリアルの新規作成、ドキュメントの追加時に使用。
---

# ドキュメント関連ファイル更新

新しい記事を作成した後、以下の3ファイルを更新して整合性を保つ。

## 更新手順

### 1. 新記事の情報を確認

まず新記事のfront matterから以下を取得する：
- `title` — 記事タイトル
- `description` — 記事の説明
- ファイルパス（例: `docs/PyTorch/11_xxx.md`, `docs/LLM/07_xxx.md`）
- カテゴリ（PyTorch / LLM / LLM/ClassicalNLP / 新規カテゴリ）

### 2. mkdocs.yml を更新

- `nav` セクションに新記事を追加
- 既存のナビゲーション構造と命名規則に従う
  - PyTorch: `- N. PyTorch xxx: PyTorch/NN_xxx.md`
  - LLM: `- N. xxx: LLM/NN_xxx.md`
  - ClassicalNLP: `- xxx: LLM/ClassicalNLP/NN_xxx.md`
- 必要に応じて `extra.tags` に新しいタグを追加

### 3. docs/index.md を更新

- 該当カテゴリの学習ガイドセクションにリンクを追加
- 既存の書式に合わせる：`N. [タイトル](パス) - 説明`
- 適切なレベル（初心者/中級者/上級者）に配置

### 4. README.md を更新

- 該当カテゴリのコンテンツ一覧にリンクを追加
- 既存の書式に合わせる：`- **[タイトル](docs/パス)** - 説明`

## 注意事項

- 各ファイルの既存フォーマットを厳密に踏襲すること
- 番号は連番を維持すること
- 新しいカテゴリの場合は、3ファイルすべてに新セクションを作成すること

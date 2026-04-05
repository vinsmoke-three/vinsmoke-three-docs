---
title: "Superpowers Subagentアーキテクチャ — Agentが Agentを管理する"
description: "Superpowersのsubagent-driven-developmentを解説。Controller / Implementer / Spec Reviewer / Code Quality Reviewerの4つの役割、Promptテンプレートの設計意図、Model選択戦略、dispatching-parallel-agentsとの使い分けを理解する。"
date: 2026-04-05
tags:
  - Claude Code
  - Superpowers
  - Subagent
  - コードレビュー
  - AI Agent
---

# Superpowers Subagentアーキテクチャ — Agentが Agentを管理する

!!! info "この記事で解説するプロジェクト"
    **Superpowers** — Claude Codeの動作をSkillで制御するフレームワーク  
    GitHub: [obra/superpowers](https://github.com/obra/superpowers)

## 基本構造

Superpowersの実装フェーズでは、対話しているメインのエージェント（Controller）は**自分ではコードを書かない**。代わりに、タスクごとに専用のサブエージェントを起動し、実装・レビュー・修正を委任する。

```
Controller（プロジェクトマネージャー）
  │
  │ タスクごとに3つのサブエージェントを起動
  │
  ├─→ Implementer（実装者）
  │     コードを書き、テストし、コミットする
  │
  ├─→ Spec Reviewer（仕様レビュー）
  │     実装が仕様通りか検証する
  │
  └─→ Code Quality Reviewer（品質レビュー）
        コードの品質を検証する
```

この3つは**常にこの順番**で実行される。並行実行はしない。

---

## なぜサブエージェントを使うのか

### 問題: コンテキスト汚染

1つのエージェントに10個のタスクを順番にやらせると、後半のタスクでは前半の議論がコンテキストを埋め尽くす。エージェントは混乱し、品質が下がる。

### 解決: タスクごとに新しいサブエージェント

各サブエージェントは**完全に新しいコンテキスト**で起動する。Controllerが必要な情報だけを厳選して渡す。前のタスクの残骸は一切持ち込まない。

これは「万全の状態のエンジニアが毎回新しくアサインされる」のと同じだ。

---

## Implementer — 実装者

### Promptテンプレートの構造

```markdown
You are implementing Task N: [task name]

## Task Description
[プランから抽出したタスク全文]

## Context
[このタスクが全体のどこに位置するか]

## Before You Begin
質問があれば今聞け。要件、アプローチ、依存関係、不明点。

## Your Job
1. 仕様通りに実装
2. テストを書く（TDD）
3. 動作確認
4. コミット
5. セルフレビュー
6. 報告

## When You're in Over Your Head
止まっていい。悪い仕事より、仕事しない方がマシ。
```

### 設計のポイント

**1. タスク全文をそのまま貼る**

```markdown
[FULL TEXT of task from plan - paste it here, don't make subagent read file]
```

サブエージェントにプランファイルを読ませない。理由は3つ：

- 関係ないタスクまで読んでTokenを無駄にする
- サブエージェントが誤読するリスク
- Controllerが文脈を補足できる

**2. 「止まっていい」と明言する**

> "It is always OK to stop and say 'this is too hard for me.' Bad work is worse than no work."

これがないと、サブエージェントはタスクを理解できなくても無理やり何かを出力する。結果は低品質なコードだ。「ブロックされた」と報告する方がはるかに有益だ。

**3. 4つのステータス**

| ステータス | 意味 | Controllerの対応 |
|---|---|---|
| `DONE` | 完了 | Spec Reviewに進む |
| `DONE_WITH_CONCERNS` | 完了だが懸念あり | 懸念を読み、対処してからReview |
| `NEEDS_CONTEXT` | 情報不足 | 追加情報を渡して再dispatch |
| `BLOCKED` | できない | モデルアップグレード、タスク分割、またはヒューマンへ |

**`BLOCKED`の対処が重要だ。** 同じモデルでリトライしても意味がない。何かを変える必要がある。

---

## Spec Reviewer — 仕様レビュー

### 最も重要な一文

```markdown
## CRITICAL: Do Not Trust the Report

The implementer finished suspiciously quickly. Their report may be 
incomplete, inaccurate, or optimistic. You MUST verify everything 
independently.
```

Spec Reviewerに**Implementerの報告を信じるな**と明示的に伝える。

### 検証する3つの軸

| 軸 | チェック内容 |
|---|---|
| **Missing** | 仕様にあるが実装されていないもの |
| **Extra** | 仕様にないが実装されたもの（過度なエンジニアリング） |
| **Misunderstanding** | 要件を誤解して別のものを作った |

### なぜ信じてはいけないのか

Implementerのセルフレポートは**楽観的バイアス**がかかる。「実装した」と書いてあっても、実際にはコーナーケースを省略していたり、仕様の一部を見落としていることがある。

Spec Reviewerは**コードを直接読んで、仕様と一行ずつ照合する**。レポートの記述は無視する。

---

## Code Quality Reviewer — 品質レビュー

### Spec Reviewの後にのみ実行

```markdown
**Only dispatch after spec compliance review passes.**
```

これが**2段階レビューの核心**だ。

| 順番 | レビュー | 問い |
|---|---|---|
| 1st | Spec Review | **正しいものを作ったか？** |
| 2nd | Code Quality Review | **正しく作ったか？** |

逆順にしたり、1回にまとめてはいけない理由：

- 美しいが間違ったコードを先に品質レビューしても無駄
- 1人のレビュアーが両方見ると、片方が疎かになる

### 追加のチェック項目

標準のコードレビューに加え、以下を確認する。

- 各ファイルが1つの明確な責務を持っているか
- ユニットが独立してテスト可能か
- プランのファイル構造に従っているか
- 新しいファイルがすでに大きくないか

---

## Controllerの役割

Controllerは**自分ではコードを書かない**。やることは：

1. **プランを読み、全タスクを事前に抽出する**
2. **各タスクにコンテキストを付けてサブエージェントに渡す**
3. **サブエージェントの質問に答える**
4. **レビュー結果に基づいてImplementerに修正を指示する**
5. **全タスク完了後、最終レビューを実施する**
6. **`finishing-a-development-branch`に引き継ぐ**

### プロセスの全体フロー

```
プランを読み、全タスクを抽出 → TodoWriteに登録
  │
  ├─ Task 1:
  │   Implementer起動 → 質問があれば回答 → 実装完了
  │   Spec Reviewer起動 → ✅ or ❌（❌なら修正→再レビュー）
  │   Code Quality Reviewer起動 → ✅ or ❌（❌なら修正→再レビュー）
  │   Task完了マーク
  │
  ├─ Task 2: （同上）
  │
  ├─ Task N: （同上）
  │
  └─ 全タスク完了 → 最終コードレビュー → finishing-a-development-branch
```

**Red Flags（やってはいけないこと）：**

- 複数のImplementerを並行dispatch（コンフリクト）
- レビューの片方をスキップ
- Spec Review前にCode Quality Reviewを実行
- サブエージェントの質問を無視
- レビューで問題が出たのに再レビューをスキップ

---

## Model選択戦略

すべてのサブエージェントに最強モデルを使う必要はない。

| タスクの複雑さ | モデル | 判断基準 |
|---|---|---|
| 低（1-2ファイル、明確なspec） | 安価・高速モデル | 機械的な実装が中心 |
| 中（複数ファイル、統合が必要） | 標準モデル | パターンマッチングと判断が必要 |
| 高（アーキテクチャ、設計、レビュー） | 最強モデル | 広い文脈理解と設計判断が必要 |

プランが十分に詳細であれば、**ほとんどのImplementerタスクは安価モデルで十分**だ。コスト削減と速度向上の両方が得られる。

---

## dispatching-parallel-agents との使い分け

Superpowersにはもう一つのサブエージェント系Skillがある。

### subagent-driven-development

- **場面：** 実装プランがある。タスクを順番に実行する
- **実行方式：** 直列（1タスクずつ）
- **レビュー：** あり（2段階）
- **Controllerの役割：** プロジェクトマネージャー

### dispatching-parallel-agents

- **場面：** 独立した複数の問題を同時に解決する
- **実行方式：** 並行（全エージェント同時）
- **レビュー：** なし（各自が報告）
- **Controllerの役割：** ディスパッチャー

### 典型的な使い分け

**subagent-driven-development を使う場面：**

```
プランに基づいて新機能を実装する
  Task 1: データモデル作成
  Task 2: APIエンドポイント作成（Task 1に依存）
  Task 3: フロントエンド作成（Task 2に依存）
```

依存関係があるため直列実行。

**dispatching-parallel-agents を使う場面：**

```
リファクタリング後に3ファイルのテストが壊れた
  login.test.ts     — 認証ロジックの問題
  batch.test.ts     — イベント構造のバグ
  abort.test.ts     — 競合条件
```

原因が独立しているため並行実行。3倍の速度で解決できる。

### 使ってはいけない場面（parallel）

- 失敗が関連している（1つ直せば他も直る可能性）
- エージェント同士が同じファイルを編集する
- 全体のシステム状態を理解する必要がある

---

## executing-plans との関係

`subagent-driven-development`にはもう一つの代替Skillがある。

| | subagent-driven-development | executing-plans |
|---|---|---|
| セッション | 同一セッション内 | 別セッション（引き継ぎ） |
| サブエージェント | 使用する | 使用しない |
| レビュー | 自動（2段階） | 手動チェックポイント |
| 速度 | 高速（自律進行） | 低速（人間の確認待ち） |

`executing-plans`は、サブエージェントが利用できない環境（一部のプラットフォーム）向けのフォールバックだ。Skill本文にもこう書かれている。

> Superpowers works much better with access to subagents. If subagents are available, use superpowers:subagent-driven-development instead of this skill.

---

## まとめ

| 要素 | 役割 |
|---|---|
| Controller | 自分ではコードを書かない。タスクの配分とレビュー管理 |
| Implementer | タスクの実装。止まっていい。4つのステータスで報告 |
| Spec Reviewer | 仕様との照合。Implementerの報告を信じない |
| Code Quality Reviewer | コード品質の検証。Spec Review通過後のみ実行 |
| 2段階レビュー | 「正しいものか」→「正しく作ったか」の順序 |
| Model選択 | タスク複雑度に応じて使い分け。安価モデルで十分な場面が多い |
| parallel agents | 独立した問題の同時解決。依存関係があれば使わない |

次回は最終回。**Skill自体をTDDで開発・テストする方法**を解説する。

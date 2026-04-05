---
title: "Superpowers 概要 — Skillで AI Agent を制御する"
description: "Superpowersの設計思想を解説。Skillが単なるドキュメントではなく「行動制御コード」として機能する仕組み、ワークフロー全体像、従来のプロンプトエンジニアリングとの決定的な違いを理解する。"
date: 2026-04-05
tags:
  - Claude Code
  - Superpowers
  - Skills
  - AI Agent
  - プロンプトエンジニアリング
---

# Superpowers 概要 — Skillで AI Agent を制御する

!!! info "この記事で解説するプロジェクト"
    **Superpowers** — Claude Codeの動作をSkillで制御するフレームワーク  
    GitHub: [obra/superpowers](https://github.com/obra/superpowers)

## Superpowersが解決する問題

Claude Codeは優秀なコーディングエージェントだが、**放っておくと暴走する**。

「機能を追加して」と言えば、設計も確認せずにコードを書き始める。テストを書く前に実装を終わらせる。「直した」と言いながら検証していない。コードレビューは面倒だからスキップする。

これらはエージェントの「悪意」ではない。**最短経路で結果を出そうとする合理的な行動**だ。しかし、ソフトウェア開発において最短経路は最善の経路ではない。

Superpowersは、この問題を**Skill**という仕組みで解決する。

---

## Skillとは何か

多くの人はSkillを「エージェント向けのドキュメント」だと思っている。

**それは間違いだ。**

Skillは**エージェントの行動を制御するコード**である。ドキュメントとの違いは明確だ。

| | ドキュメント | Skill |
|---|---|---|
| 目的 | 情報を伝える | 行動を強制する |
| 語気 | 「〜を推奨します」 | 「YOU MUST」「No exceptions」 |
| テスト | レビューで確認 | TDDで対抗的に検証 |
| 失敗への対応 | 読み直してもらう | 合理化の経路を事前に塞ぐ |

この「行動制御コード」という設計思想が、Superpowers全体を貫く最も重要な概念だ。

---

## CLAUDE.mdとの違い

[Hooks入門の記事](../ClaudeCode/claude_code_hooks.md)で、CLAUDE.mdは「提案」であり、Hooksは「保証」だと書いた。

Skillはその中間に位置する。

```
CLAUDE.md  → 「テストを先に書いてください」     → 提案（従わないことがある）
Hooks      → exit code 2 でブロック            → 保証（物理的に止める）
Skill      → 「YOU MUST」＋合理化防止テーブル    → 強制（心理的に従わせる）
```

Hooksは「物理的なガードレール」で、特定のツール呼び出しをブロックできる。しかし、「テストを先に書く」「設計を確認してから実装する」といった**プロセス全体の制御**はHooksでは不可能だ。

Skillは、エージェントの「判断」そのものに介入する。

---

## 全体のワークフロー

Superpowersは14個のSkillで構成されている。その中核は、以下のパイプラインだ。

```
brainstorming（何を作るか明確にする）
  → writing-plans（実装計画を書く）
    → using-git-worktrees（隔離ブランチを作成）
      → subagent-driven-development（サブエージェントが順番に実装）
        → test-driven-development（RED-GREEN-REFACTOR）
          → requesting-code-review（コードレビュー）
            → verification-before-completion（検証してから完了宣言）
              → finishing-a-development-branch（マージ / PR / クリーンアップ）
```

各Skillの末尾に「次に呼び出すSkill」が明示されており、エージェントが勝手にステップをスキップできないようになっている。

例えば、`brainstorming`のSkillにはこう書かれている。

> **The terminal state is invoking writing-plans.** Do NOT invoke frontend-design, mcp-builder, or any other implementation skill. The ONLY skill you invoke after brainstorming is writing-plans.

この「次のSkillはこれだ、他は呼ぶな」という指示が、ワークフロー全体の強制力を生んでいる。

---

## なぜエージェントは従うのか

ここが最も興味深い部分だ。Skillには**物理的な強制力がない**。Hooksのようにexit code 2でブロックすることはできない。では、なぜエージェントは従うのか。

### 1. SessionStart Hookによる強制注入

Superpowersをインストールすると、`SessionStart` Hookが登録される。新しいセッションが始まるたびに、`using-superpowers`というSkillの**全文**がコンテキストに注入される。

このSkillがエージェントに最初に教えることは：

> **「1%でも適用できるSkillがある可能性があるなら、絶対にそのSkillを呼び出せ」**

これにより、エージェントのデフォルトの行動が「Skillを使わない」から「Skillを使う」に反転する。

### 2. 合理化の経路を事前に塞ぐ

エージェントは賢い。「今回はSkillを使わなくていい」と自分を説得する理由をいくらでも考え出せる。

Superpowersはこれを**Red Flagsテーブル**で対処する。

| エージェントの内心 | 現実 |
|---|---|
| 「これはシンプルな質問だ」 | 質問もタスク。Skillを確認しろ |
| 「先にコンテキストを把握したい」 | Skillの確認はコンテキスト把握の前だ |
| 「Skillは大げさすぎる」 | シンプルなものが複雑になる。使え |
| 「先にこれだけやらせて」 | 何かをやる前にSkillを確認しろ |
| 「Skillの内容は覚えている」 | Skillは進化する。最新版を読め |

これらの「内心」は、実際にエージェントをSkillなしでテストした時に記録された合理化パターンだ。推測ではなく、実験データに基づいている。

### 3. 説得心理学の科学的応用

Superpowersの設計は、Cialdini (2021) の説得心理学と、Meincke et al. (2025) のLLM実験（N=28,000、説得テクニックで遵守率が33%→72%に上昇）に基づいている。

| 原則 | Skillでの応用 |
|---|---|
| **Authority（権威）** | 「YOU MUST」「No exceptions」「Delete means delete」 |
| **Commitment（一貫性）** | Skill使用を宣言させる。TodoWriteでチェックリスト追跡 |
| **Scarcity（緊迫感）** | 「BEFORE proceeding」「IMMEDIATELY after」 |
| **Social Proof（社会的証明）** | 「Every time」「Always」「X without Y = failure」 |
| **Unity（一体感）** | 「your human partner」（userではなくpartner） |

特に注目すべきは**Unity**だ。Superpowersはエージェントのことを「user」ではなく「**your human partner**」と呼ぶ。これは「ツールと利用者」の関係ではなく、「同僚」の関係を構築するためだ。同僚に対しては、忖度よりも正直なフィードバックが優先される。

---

## 従来のプロンプトエンジニアリングとの違い

### アプローチの根本的な差

従来のプロンプトエンジニアリングは「何をさせるか」を記述する。

```markdown
# 従来のアプローチ
テストを先に書いてから実装してください。
TDDのRED-GREEN-REFACTORサイクルに従ってください。
```

Superpowersは「何をさせないか」を記述する。

```markdown
# Superpowersのアプローチ
Write code before the test? Delete it. Start over.

**No exceptions:**
- Don't keep it as "reference"
- Don't "adapt" it while writing tests
- Don't look at it
- Delete means delete
```

従来のアプローチは「正しいことを伝えれば従うだろう」という前提に立っている。Superpowersは「伝えても従わない場合がある」という前提に立ち、**従わない経路を一つずつ塞いでいく**。

### テスト方法の差

従来のSkill開発は「書いて、試して、うまくいけばOK」だ。

Superpowersは**TDD（テスト駆動開発）をSkill自体に適用**する。

```
RED:   Skillなしでエージェントを走らせ、どう失敗するか記録する
GREEN: その失敗パターンに対処するSkillを書く
REFACTOR: 新たな合理化パターンを見つけて塞ぐ
```

Skillを書く前に「エージェントがどうサボるか」を実験で確認し、そのサボり方に合わせてSkillを設計する。逆は許されない。

---

## Skillの構造

各Skillは独立したディレクトリに格納される。

```
skills/
  test-driven-development/
    SKILL.md              # メインファイル（必須）
    testing-anti-patterns.md  # 補足資料（必要に応じて）
  brainstorming/
    SKILL.md
    visual-companion.md
    scripts/
```

`SKILL.md`のフロントマターには、`name`と`description`の2つのフィールドが必須だ。

```yaml
---
name: test-driven-development
description: Use when implementing any feature or bugfix, before writing implementation code
---
```

### descriptionの設計：最も重要な発見

Superpowersが実際のテストで発見した最も重要なルールがある。

**descriptionには「いつ使うか」だけを書き、「何をするか」は絶対に書かない。**

```yaml
# ❌ NG: フローを要約している
description: Use when executing plans - dispatches subagent per task 
  with code review between tasks

# ✅ OK: トリガー条件のみ
description: Use when executing implementation plans with independent 
  tasks in the current session
```

なぜか。テストの結果、descriptionにワークフローを要約すると、**Claudeはdescriptionの要約だけに従い、Skill本文を読み飛ばす**ことが判明した。

例えば、descriptionに「code review between tasks」と書いた場合、Claudeは1回だけレビューを行った。しかし、Skill本文のフローチャートには明確に2段階レビュー（spec compliance → code quality）が定義されていた。descriptionからフロー要約を削除したところ、Claudeは正しく本文を読み、2段階レビューを実行した。

**LLMはショートカットを見つけたら必ずそれを使う。** 設計者の仕事は、ショートカットを存在させないことだ。

---

## まとめ

| 概念 | 要点 |
|---|---|
| Skillの本質 | ドキュメントではなく、行動制御コード |
| 従う理由 | SessionStart Hookで注入 ＋ 合理化防止 ＋ 説得心理学 |
| ワークフロー | brainstorming → plan → TDD → review → finish の強制パイプライン |
| テスト手法 | Skill自体をTDDで開発。エージェントの失敗パターンから逆算 |
| description設計 | トリガー条件のみ。フロー要約はClaudeにショートカットさせる |

次回は、このSkillシステムの入口となる**SessionStart Hookの仕組み**を詳しく解説する。

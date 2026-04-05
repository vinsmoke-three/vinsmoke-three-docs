---
title: "Superpowers Skill設計哲学 — LLMを従わせる技術"
description: "SuperpowersのSkillが行動制御コードとして機能する設計パターンを解説。Iron Law、Red Flagsテーブル、合理化防止、説得心理学の応用、Anthropic公式との設計差異を詳しく掘り下げる。"
date: 2026-04-05
tags:
  - Claude Code
  - Superpowers
  - Skills
  - プロンプトエンジニアリング
  - 説得心理学
---

# Superpowers Skill設計哲学 — LLMを従わせる技術

!!! info "この記事で解説するプロジェクト"
    **Superpowers** — Claude Codeの動作をSkillで制御するフレームワーク  
    GitHub: [obra/superpowers](https://github.com/obra/superpowers)

## この記事のスコープ

[第1回](superpowers_overview.md)でSkillは「行動制御コード」だと書いた。この記事では、その具体的な設計パターンを掘り下げる。

Superpowersの14個のSkillは、大きく2つのカテゴリに分かれる。

| カテゴリ | 例 | 設計方針 |
|---|---|---|
| **Rigid（規律型）** | TDD、verification、debugging | 一字一句従わせる。適応禁止 |
| **Flexible（柔軟型）** | brainstorming、writing-plans | 原則に従い、コンテキストに合わせて適応 |

この記事では、特に設計が高度な**規律型Skill**のパターンを中心に解説する。

---

## パターン1: Iron Law — 絶対的なルール

すべての規律型Skillには、冒頭に**Iron Law**が宣言されている。

**test-driven-development:**
```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

**systematic-debugging:**
```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

**verification-before-completion:**
```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

共通する設計原則：

- **全大文字**で視覚的に目立たせる
- **否定形**（NO ... WITHOUT）で「やってはいけないこと」を先に示す
- **例外なし**。条件分岐なし。曖昧さなし

なぜ否定形なのか。「テストを先に書いてください」は提案に聞こえる。「テストなしのコードは許可しない」は禁止に聞こえる。LLMは禁止に対してより強く反応する。

---

## パターン2: 合理化テーブル — 言い訳を事前に潰す

エージェントは「今回だけは例外」と自分を説得する能力が極めて高い。Iron Lawだけでは、この合理化を防げない。

Superpowersは**Rationalization Table**でこれに対処する。TDD Skillの例：

| 言い訳 | 現実 |
|---|---|
| 「シンプルすぎてテスト不要」 | シンプルなコードも壊れる。テストは30秒 |
| 「後でテスト書く」 | 後から書いたテストは即通過。何も証明しない |
| 「後からテストしても同じ目的を達成できる」 | 後から書いたテストは「何をしたか」の確認。先に書くテストは「何をすべきか」の定義 |
| 「手動テスト済み」 | アドホック≠体系的。記録なし、再実行不可 |
| 「X時間の作業を捨てるのは無駄」 | サンクコスト。信頼できないコードを残す方が無駄 |
| 「TDDは教条的。実用的に」 | TDDこそ実用的。デバッグより早い |

このテーブルの各行は**推測ではなく実験結果**だ。Skillなしでエージェントを走らせ、実際に言った合理化を一語一句記録し、それに対する反論を書いている。

---

## パターン3: Red Flags — 自己チェックリスト

合理化テーブルは「言い訳に対する反論」だ。Red Flagsは、その一歩手前——**「自分が今まさに合理化しようとしている」ことに気づかせる**仕組みだ。

TDD Skillの例：

```markdown
## Red Flags - STOP and Start Over

- Code before test
- Test after implementation
- Test passes immediately
- Can't explain why test failed
- Tests added "later"
- Rationalizing "just this once"
- "I already manually tested it"
- "Tests after achieve the same purpose"
- "It's about spirit not ritual"
- "Keep as reference" or "adapt existing code"
- "Already spent X hours, deleting is wasteful"
- "TDD is dogmatic, I'm being pragmatic"
- "This is different because..."

**All of these mean: Delete code. Start over with TDD.**
```

最後の一行が重要だ。「該当したらどうするか」を曖昧にしない。**全部同じ対応：コードを消して最初からやり直せ。**

---

## パターン4: 「精神 vs 文字」を先に潰す

エージェントが最もよく使う高度な合理化パターンがある。

> 「ルールの文字通りには従っていないが、精神には従っている」

これは強力な論法だ。一見合理的に聞こえるし、多くの場面で実際に正しい。だからこそ危険だ。

Superpowersは、規律型Skillの冒頭に**この論法を先に無効化する一文**を置く。

```markdown
**Violating the letter of the rules is violating the spirit of the rules.**
```

この一文で「精神に従っている」という合理化の経路全体が塞がれる。TDD、debugging、verificationの3つの規律型Skillすべてにこの一文がある。

---

## パターン5: descriptionのトリガー条件設計

[第1回](superpowers_overview.md)で触れたが、ここでより詳しく解説する。

### 発見の経緯

`subagent-driven-development`のdescriptionを以下のように書いていた。

```yaml
description: Use when executing plans - dispatches subagent per task 
  with code review between tasks
```

テストの結果、Claudeは「code review between tasks」を読み、各タスクの間に**1回だけ**レビューを行った。しかし、Skill本文のフローチャートは明確に**2段階レビュー**（spec compliance → code quality）を定義していた。

descriptionを以下に変更したところ、Claudeは正しくSkill本文を読み、2段階レビューを実行した。

```yaml
description: Use when executing implementation plans with independent 
  tasks in the current session
```

### 設計ルール

この発見から導かれるルールは明確だ。

```yaml
# ❌ フローを書いている — Claudeがdescriptionだけで行動する
description: Use for TDD - write test first, watch it fail, 
  write minimal code, refactor

# ✅ トリガー条件のみ — Claudeは本文を読むしかない
description: Use when implementing any feature or bugfix, 
  before writing implementation code
```

**LLMはショートカットを見つけたら必ず使う。** descriptionにフローを書くことは、本文を読み飛ばすショートカットを提供することと同じだ。

---

## パターン6: フローチャートによる決定制御

Superpowersは`Graphviz dot`記法でフローチャートを埋め込んでいる。ただし、**すべてにフローチャートを使うわけではない**。

### 使う場面

- 非自明な分岐がある時（A or B の判断が必要）
- 見落としやすいループがある時（早すぎる終了を防ぐ）
- 「いつAを使い、いつBを使うか」の判断

### 使わない場面

- 参照情報 → テーブル
- コード例 → マークダウン
- 線形の手順 → 番号付きリスト

`brainstorming`のフローチャートの例：

```dot
digraph brainstorming {
    "Explore project context" -> "Visual questions ahead?";
    "Visual questions ahead?" -> "Offer Visual Companion" [label="yes"];
    "Visual questions ahead?" -> "Ask clarifying questions" [label="no"];
    ...
    "User approves design?" -> "Present design sections" [label="no, revise"];
    "User approves design?" -> "Write design doc" [label="yes"];
    ...
    "User reviews spec?" -> "Invoke writing-plans skill" [label="approved"];
}
```

重要なのは**終端ノード**だ。`brainstorming`の終端は`"Invoke writing-plans skill"`であり、本文にもこう書かれている。

> **The terminal state is invoking writing-plans.** Do NOT invoke frontend-design, mcp-builder, or any other implementation skill.

フローチャートと文章の両方で「次のSkillはこれだけ」を強制している。

---

## パターン7: Skill間の強制チェーン

各Skillの末尾に「次に呼び出すSkill」が指定されている。これは設定ファイルではなく、**Skill本文の自然言語**で書かれている。

| 現在のSkill | 次のSkill | 指示文 |
|---|---|---|
| brainstorming | writing-plans | "The ONLY skill you invoke after brainstorming is writing-plans" |
| writing-plans | subagent-driven-development / executing-plans | プランのヘッダーに記載 |
| subagent-driven-development | finishing-a-development-branch | フローチャートの終端ノード |
| executing-plans | finishing-a-development-branch | "REQUIRED SUB-SKILL" |
| systematic-debugging Phase 4 | test-driven-development | "Use the superpowers:test-driven-development skill" |

**「何を呼ぶか」だけでなく「何を呼ぶな」も指定する**のがポイントだ。brainstormingは「writing-plansだけ」と言うだけでなく、「frontend-design、mcp-builderなど他は呼ぶな」と明示的に禁止している。

---

## Anthropic公式ガイドとの比較

Superpowersのリポジトリには、Anthropic公式のSkill設計ガイド（`anthropic-best-practices.md`）が含まれている。両者の設計思想を比較する。

### 一致する点

| 項目 | 公式 | Superpowers |
|---|---|---|
| フロントマター | name + description必須 | 同じ |
| 命名 | gerund形式（-ing） | 同じ |
| 段階的開示 | SKILL.md→サブファイル | 同じ |
| SKILL.md長さ | 500行以内 | 同じ |
| description視点 | 第三人称 | 同じ |
| テスト優先 | 先にevalを作る | 同じ（TDD形式） |

### 明確に異なる点

**1. Claudeの能力への前提**

| | 公式 | Superpowers |
|---|---|---|
| 基本前提 | 「Claude is already very smart」 | 「Claudeは賢いが、賢く怠ける」 |
| 方針 | 余計な説明を省く | 規律型は徹底的に繰り返す |

**2. descriptionの書き方**

| | 公式 | Superpowers |
|---|---|---|
| 推奨 | 「何をするか」＋「いつ使うか」 | 「いつ使うか」**のみ** |
| 理由 | 発見を助ける | フロー要約を書くとClaudeが本文を飛ばす |

**3. 語気**

| | 公式 | Superpowers |
|---|---|---|
| トーン | 穏やかな指導 | 命令と禁止 |
| 例 | 「Consider writing tests first」 | 「YOU MUST. No exceptions. Delete means delete」 |

**4. テスト方法**

| | 公式 | Superpowers |
|---|---|---|
| 形式 | JSON評価ケース（input/output） | 対抗的圧力テスト（複数プレッシャー） |
| 焦点 | 「正しく動くか」 | 「圧力下でも従うか」 |

### なぜ異なるのか

公式ガイドは**汎用的なSkill**（APIドキュメント、ツールリファレンス）を想定している。「PDFからテキストを抽出する方法」を教えるSkillに、合理化防止は不要だ。

Superpowersは**規律型Skill**という、より困難な問題に取り組んでいる。「テストを先に書け」「根本原因を調べてから直せ」といったルールは、エージェントにとってコストが高い。コストが高いルールほど、合理化される。

CLAUDE.mdにもこう書かれている。

> Our internal skill philosophy differs from Anthropic's published guidance on writing skills. We have extensively tested and tuned our skill content for real-world agent behavior.

---

## まとめ：設計パターン一覧

| パターン | 目的 | 例 |
|---|---|---|
| Iron Law | 例外なしの絶対ルール | `NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST` |
| Rationalization Table | 言い訳を事前に反論 | 「後でテスト」→「即通過。何も証明しない」 |
| Red Flags | 合理化の自己検出 | 「just this once = STOP」 |
| 精神 vs 文字の無効化 | 高度な合理化を塞ぐ | 「letter違反 = spirit違反」 |
| descriptionトリガー | 本文の読み飛ばし防止 | トリガー条件のみ記載 |
| フローチャート | 非自明な分岐の制御 | dot記法で終端ノードを明示 |
| Skill間チェーン | ワークフローのスキップ防止 | 「次はwriting-plans。他は呼ぶな」 |

次回は、これらのパターンを実際に使う**Subagentアーキテクチャ**を解説する。

---
title: "Superpowers の入口 — SessionStart Hook がエージェントを支配する仕組み"
description: "Superpowersがセッション開始時にusing-superpowersスキルを強制注入する仕組みを解説。hooks.json → run-hook.cmd → session-start スクリプトの完全な実行チェーンと、クロスプラットフォーム対応の設計を理解する。"
date: 2026-04-05
tags:
  - Claude Code
  - Superpowers
  - Hooks
  - SessionStart
  - Skills
---

# Superpowers の入口 — SessionStart Hook がエージェントを支配する仕組み

!!! info "この記事で解説するプロジェクト"
    **Superpowers** — Claude Codeの動作をSkillで制御するフレームワーク  
    GitHub: [obra/superpowers](https://github.com/obra/superpowers)

## なぜHookが必要なのか

[前回の記事](superpowers_overview.md)で、Superpowersは14個のSkillで構成されていると書いた。

しかし、ここに根本的な問題がある。Claude Codeはセッション開始時にSkillの**名前とdescription**しか読まない。Skill本文は、エージェントが「このSkillを使おう」と判断して初めて読み込まれる。

つまり、エージェントが自発的にSkillを使う気にならなければ、Superpowersの14個のSkillは**存在しないのと同じ**だ。

エージェントのデフォルトの行動は「直接コードを書く」だ。Skill一覧を見て「あ、brainstormingを先にやるべきだな」と自分で判断することは、ほとんどない。

この問題を解決するのが`SessionStart` Hookだ。

---

## 全体の実行チェーン

Superpowersをインストールすると、以下のチェーンが登録される。

```
セッション開始（startup / clear / compact）
  → Claude Codeが hooks.json を読む
    → run-hook.cmd を実行
      → session-start スクリプトを実行
        → using-superpowers/SKILL.md の全文を読み込む
          → JSON形式で additionalContext として出力
            → エージェントのコンテキストに注入される
```

一つずつ見ていく。

---

## ステップ1: hooks.json — トリガーの登録

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "\"${CLAUDE_PLUGIN_ROOT}/hooks/run-hook.cmd\" session-start"
          }
        ]
      }
    ]
  }
}
```

ポイントは`matcher`の値だ。

| イベント | 発火タイミング |
|---|---|
| `startup` | 新しいセッションを開始した時 |
| `clear` | `/clear` コマンドを実行した時 |
| `compact` | コンテキストが圧縮された時 |

`compact`が含まれている理由は重要だ。長い会話でコンテキストが圧縮されると、最初に注入されたSkill内容が消える可能性がある。`compact`でも再注入することで、**会話がどれだけ長くなっても`using-superpowers`が消えない**ことを保証している。

---

## ステップ2: run-hook.cmd — クロスプラットフォーム対応

```bash
: << 'CMDBLOCK'
@echo off
REM Windows: cmd.exeがバッチ部分を実行し、bashを探して呼び出す

if exist "C:\Program Files\Git\bin\bash.exe" (
    "C:\Program Files\Git\bin\bash.exe" "%HOOK_DIR%%~1" %2 %3 %4 %5
    exit /b %ERRORLEVEL%
)
REM ...Git Bash, MSYS2, Cygwin を順番に探す
CMDBLOCK

# Unix: スクリプトを直接実行
exec bash "${SCRIPT_DIR}/${SCRIPT_NAME}" "$@"
```

このファイルは**cmd/bashのポリグロットスクリプト**だ。

- **Windows**: `:` はcmd.exeでは無視される。`@echo off`以降のバッチ部分が実行され、Git BashなどのBash環境を探して`session-start`を呼び出す
- **Unix**: `:`はbashのno-op（何もしない）コマンド。ヒアドキュメントでバッチ部分をスキップし、最後の`exec bash`で直接実行する

Bashが見つからない場合は**サイレントに終了**する（exit 0）。プラグインの他の機能は動作し続け、SessionStartのコンテキスト注入だけがスキップされる。

---

## ステップ3: session-start — 本体

ここが核心だ。このスクリプトは3つのことをする。

### 3-1. SKILL.mdの全文を読み込む

```bash
using_superpowers_content=$(cat "${PLUGIN_ROOT}/skills/using-superpowers/SKILL.md")
```

`using-superpowers/SKILL.md`はフロントマター含めて約120行。この全文がコンテキストに注入される。

### 3-2. JSONに安全にエスケープする

```bash
escape_for_json() {
    local s="$1"
    s="${s//\\/\\\\}"     # バックスラッシュ
    s="${s//\"/\\\"}"     # ダブルクオート
    s="${s//$'\n'/\\n}"   # 改行
    s="${s//$'\r'/\\r}"   # キャリッジリターン
    s="${s//$'\t'/\\t}"   # タブ
    printf '%s' "$s"
}
```

Skill内容をJSON文字列として埋め込むため、特殊文字をエスケープする。bashのパラメータ置換（`${s//old/new}`）を使っており、文字単位のループより高速だ。

### 3-3. プラットフォームに応じた出力

```bash
if [ -n "${CURSOR_PLUGIN_ROOT:-}" ]; then
  # Cursor
  printf '{"additional_context": "%s"}\n' "$session_context"
elif [ -n "${CLAUDE_PLUGIN_ROOT:-}" ] && [ -z "${COPILOT_CLI:-}" ]; then
  # Claude Code
  printf '{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "%s"}}\n' "$session_context"
else
  # Copilot CLI / その他
  printf '{"additionalContext": "%s"}\n' "$session_context"
fi
```

**各プラットフォームが期待するJSONフォーマットが異なる。**

| プラットフォーム | 判定方法 | JSONフォーマット |
|---|---|---|
| Cursor | `CURSOR_PLUGIN_ROOT`が存在 | `additional_context`（スネークケース） |
| Claude Code | `CLAUDE_PLUGIN_ROOT`が存在 + `COPILOT_CLI`なし | `hookSpecificOutput.additionalContext`（ネスト） |
| Copilot CLI | `COPILOT_CLI`が存在 | `additionalContext`（トップレベル） |
| その他 | デフォルト | `additionalContext`（SDK標準） |

!!! warning "重複注入の防止"
    Claude Codeは`additional_context`と`hookSpecificOutput`の**両方を読む**が、重複排除はしない。そのため、Claude Codeの場合はネスト形式のみを出力し、トップレベルの`additional_context`は出力しない。両方出力すると、同じ内容が2回注入されてしまう。

---

## 注入される内容

最終的にエージェントのコンテキストに注入されるのは、以下のような構造だ。

```
<EXTREMELY_IMPORTANT>
You have superpowers.

**Below is the full content of your 'superpowers:using-superpowers' skill
- your introduction to using skills. For all other skills, use the 'Skill' tool:**

---
name: using-superpowers
description: Use when starting any conversation...
---

<EXTREMELY-IMPORTANT>
If you think there is even a 1% chance a skill might apply...
</EXTREMELY-IMPORTANT>

## The Rule
Invoke relevant or requested skills BEFORE any response or action.

## Red Flags
| Thought | Reality |
...

</EXTREMELY_IMPORTANT>
```

注目すべき点：

- **`<EXTREMELY_IMPORTANT>`タグで全体を囲む** — Claude Codeのシステムプロンプトにおいて、このタグは最高優先度のコンテンツであることを示す
- **Skill全文を注入する** — descriptionだけでなく、Red Flagsテーブル、フローチャート、優先度ルールのすべてが含まれる
- **他のSkillは注入しない** — `using-superpowers`だけが特別扱い。他の13個のSkillは、エージェントが必要に応じてSkillツールで読み込む

---

## レガシーSkillディレクトリの検出

`session-start`スクリプトにはもう一つの役割がある。

```bash
legacy_skills_dir="${HOME}/.config/superpowers/skills"
if [ -d "$legacy_skills_dir" ]; then
    warning_message="⚠️ WARNING: Superpowers now uses Claude Code's skills system.
    Custom skills in ~/.config/superpowers/skills will not be read.
    Move custom skills to ~/.claude/skills instead."
fi
```

旧バージョンのSuperpowersは`~/.config/superpowers/skills`にカスタムSkillを配置していた。現在はClaude Code標準の`~/.claude/skills`に移行している。旧ディレクトリが残っている場合、**最初の応答で必ずユーザーに警告する**よう指示が注入される。

---

## なぜSkillの自動スキャンだけでは不十分なのか

Claude Codeは起動時にすべてのプラグインのSkillを自動スキャンし、名前とdescriptionをシステムプロンプトに追加する。

```
The following skills are available for use with the Skill tool:
- superpowers:brainstorming
- superpowers:test-driven-development
- superpowers:writing-plans
...
```

これだけで十分に見えるが、**2つの問題がある**。

### 問題1: エージェントは能動的にSkillを使わない

Skillの存在を知っていることと、実際に使うことは全く違う。

会社の新入社員が「社内マニュアルがあります」と聞いても読まないのと同じだ。エージェントは「機能を追加して」と言われれば、Skill一覧を確認する前にコードを書き始める。

### 問題2: descriptionだけでは行動を変えられない

descriptionはSkillの「いつ使うか」を記述するものだ。「Skillを使う前にまずSkillを確認しろ」というメタレベルの指示は、descriptionの範囲外にある。

`using-superpowers`は**メタSkill**だ。「他のSkillをいつ・どう使うか」を教えるSkillであり、description経由の自然な発見では機能しない。だからSessionStart Hookで**全文を強制注入**する必要がある。

---

## Cursorでの動作

Cursorは別のHooks設定ファイルを使う。

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "command": "./hooks/session-start"
      }
    ]
  }
}
```

Claude Codeの`hooks.json`との違い：

| | Claude Code | Cursor |
|---|---|---|
| matcher | `"startup\|clear\|compact"` | なし（セッション開始時のみ） |
| command | `run-hook.cmd`経由 | `session-start`を直接実行 |
| JSON出力 | `hookSpecificOutput`（ネスト） | `additional_context`（スネークケース） |

`session-start`スクリプト自体は共通で、環境変数（`CURSOR_PLUGIN_ROOT` / `CLAUDE_PLUGIN_ROOT`）で出力フォーマットを切り替える。

---

## まとめ

| 要素 | 役割 |
|---|---|
| `hooks.json` | SessionStartイベントにHookを登録する |
| `run-hook.cmd` | Windows/Unix両対応のポリグロットラッパー |
| `session-start` | SKILL.md全文を読み、プラットフォーム別JSONで出力 |
| `using-superpowers` | エージェントに「Skillを使え」と強制するメタSkill |
| `compact`対応 | コンテキスト圧縮後も再注入し、消失を防ぐ |

Hookが1つしか登録されていないのに、Superpowersの全Skillが機能する。`using-superpowers`が注入されれば、そこから先は**エージェント自身が他のSkillを発見し、呼び出す**からだ。

1つのHookで14個のSkillを支配する。これがSuperpowersの入口設計だ。

次回は、各Skillの中身である**行動制御の設計哲学**を詳しく解説する。

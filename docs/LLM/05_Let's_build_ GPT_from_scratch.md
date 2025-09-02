---
title: "GPTをゼロから構築する完全ガイド - Transformerアーキテクチャの実装と解説"
description: "ChatGPTの仕組みを理解し、PyTorchを使ってGPTモデルをゼロから実装する詳細チュートリアル。Self-Attention、位置エンコーディング、重み初期化、テキスト生成アルゴリズムまで完全解説"
date: "2025-08-26"
tags: ["GPT", "Transformer", "Self-Attention", "PyTorch", "言語モデル", "深層学習", "ニューラルネットワーク", "自然言語処理", "機械学習", "アテンション機構", "位置エンコーディング", "テキスト生成", "オートリグレッシブ", "ChatGPT"]
---

# GPTをゼロから構築する完全ガイド - Transformerアーキテクチャの実装と解説

## 概要

本ガイドでは、**GPT（Generative Pre-trained Transformer）**をPyTorchを使って一から実装する方法を詳しく解説します。単なるコードのコピー&ペーストではなく、各コンポーネントの理論的背景から実装の詳細、最適化手法まで体系的に学習できる内容となっています。

### 学習目標

このガイドを修了することで、以下の知識とスキルを身につけることができます：

**基礎理解**

- ChatGPT・GPTの動作原理とTransformerアーキテクチャの理解
- 文字レベル言語モデリングの仕組み
- トークン化・エンコーディングの設計と実装
- データ準備とバッチ処理の最適化手法

**コア実装技術**

- BigramモデルからGPTへの段階的構築
- Self-Attentionメカニズムの数学的理解と実装
- Multi-Head AttentionとFeed-Forward Networkの構築
- 位置エンコーディングと重み初期化の詳細実装

**高度な技術要素**

- Pre-Norm vs Post-Normアーキテクチャの比較理解
- 残差接続とLayer Normalizationの効果
- オートリグレッシブなテキスト生成アルゴリズム
- 損失関数の設計と訓練ループの最適化

### 前提知識

このガイドを最大限活用するために、以下の知識があることを推奨します：

- **Python プログラミング**: 基本的な構文、クラス、関数の理解
- **PyTorch 基礎**: テンソル操作、自動微分、nn.Moduleの基本概念
- **深層学習の基本**: ニューラルネットワーク、バックプロパゲーション、勾配降下法
- **線形代数**: 行列乗算、ベクトル演算の基本概念
- **確率・統計**: 確率分布、サンプリング手法の理解

!!! info "参考資料"
    本ガイドは [Andrej Karpathy](https://karpathy.ai/) 氏の「Let's build GPT: from scratch, in code, spelled out」を基に、日本語での詳細解説と追加の実践的な内容を加えたものです。

### 実装アーキテクチャ概要

私たちが構築するGPTモデルは以下の主要コンポーネントから構成されています。

![](05_Let's_build_ GPT_from_scratch_files/Full_GPT_architecture.png)

## 1. 言語モデリングの基礎と開発環境構築

### 言語モデリングとは

言語モデリングは、**テキストの確率分布を学習する**機械学習タスクです。具体的には、与えられた文脈に基づいて「次に来る可能性の高い単語（またはトークン）」を予測することを学習します。

**数学的表現**：
```
P(x₁, x₂, ..., xₙ) = ∏ᵢ₌₁ⁿ P(xᵢ|x₁, x₂, ..., xᵢ₋₁)
```

この式は、文章全体の確率を、各単語が前の文脈に条件付けられた確率の積として表現しています。

### なぜ文字レベル言語モデルなのか？

このガイドでは、**文字レベル（Character-level）**の言語モデルを構築していきます。

**利点**:

- **シンプルさ**: 語彙サイズが小さい（英語なら26文字 + 記号）
- **OOV問題の回避**: 未知語（Out-of-Vocabulary）の問題が発生しない
- **理解しやすさ**: トークン化の複雑さを避けて本質に集中できる

**制約**:

- **効率性**: 単語レベルより長いシーケンスが必要
- **意味理解**: 文字レベルから単語の意味を学習する必要がある

### 開発環境とライブラリ

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# GPU利用可能性の確認
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f"使用デバイス: {device}")
```

## 2. データセットの準備とトークン化

### 2.1 データセットの選択：Tiny Shakespeare

**Tiny Shakespeare**を使用する理由：

1. **適度なサイズ**: 約1.1MBで訓練に適している
2. **構造化されたテキスト**: 明確な文法と文体
3. **英語圏での標準**: 多くの研究で使用されている
4. **複雑性のバランス**: 単純すぎず、複雑すぎない

### 2.2 データ準備の詳細プロセス

```python
# データセットのダウンロード
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

```python
# テキストファイルの読み込みと基本情報確認
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
```

**実行結果:**
```
length of dataset in characters:  1115394
```

```python
# 最初の1000文字を確認してデータの内容を把握
print(text[:1000])
```

**実行結果:**
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
```

### 2.3 語彙（Vocabulary）の構築

テキストデータを数値形式に変換するため、まずデータセット内の**全ユニーク文字を特定**します：

```python
# データセット内の全ユニーク文字の抽出と語彙構築
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
```

**実行結果:**
```
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
65
```

### 2.4 トークン化（Tokenization）の実装

コンピュータは文字を直接理解できないため、**文字↔数値の双方向マッピング**を作成します：

#### エンコーディング設計の原則

1. **一意性**: 各文字に唯一のIDを割り当て
2. **可逆性**: 数値から文字への復元が完全に可能
3. **効率性**: シンプルで高速な変換

```python
# 文字→整数、整数→文字のマッピング辞書を作成
stoi = { ch:i for i, ch in enumerate(chars)}
print(stoi)
itos = { i:ch for i, ch in enumerate(chars)}
print(itos)

# エンコーダー関数: 文字列を整数リストに変換
encode = lambda s: [stoi[c] for c in s]
# デコーダー関数: 整数リストを文字列に変換
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))
```

**実行結果:**
```
{'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}
{0: '\n', 1: ' ', 2: '!', 3: '$', 4: '&', 5: "'", 6: ',', 7: '-', 8: '.', 9: '3', 10: ':', 11: ';', 12: '?', 13: 'A', 14: 'B', 15: 'C', 16: 'D', 17: 'E', 18: 'F', 19: 'G', 20: 'H', 21: 'I', 22: 'J', 23: 'K', 24: 'L', 25: 'M', 26: 'N', 27: 'O', 28: 'P', 29: 'Q', 30: 'R', 31: 'S', 32: 'T', 33: 'U', 34: 'V', 35: 'W', 36: 'X', 37: 'Y', 38: 'Z', 39: 'a', 40: 'b', 41: 'c', 42: 'd', 43: 'e', 44: 'f', 45: 'g', 46: 'h', 47: 'i', 48: 'j', 49: 'k', 50: 'l', 51: 'm', 52: 'n', 53: 'o', 54: 'p', 55: 'q', 56: 'r', 57: 's', 58: 't', 59: 'u', 60: 'v', 61: 'w', 62: 'x', 63: 'y', 64: 'z'}
[46, 47, 47, 1, 58, 46, 43, 56, 43]
hii there
```

### 2.5 データセットの準備と分割

```python
# 全テキストデータをPyTorchテンソルに変換
import torch
data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
print(data[:100])  # 最初の100文字の数値表現を確認
```

**実行結果:**
```
torch.Size([1115394]) torch.int64
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 14, 43, 44,
        53, 56, 43,  1, 61, 43,  1, 54, 56, 53, 41, 43, 43, 42,  1, 39, 52, 63,
         1, 44, 59, 56, 58, 46, 43, 56,  6,  1, 46, 43, 39, 56,  1, 51, 43,  1,
        57, 54, 43, 39, 49,  8,  0,  0, 13, 50, 50, 10,  0, 31, 54, 43, 39, 49,
         6,  1, 57, 54, 43, 39, 49,  8,  0,  0, 18, 47, 56, 57, 58,  1, 15, 47,
        58, 47, 64, 43, 52, 10,  0, 37, 53, 59])
```

**訓練/検証分割**: 

訓練用と検証用のデータセットを作成します。
データの最初の90%を訓練データ、最後の10%を検証データとして使用し、モデルの過学習を評価します。

```python
# データを訓練用（90%）と検証用（10%）に分割
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
```

## 3. バッチ処理とベースラインモデル

### 3.1 バッチ処理とシーケンス構築

**チャンク処理**: Transformerは全テキストを一度に処理できないため、データの「小さなランダムなチャンク」をサンプリングして訓練します。

- ブロックサイズ: チャンクの最大長。「コンテキスト長」とも呼ばれる。
- 複数の例のパック: 各チャンクには、文脈とその次のターゲット文字の複数の予測例が含まれる。
- バッチ処理: GPUの効率的な利用のため、「複数のチャンクを単一のテンソルにまとめてスタックする」。

```python
block_size = 8
train_data[:block_size + 1]
```

**実行結果:**
```
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])
```

```python
# 単一のチャンクから複数の訓練例を生成する仕組みを確認
x = train_data[:block_size]
y = train_data[1:block_size+1]
print(y)
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"{t}. when input is {context} the target: {target}")
```

**実行結果:**
```
tensor([47, 56, 57, 58,  1, 15, 47, 58])
0. when input is tensor([18]) the target: 47
1. when input is tensor([18, 47]) the target: 56
2. when input is tensor([18, 47, 56]) the target: 57
3. when input is tensor([18, 47, 56, 57]) the target: 58
4. when input is tensor([18, 47, 56, 57, 58]) the target: 1
5. when input is tensor([18, 47, 56, 57, 58,  1]) the target: 15
6. when input is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47
7. when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58
```

**バッチ処理**

```python
torch.manual_seed(1337)  # 乱数シードを固定して再現性を確保
batch_size = 4           # バッチサイズ（同時に処理するデータ数）
block_size = 8           # 1サンプルのコンテキスト長（トークン数）

def get_batch(split):
    # データセットの選択：trainまたはvalidation
    data = train_data if split == 'train' else val_data
    
    # ランダムな開始位置を生成（batch_size個）
    # len(data) - block_sizeにすることで、境界オーバーを防ぐ
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    
    # 入力シーケンス（x）：各開始位置からblock_size分のデータを取得
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    
    # ターゲットシーケンス（y）：xより1つずつ後ろにシフトしたデータ
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  
    
    return x, y  # (batch_size, block_size)の形状で返す

xb, yb = get_batch("train")  # 訓練データからバッチを取得

print("inputs")
print(xb.shape)              # 入力テンソルの形状を表示
print(xb)                    # 入力テンソルの内容を表示
print("targets")
print(yb.shape)              # ターゲットテンソルの形状を表示
print(yb)                    # ターゲットテンソルの内容を表示

print("--------")

# 各バッチ・各時刻ごとに、与えられたコンテキストとターゲットを表示
for b in range(batch_size): # バッチ次元
    for t in range(block_size): # 時系列次元
        context = xb[b, :t+1]   # t時点までのコンテキスト
        target = yb[b, t]       # 予測すべきターゲット
        print(f"when input is {context.tolist()} the target: {target}")
```

**実行結果:**
```
inputs
torch.Size([4, 8])
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
targets
torch.Size([4, 8])
tensor([[43, 58,  5, 57,  1, 46, 43, 39],
        [53, 56,  1, 58, 46, 39, 58,  1],
        [58,  1, 58, 46, 39, 58,  1, 46],
        [17, 27, 10,  0, 21,  1, 54, 39]])
--------
when input is [24] the target: 43
when input is [24, 43] the target: 58
when input is [24, 43, 58] the target: 5
when input is [24, 43, 58, 5] the target: 57
when input is [24, 43, 58, 5, 57] the target: 1
when input is [24, 43, 58, 5, 57, 1] the target: 46
when input is [24, 43, 58, 5, 57, 1, 46] the target: 43
when input is [24, 43, 58, 5, 57, 1, 46, 43] the target: 39
when input is [44] the target: 53
when input is [44, 53] the target: 56
when input is [44, 53, 56] the target: 1
when input is [44, 53, 56, 1] the target: 58
when input is [44, 53, 56, 1, 58] the target: 46
when input is [44, 53, 56, 1, 58, 46] the target: 39
when input is [44, 53, 56, 1, 58, 46, 39] the target: 58
when input is [44, 53, 56, 1, 58, 46, 39, 58] the target: 1
when input is [52] the target: 58
when input is [52, 58] the target: 1
when input is [52, 58, 1] the target: 58
when input is [52, 58, 1, 58] the target: 46
when input is [52, 58, 1, 58, 46] the target: 39
when input is [52, 58, 1, 58, 46, 39] the target: 58
when input is [52, 58, 1, 58, 46, 39, 58] the target: 1
when input is [52, 58, 1, 58, 46, 39, 58, 1] the target: 46
when input is [25] the target: 17
when input is [25, 17] the target: 27
when input is [25, 17, 27] the target: 10
when input is [25, 17, 27, 10] the target: 0
when input is [25, 17, 27, 10, 0] the target: 21
when input is [25, 17, 27, 10, 0, 21] the target: 1
when input is [25, 17, 27, 10, 0, 21, 1] the target: 54
when input is [25, 17, 27, 10, 0, 21, 1, 54] the target: 39
```

```python
print(xb) # Transformerへの入力データ
```

**実行結果:**
```
tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
```

!!! tip "ランダムサンプリングについて"

    言語モデルの訓練では、データからランダムにblock_size分のシーケンスを取得する。この時、全てのblockが訓練に使われるわけではない。一部のblockは何度も選ばれ、一部は一度も選ばれない可能性がある。
    しかし、これは問題ではない。深層学習は統計学習であり、全てのデータを見る必要はない。言語には多くの重複パターンがあるため、ランダムサンプルでも言語の統計的規則を十分学習できる。また、ランダムサンプリングはデータの局所的相関を打破し、効率的な学習を促進する。
    十分な訓練ステップを行えば、統計的に大部分のblockは最終的にサンプルされることになる。

### 3.2 最もシンプルなモデル: バイグラム言語モデル

最初に、最もシンプルなバイグラム言語モデルを実装して基本概念を理解する：

- **実装**: PyTorchのnn.Moduleでバイグラム言語モデルを実装
- **埋め込み層**: nn.Embeddingを使用して、各トークン（文字）をvocab_size x vocab_sizeの埋め込みベクトルに変換する。入力の各整数が埋め込み層の行を選択する
- **ロジットと損失**: 出力は次の文字の「スコア」（logit）となる。損失は「Negative Log Likelihood Loss」（PyTorchではCrossEntropyLoss）で評価する
- **生成**: 現在のコンテキストに基づいて次のトークンを生成する。softmaxで確率を計算し、torch.multinomialでサンプリングする
- **訓練**: Adamオプティマイザを使用し、損失を最小化するようにモデルパラメータを更新する。バイグラムモデルでは、「入力は完全にランダム」な出力を生成するが、訓練によりいくらか改善される

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 各トークンは埋め込み層から次のトークンのロジットを直接読み取る
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idxとtargetsはどちらも整数の(B, T)テンソル
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idxは現在のコンテキストのインデックスの(B, T)配列
        for _ in range(max_new_tokens):
            # 予測を取得
            logits, loss = self(idx)
            # 最後の時間ステップにのみ焦点を当てる
            logits = logits[:, -1, :] # (B, C)になる
            # softmaxを適用して確率を取得
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 分布からサンプル
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # サンプルされたインデックスを実行シーケンスに追加
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

print(f"inputs: {xb.shape}") # 入力テンソルの形状を表示
print(f"targets: {yb.shape}") # ターゲットテンソルの形状を表示
model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```

**実行結果:**
```
inputs: torch.Size([4, 8])
targets: torch.Size([4, 8])
torch.Size([32, 65])
tensor(4.8786, grad_fn=<NllLossBackward0>)

SKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHjgChzbQ?u!3bLIgwevmyFJGUGp
wnYWmnxKWWev-tDqXErVKLgJ
```

オプティマイザ: AdamW (Adamの改良版) を利用します。

```python
# PyTorchオプティマイザの作成
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

```python
# 訓練ループの実行
batch_size = 32
for steps in range(5000): # より良い結果を得るにはステップ数を増やす...
    # データのバッチをサンプル
    xb, yb = get_batch("train")

    # 損失を評価
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"loss: {loss.item()}")
```

**実行結果:**
```
loss: 4.801066875457764
```

```python
print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
```

**実行結果:**
```
Yc;A-wF-Idfyh$-HSVLzR'QljxYiykGzMGJmKUfN-IJL-mZI-PWT;AnYju'KZRmXc;ha;hEq-epHAA:NJjmyhjd
3Q&vW:PiHA fotx:VFv$VrgCOlvyeE?BK-j,lzoUlx;ALIB;&srWk:PqNgCJW?nOphstDM$zW.:x:a-gfr,LC'KDIEuC -LzRwfqoTF&!wKS.byhixJJi?aUJa,SjNYwJUDnLgBZ!zPYQQ&oiOlv'bFOYGxrDGM?iNubZwJFbqAm,.pSrgi,
W:CVz zlfFC!ufNevU'd f?Y3fAr.uYCJr&!Aqm,p$YCUAbV!PnQhhs;:wiBk!a!!Q!ITh
RwEiMWkzoXn!a!ajn&o;cLgFVTq,Njx.y,cUNfI'qUzdpG;EZvmJMlH;w$ vhzzrgKtdlgM;zPNXrzVvix?LTNXx,ehfVJUZbW -.jxXW:x;i?nQUh&yJh;tQJ$a:Cj$uUEa-PyFERV
oEBCKpF$ehSKgKYIoaiB
```

損失は下がりましたが、予測の内容は全然ダメですね。
それはもっともシンプルなモデルですから、featuresが独立して、関連性がなかったです。






## 4. Self-Attentionメカニズムの構築

### 4.1 Self-Attention（自己注意機構）の理論

#### 効率的な重み付き集約の数学的トリック

まず、行列乗算を使った重み付き集約の仕組みを理解するために、シンプルなトイ例から始めましょう。この例では、**下三角行列**と**重み正規化**を使って、過去のトークンのみを参照する仕組みを作ります。

```python
# 行列乗算が「重み付き集約」にどのように使用されるかを示すトイ例
torch.manual_seed(42)
a_original = torch.tril(torch.ones(3,3))
a = a_original / torch.sum(a_original, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print("a_original=")
print(a_original)
print(f"a={a}")
print(f"b={b}")
print(f"c={c}")
```

**実行結果:**
```
a_original=
tensor([[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]])
a=tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
b=tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
c=tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]])
```

それでは、実際のテンソルを使ってこの概念を確認してみましょう。以下では、**バッチ処理**を含む3次元テンソル（バッチ、時間、チャンネル）で重み付き集約がどのように動作するかを見ていきます。

```python
# 以下のトイ例を考えてみましょう：
torch.manual_seed(1337)
B,T,C = 4,8,2 # バッチ、時間、チャンネル
x = torch.randn(B,T,C)
print(x.shape)
print(x[0])
print(x[0, :2])
```

**実行結果:**
```
torch.Size([4, 8, 2])
tensor([[ 0.1808, -0.0700],
        [-0.3596, -0.9152],
        [ 0.6258,  0.0255],
        [ 0.9545,  0.0643],
        [ 0.3612,  1.1679],
        [-1.3499, -0.5102],
        [ 0.2360, -0.2398],
        [-0.9211,  1.5433]])
tensor([[ 0.1808, -0.0700],
        [-0.3596, -0.9152]])
```

```python
# x[b,t] = mean_{i<=t} x[b,i] を実現したい
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

print(xbow.shape)
print(xbow[0])
```

**実行結果:**
```
torch.Size([4, 8, 2])
tensor([[ 0.1808, -0.0700],
        [-0.0894, -0.4926],
        [ 0.1490, -0.3199],
        [ 0.3504, -0.2238],
        [ 0.3525,  0.0545],
        [ 0.0688, -0.0396],
        [ 0.0927, -0.0682],
        [-0.0341,  0.1332]])
```

```python
# バージョン2: 重み付き集約に行列乗算を使用
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B,T,T)@(B,T,C)----> (B,T,C)
print(torch.allclose(xbow, xbow2))

print(xbow2[0])
```

**実行結果:**
```
True
tensor([[ 0.1808, -0.0700],
        [-0.0894, -0.4926],
        [ 0.1490, -0.3199],
        [ 0.3504, -0.2238],
        [ 0.3525,  0.0545],
        [ 0.0688, -0.0396],
        [ 0.0927, -0.0682],
        [-0.0341,  0.1332]])
```

**形状変換の仕組み**
```
wei.shape = [T, T] = [8, 8]
x.shape = [B, T, C] = [4, 8, 2]
broadcastingにより: [8, 8] → [4, 8, 8]
最終結果: [4, 8, 8] @ [4, 8, 2] → [4, 8, 2]
```

**matrix multiplicationのルール**

- 最後の2次元のみが行列乗算に参加 
- 前の次元はバッチ処理（for loopに相当）
- 4次元なら2重for loop、5次元なら3重for loopと同等

**weiマトリックスの役割**
```
wei = [[1.0, 0.0, 0.0, ...],   # 時刻0: 自分のみ
       [0.5, 0.5, 0.0, ...],   # 時刻1: 0,1の平均  
       [0.33,0.33,0.33,...],   # 時刻2: 0,1,2の平均
       ...]
```

**なぜこんなに巧みなのか？**

各行が「集約ルール」を定義している：
```
第0行：自分のみを見る → [1, 0, 0, 0]
第1行：自分と前の1つを見る → [0.5, 0.5, 0, 0]
第2行：自分と前の2つを見る → [1/3, 1/3, 1/3, 0]
```
matrix multiplicationがこれらのルールを自動適用する！

これがmatrix multiplicationが深層学習でこれほど中核的な理由 - 複雑な集約操作を優雅に表現できる！

次に、**Softmax関数**を使った重み付きを実装してみましょう。Softmaxは、マスキングされていない部分の重みを正規化し、より安定した学習を可能にします。この手法は、実際のTransformerモデルで使用されています。

```python
# バージョン3: Softmaxを使用
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros(T,T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
xbow3 = wei @ x
print(torch.allclose(xbow,xbow3))
print(xbow3[0])
```

**実行結果:**
```
True
tensor([[ 0.1808, -0.0700],
        [-0.0894, -0.4926],
        [ 0.1490, -0.3199],
        [ 0.3504, -0.2238],
        [ 0.3525,  0.0545],
        [ 0.0688, -0.0396],
        [ 0.0927, -0.0682],
        [-0.0341,  0.1332]])
```

#### セルフアテンションヘッドの構築

セルフアテンションヘッドは、Transformerモデルの心臓部であり、文脈を理解するために各トークン（単語や文字）が他のすべてのトークンとの関係を計算する仕組みです。このプロセスは、以下のステップで進みます。

**1. クエリ、キー、バリューの生成**

まず、入力された各トークンは、自身の情報を基に3つの異なるベクトル**「クエリ（Query）」、「キー（Key）」、「バリュー（Value）」**を生成します。これらはそれぞれ以下の役割を担います。

- クエリ（Query）: 「私は何を探しているか？」という問いを表します。
- キー（Key）: 「私が何を持っているか？」という情報を含みます。
- バリュー（Value）: 「何を伝えるべき情報か？」という内容そのものです。

このプロセスでは、nn.Linearレイヤーを使って、入力Xからkey_linear、query_linear、value_linearという3つのベクトルが生成されます。

**2. アフィニティ（関連度）の計算**

次に、各トークンのクエリと、他のすべてのトークンのキーの内積（ドット積）を計算します。これにより、それぞれのトークンが互いにどれだけ関連しているかを示す**「アフィニティ行列」**が生成されます。

この行列は、モデルがどのトークンを「重要」と見なすべきかを判断する際の土台となります。特に、B x T x head_sizeのクエリとB x head_size x Tのキーを乗算することで、B x T x Tという形状のアフィニティ行列weiが生成されます。この計算により、バッチ内のデータごとに異なる関連度が計算され、データに依存した柔軟な相互作用が可能になります。

**3. スケーリングとマスキング**

計算されたアフィニティweiは、sqrt(head_size)で割ることでスケーリングされます。これは、softmax関数を適用する際に値が極端に大きくなりすぎるのを防ぎ、学習を安定させるための重要なステップです。

また、文章生成などのタスクでは、未来の情報を参照してしまわないように、マスキングが適用されます。下三角行列を使うことで、各トークンが自身より前のトークンにのみ注意を向けるようになり、予測が過去の文脈にのみ基づく**「オートリグレッシブ」**な性質を保ちます。

**4. ソフトマックスとバリューの集約**

スケーリングとマスキングの処理を終えたアフィニティweiにsoftmaxを適用し、それぞれの関連度を正規化された確率分布に変換します。これにより、関連度が0から1の範囲に収まり、**「どれだけ注意を向けるか」**という重み付けが明確になります。

最後に、この正規化された重みweiと、各トークンのバリューを行列乗算します。この処理によって、関連性の高いトークンのバリューが「集約」され、文脈を考慮した新たなベクトルが出力されます。このベクトルこそが、セルフアテンションヘッドが「このヘッドの目的のために集約された情報」です。

```python
# バージョン4: セルフアテンション！
torch.manual_seed(1337)
B,T,C = 4,8,32 # バッチ、時間、チャンネル
x = torch.randn(B,T,C)

# 単一のヘッドでセルフアテンションを実行してみよう
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16)@(B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1) # [4, 8, 8]

v = value(x) # [4, 8, 16]
out = wei @ v #  [4, 8, 8]@ [4, 8, 16] -> [4, 8, 16]

wei[0]
```

**実行結果:**
```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
        [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
        [0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
        [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]],
       grad_fn=<SelectBackward0>)
```

#### アテンションに関する注意点

- **通信メカニズム**: アテンションは、指向性グラフ内のノード間の通信メカニズムとして機能する
- **空間の概念がない**: アテンションはデフォルトで空間の概念を持たないため、位置エンコーディングが必要
- **バッチ次元の独立性**: バッチ内の各例は独立して処理される
- **デコーダーブロック**（我々の実装）: 未来のトークンからの通信をマスクする（オートリグレッシブ）
- **エンコーダーブロック**: 全てのノードが完全に通信できる（マスクなし）
- **自己注意（Self-Attention）**: クエリ、キー、バリューが全て同じソースから生成される
- **交差注意（Cross-Attention）**: クエリが自身のソースから、キーとバリューが外部ソースから生成される（例：エンコーダー・デコーダーTransformer）

### 4.2 Self-Attentionヘッドの統合

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# ハイパーパラメータの設定
batch_size = 16 # 並列処理する独立シーケンスの数
block_size = 32 # 予測のための最大コンテキスト長
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)
```

```python
def get_batch(split):
    # データセットの選択：trainまたはvalidation
    data = train_data if split == 'train' else val_data
    
    # ランダムな開始位置を生成（batch_size個）
    # len(data) - block_sizeにすることで、境界オーバーを防ぐ
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    
    # 入力シーケンス（x）：各開始位置からblock_size分のデータを取得
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    
    # ターゲットシーケンス（y）：xより1つずつ後ろにシフトしたデータ
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  
    x, y = x.to(device), y.to(device)
    return x, y  # (batch_size, block_size)の形状で返す
```

```python
class Head(nn.Module):
    """一つの自己注意ヘッド"""

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # サイズ(batch, time-step, channels)の入力
        # サイズ(batch, time-step, head size)の出力
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # アテンションスコア（「アフィニティ」）を計算
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # バリューの重み付き集約を実行
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
```

!!! tip "なぜ√d_kで除算するのか？"
    これはスケール内積注意力の重要な技術：

    - 次元d_kが大きい時、内積の分散が大きくなる
    - 大きな内積値はsoftmax後に極端な重み分布を生成
    - √d_kで除算することで内積の分散を制御し、勾配をより安定させる
    数学的に：ベクトル要素が独立同分布の場合、d_k次元内積の分散は約d_kです。

## 5. マルチヘッドアテンションとTransformerブロック

### 5.1 Multi-Head Attention

複数の自己注意ヘッドを並列に実行し、その結果を連結する。
各ヘッドは異なる「通信チャネル」として機能し、異なるタイプの特徴を学習できます。

```python
class MultiHeadAttention(nn.Module):
    """複数の自己注意ヘッドを並列実行"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

### 5.2 Feed-Forward Network (MLP)

マルチヘッド自己注意の後に配置されるシンプルな多層パーセプトロン（MLP）。

- nn.Linear、ReLU非線形性で構成される
- トークンは、通信で得た情報を個別に「考える」ための時間と計算能力を得る
- トークンごとに独立して適用される

```python
class FeedFoward(nn.Module):
    """非線形性を伴うシンプルな線形層"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
```

### 5.3 Transformerブロックの構築と最適化

Transformerブロック:Multi-Head Self-AttentionとFeed-Forward Networkを組み合わせたもの。

通信（Self-Attention）と計算（Feed-Forward）を交互に行います。

複数のブロックをシーケンシャルに適用することで、より深いネットワークを構築します。

**残差接続（Residual Connections / Skip Connections）**: x = x + self.attention(x)のように、変換されたデータに入力データを直接加算します。

勾配がネットワークの深い層まで妨げられずに流れる「スーパーハイウェイ」を提供し、最適化を劇的に改善します。

ブロック内の各部分の後に適用されます。

出力層への「プロジェクション」（nn.Linear）を導入します。

**レイヤー正規化（Layer Normalization）**: batch_normに似ていますが、バッチ次元ではなく特徴次元（行）に沿って正規化を行います。

**ドロップアウト（Dropout）**: 訓練中にランダムに一部のニューロンを無効にする（ゼロに設定する）正則化技術です。

```python
class Block(nn.Module):
    """Transformerブロック: 通信の後に計算を行う"""

    def __init__(self, n_embd, n_head):
        # n_embd: 埋め込み次元, n_head: 使用したいヘッド数
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差接続
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

**Pre-Norm vs Post-Norm アーキテクチャの比較**

**原始Transformer (Post-Norm, 2017)**
```python
# Attention is All You Need の設計
x = embedding + positional_encoding
x = self.ln1(x + self.sa(x))        # LayerNormが残差接続の後
x = self.ln2(x + self.ffwd(x))      # LayerNormが残差接続の後
```

**現代GPT (Pre-Norm)**
```python
# GPTシリーズの設計
x = embedding + positional_encoding  
x = x + self.sa(self.ln1(x))        # LayerNormがモジュールの前
x = x + self.ffwd(self.ln2(x))      # LayerNormがモジュールの前
```

**なぜ変化したのか？**

**Post-Normの問題点:**
- 深層ネットワークで勾配消失が起こりやすい
- 勾配がLayerNormを通過する際に弱くなる
- 学習率のウォームアップが必要

**Pre-Normの利点:**
- 直接的な勾配パス: x = x + module(ln(x)) で恒等写像が保証される
- 深層ネットワークの安定訓練: 12層以上でも安定
- 学習の高速化: より大きな学習率を使用可能
- 勾配爆発の抑制: 数値的により安定

**実用的な違い**
- 原始Transformer: 機械翻訳など（比較的浅層）
- GPT: 言語生成タスク（深層ネットワーク必須）

**現在の主流**

現代の大規模言語モデル（GPT、BERT変種など）は、深層ネットワークの訓練安定性のためPre-Normアーキテクチャを採用している。

### 5.4 位置エンコーディング（Positional Encoding）の重要性

Transformerアーキテクチャでは、**位置エンコーディング**が極めて重要な役割を果たします。

#### なぜ位置情報が必要なのか？

**1. Self-Attentionの性質**
   
Self-Attentionは全てのトークン間の関係を並列計算しますが、この処理には**順序の概念**がありません。言い換えると、「私は学校に行く」と「学校に私は行く」を同じものとして扱ってしまう可能性があります。

**2. バッグオブワード問題**
   
位置情報がない場合、モデルは単語の「袋（bag）」として文章を理解し、文法や語順による意味の違いを捉えることができません。

**3. 文脈の理解**
   
言語において、単語の位置は意味に大きく影響します。例えば：
- 「田中さんは山田さんを紹介した」
- 「山田さんは田中さんを紹介した」
   
これらは語順によって全く意味が異なります。

#### 位置エンコーディングの仕組み

```python
# 位置エンコーディングの実装例
pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
x = tok_emb + pos_emb # トークン埋め込みと位置埋め込みの加算
```

この加算により、各トークンは「何の単語か」と「どの位置にあるか」の両方の情報を持つことになります。

!!! tip "学習可能な位置エンコーディング vs 固定位置エンコーディング"
    - **学習可能**（我々の実装）: nn.Embeddingで位置ごとに異なるベクトルを学習
    - **固定式**（原論文）: sin/cos関数を使用した数学的パターン
    
    学習可能な方式は、特定のデータセットに最適化される利点がありますが、最大系列長が固定されるという制約があります。

## 6. 完全なGPTモデルの構築

### 6.1 GPTモデルの実装

以下では、これまで学習したすべてのコンポーネントを統合してGPTモデルを実装します。

**主要コンポーネント**:

- トークン埋め込み（Token Embedding）
- 位置埋め込み（Position Embedding）  
- Multi-Head Self-Attention
- Feed-Forward Networks
- Layer Normalization
- 残差接続（Residual Connections）
- 適切な重み初期化

以下では、実際のGPTモデルの実装を通してこれらを統合していきます：

```python
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 各トークンは、埋め込み層から次のトークンのロジットを直接読み取る
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embbedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # 最終レイヤー正規化
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 重み初期化は訓練の成功に極めて重要
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idxとtargetsはどちらも整数の(B, T)テンソル
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embbedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idxは現在のコンテキストのインデックスの(B, T)配列
        for _ in range(max_new_tokens):
            # idxを最後のblock_sizeトークンにクロップ
            idx_cond = idx[:, -block_size:]
            # 予測を取得
            logits, loss = self(idx_cond)
            # 最後の時間ステップにのみ焦点を当てる
            logits = logits[:, -1, :] # (B, C)になる
            # softmaxを適用して確率を取得
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 分布からサンプル
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # サンプルされたインデックスを実行シーケンスに追加
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```

### 6.2 トレーニングループの構築

GPTモデルが完成したので、実際にトレーニングを行うためのループを構築していきます。効果的な訓練には以下の要素が必要です：

- **損失評価関数**: 訓練データと検証データでの性能測定
- **オプティマイザ設定**: AdamWオプティマイザによる勾配更新
- **訓練ループ**: バッチごとの順伝播・逆伝播・重み更新

まず、モデルをインスタンス化し、パラメータ数を確認します：
```

```python
model = GPTLanguageModel()
m = model.to(device)
# モデル内のパラメータ数を出力
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
```

**実行結果:**
```
0.209729 M parameters
```

#### パラメータ数の詳細内訳

モデルサイズをより詳しく理解するために、各コンポーネントのパラメータ数を計算してみます：

**パラメータ数の内訳**（設定例：vocab_size=65, n_embd=64, n_head=4, n_layer=4）

```
Token Embedding:        65 × 64 = 4,160    パラメータ
Position Embedding:     32 × 64 = 2,048    パラメータ
各Transformer Block:             ~49,400   パラメータ
4つのBlock合計:       49,400 × 4 = 197,600 パラメータ
最終出力層(lm_head):    64 × 65 = 4,160    パラメータ
Layer Normalization:           少量        パラメータ
────────────────────────────────────────────────────
総計:                          ~0.21M     パラメータ
```

**Transformer Block内の詳細**：
```
Multi-Head Attention:
- Query/Key/Value線形層: 64×64×3 = 12,288
- 出力プロジェクション: 64×64 = 4,096
Feed-Forward Network:
- 第1層: 64×256 = 16,384
- 第2層: 256×64 = 16,384
Layer Normalization ×2: 64×2 = 128
────────────────────────────────────
Block合計:              約49,400
```

この計算により、モデルのサイズが把握でき、メモリとGPU使用量の目安がわかります。

#### 損失評価関数の実装

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

#### メインの訓練ループ

訓練データと検証データで定期的に損失を評価しながら、AdamWオプティマイザでモデルを訓練します：

```python
# PyTorchオプティマイザの作成
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # 定期的に訓練セットと検証セットで損失を評価
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # データのバッチをサンプル
    xb, yb = get_batch('train')

    # 損失を評価
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

**実行結果:**
```
step 0: train loss 4.1959, val loss 4.1962
step 100: train loss 2.6228, val loss 2.6158
step 200: train loss 2.4580, val loss 2.4510
step 300: train loss 2.3817, val loss 2.3929
step 400: train loss 2.3220, val loss 2.3230
step 500: train loss 2.2391, val loss 2.2578
step 600: train loss 2.1846, val loss 2.2275
step 700: train loss 2.1361, val loss 2.1595
step 800: train loss 2.1026, val loss 2.1439
step 900: train loss 2.0587, val loss 2.1057
step 1000: train loss 2.0449, val loss 2.0912
step 1100: train loss 2.0026, val loss 2.0696
step 1200: train loss 1.9823, val loss 2.0701
step 1300: train loss 1.9682, val loss 2.0451
step 1400: train loss 1.9421, val loss 2.0375
step 1500: train loss 1.9029, val loss 1.9990
step 1600: train loss 1.8860, val loss 1.9939
step 1700: train loss 1.8778, val loss 1.9785
step 1800: train loss 1.8865, val loss 1.9925
step 1900: train loss 1.8428, val loss 1.9692
step 2000: train loss 1.8349, val loss 1.9544
step 2100: train loss 1.8397, val loss 1.9828
step 2200: train loss 1.8097, val loss 1.9398
step 2300: train loss 1.8034, val loss 1.9342
step 2400: train loss 1.7814, val loss 1.9171
step 2500: train loss 1.7723, val loss 1.9080
step 2600: train loss 1.7570, val loss 1.8970
step 2700: train loss 1.7583, val loss 1.9077
step 2800: train loss 1.7523, val loss 1.8983
step 2900: train loss 1.7405, val loss 1.8941
step 3000: train loss 1.7512, val loss 1.8875
step 3100: train loss 1.7413, val loss 1.8903
step 3200: train loss 1.7300, val loss 1.8864
step 3300: train loss 1.7250, val loss 1.8710
step 3400: train loss 1.7151, val loss 1.8758
step 3500: train loss 1.7164, val loss 1.8638
step 3600: train loss 1.7095, val loss 1.8521
step 3700: train loss 1.6988, val loss 1.8557
step 3800: train loss 1.7032, val loss 1.8610
step 3900: train loss 1.6901, val loss 1.8394
step 4000: train loss 1.6839, val loss 1.8307
step 4100: train loss 1.6743, val loss 1.8447
step 4200: train loss 1.6673, val loss 1.8499
step 4300: train loss 1.6743, val loss 1.8297
step 4400: train loss 1.6674, val loss 1.8384
step 4500: train loss 1.6638, val loss 1.8206
step 4600: train loss 1.6467, val loss 1.8364
step 4700: train loss 1.6564, val loss 1.8121
step 4800: train loss 1.6547, val loss 1.8267
step 4900: train loss 1.6436, val loss 1.8231
step 4999: train loss 1.6491, val loss 1.8256
```

#### 訓練済みモデルでのテキスト生成

訓練が完了したので、モデルを使って新しいテキストを生成してみましょう：

```python
# モデルから生成
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
```

**実行結果:**
```
God, O, that of you: life, whereep, them with
force this cordon'd his encleours.
Thou thy enleiter judget lut womal
To him; haver core' venray,
Bon the wrights a cause in thisfiend prepossity treak his sound,
And boy pring lack marrougs maste I hand'd naccuse,
That that keep reasw the underer you: it
That fight thee
fall Julatest: God! hatt 't a bed's mear acannot.
Nos weld?

DUKE VINCESD IVERS:
Mere folies a gentrefarefien a fawaither wither in bawdel,--sain,
In centerfore to hope curce mame court us:
sell not forf.

PRINCES:
Nob tull teare them one
Ongive me
Where pat here appeased Edperate,
But thou mean; nastesders home on wletgeedate to frise tense,
Come a in on this friend 
With polows
Julate both our tran in there
mour worthern: it he sout at I cried!
By gent him to the god poors, time same a dal Bowery.

CLAUDIO:
O, bearech it the is wis to Lorde,
What, he bare he of the bund-heddeige,
That laie agg to that you usworld that shall ound to wack.

That hopose.

WARWICKBERS:
O, for
```

## 7. 最終実装とテキスト生成

### 7.1 統合されたGPTモデルの実装

ここまでで学習したすべてのコンポーネントを統合し、**完全なGPTモデル**を構築します。以下のコードには、これまで段階的に説明してきた全ての要素が含まれています：

- 文字レベルトークン化システム
- マルチヘッド自己注意メカニズム
- フィードフォワードネットワーク
- 位置エンコーディング
- 残差接続とLayer Normalization
- 適切な重み初期化

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# ハイパーパラメータ
batch_size = 64 # 並列処理する独立シーケンスの数
block_size = 256 # 予測のための最大コンテキスト長
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# このテキストに出現する全てのユニーク文字
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 文字から整数へのマッピングを作成
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # エンコーダー: 文字列を取得し、整数のリストを出力
decode = lambda l: ''.join([itos[i] for i in l]) # デコーダー: 整数のリストを取得し、文字列を出力

# 訓練・テスト分割
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 最初の90%が訓練、残りが検証
train_data = data[:n]
val_data = data[n:]

# データローディング
def get_batch(split):
    # 入力xとターゲットyの小さなデータバッチを生成
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ 一つの自己注意ヘッド """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # サイズ(batch, time-step, channels)の入力
        # サイズ(batch, time-step, head size)の出力
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # アテンションスコア（「アフィニティ」）を計算
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # バリューの重み付き集約を実行
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ 複数の自己注意ヘッドを並列実行 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 非線形性を伴うシンプルな線形層 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformerブロック: 通信の後に計算を行う """

    def __init__(self, n_embd, n_head):
        # n_embd: 埋め込み次元, n_head: 使用したいヘッド数
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 各トークンは、埋め込み層から次のトークンのロジットを直接読み取る
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # 最終レイヤー正規化
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # より良い初期化、元のGPTビデオでは扱わなかったが重要、フォローアップビデオで扱う予定
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        GPT式重み初期化：標準偏差0.02の正規分布を使用
        この値はTransformerアーキテクチャに最適化された経験的結果
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idxとtargetsはどちらも整数の(B,T)テンソル
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idxは現在のコンテキストのインデックスの(B, T)配列
        for _ in range(max_new_tokens):
            # idxを最後のblock_sizeトークンにクロップ
            idx_cond = idx[:, -block_size:]
            # 予測を取得
            logits, loss = self(idx_cond)
            # 最後の時間ステップにのみ焦点を当てる
            logits = logits[:, -1, :] # (B, C)になる
            # softmaxを適用して確率を取得
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 分布からサンプル
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # サンプルされたインデックスを実行シーケンスに追加
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
model = GPTLanguageModel()
m = model.to(device)
# モデル内のパラメータ数を出力
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# PyTorchオプティマイザの作成
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # 定期的に訓練セットと検証セットで損失を評価
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # データのバッチをサンプル
    xb, yb = get_batch('train')

    # 損失を評価
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# モデルから生成
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

**実行結果:**
```
--2025-08-28 21:14:55--  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
raw.githubusercontent.com (raw.githubusercontent.com) をDNSに問い合わせています... 2606:50c0:8001::154, 2606:50c0:8000::154, 2606:50c0:8003::154, ...
raw.githubusercontent.com (raw.githubusercontent.com)|2606:50c0:8001::154|:443 に接続しています... 接続しました。
HTTP による接続要求を送信しました、応答を待っています... 200 OK
長さ: 1115394 (1.1M) [text/plain]
`input.txt' に保存中

input.txt           100%[===================>]   1.06M  --.-KB/s 時間 0.1s       

2025-08-28 21:14:56 (7.62 MB/s) - `input.txt' へ保存完了 [1115394/1115394]

10.788929 M parameters
step 0: train loss 4.2221, val loss 4.2306
step 500: train loss 1.7444, val loss 1.9058
step 1000: train loss 1.3914, val loss 1.5998
step 1500: train loss 1.2659, val loss 1.5262
step 2000: train loss 1.1856, val loss 1.5041
step 2500: train loss 1.1206, val loss 1.4968
step 3000: train loss 1.0732, val loss 1.4862
step 3500: train loss 1.0168, val loss 1.5054
step 4000: train loss 0.9598, val loss 1.5148
step 4500: train loss 0.9094, val loss 1.5358
step 4999: train loss 0.8632, val loss 1.5685

a man one of a most wit that; a thousands
with a most opiny. And a jest but
to the fish self-bond, a disesimple; a monster.

Shepherd:
Lets for him, he desires and man of dain impojects. I reason
to them, fell fear out of the sight: the Good take it advantage
to your agnoly then bitine home. Stand, for for
mine with Bianca. Powpan, the drink that you onceed his
stair and the mair people with him: he's a song of this cause
to live Escalus; he lies and kneel I
require out absen; and ever
be a judg
```

### 7.2 テキスト生成アルゴリズムの詳細解説

GPTがどのようにテキストを生成するのか、ステップごとに詳しく見ていきましょう。

#### なぜGPTはテキストを生成できるのか？

GPTは**「次に来る単語（文字）を予測する」**ように訓練されています。この予測能力を使って、以下のように文章を生成します：

**具体例で理解する生成プロセス**：
```
Step 1: 入力「Hello, my name is」→ GPTが予測「John」
Step 2: 入力「Hello, my name is John」→ GPTが予測「.」  
Step 3: 入力「Hello, my name is John.」→ GPTが予測「I」
Step 4: 入力「Hello, my name is John. I」→ GPTが予測「am」
...このプロセスを繰り返す
```

これを**オートリグレッシブ生成**と呼びます。「自分の出力を次の入力に使う」という意味です。

#### GPT生成メソッドの5つのステップ

それでは、実際のコードを見ながら、GPTがどのように1つずつ文字を生成するかを詳しく解説します。

**Step 1: 入力の長さを制限する**
```python
idx_cond = idx[:, -block_size:]  # 最新のblock_size個のトークンのみ使用
```

**なぜ長さを制限するの？**

- **メモリの節約**: 長い文章全体を覚える必要がない
- **処理速度**: 短い文章の方が計算が速い
- **学習時と同じ条件**: 訓練時も同じ長さで学習したから

**例**: 本が長くても、最新の8ページだけ読んで次を予測する感じです
```
元の文章: [1,2,3,4,5,6,7,8,9,10,11] （11文字）
↓ block_size=8の場合
使用する部分: [4,5,6,7,8,9,10,11] （最新8文字のみ）
```

**Step 2: モデルに予測させる**
```python
logits, loss = self(idx_cond)  # GPTモデルで予測
logits = logits[:, -1, :]      # 最後の位置の予測のみ取得
```

**GPTの予測の仕組み**:
- GPTは入力文字列の「各位置」で次の文字を予測します
- でも生成時は、「最後の位置の予測」だけが必要です

**例**: 「こんにち」という入力の場合
```
位置1: 「こ」→ 次は「ん」を予測
位置2: 「ん」→ 次は「に」を予測  
位置3: 「に」→ 次は「ち」を予測
位置4: 「ち」→ 次は「は」を予測 ← これだけが重要！
```

**Step 3: 生スコアを確率に変換**
```python
probs = F.softmax(logits, dim=-1)  # 確率に変換
```

**なぜ確率に変換？**
GPTの出力は「生スコア」（ロジット）です。これを0〜1の確率に変換します。

**例**: 次の文字の予測スコア
```
生スコア: 「は」:8.2, 「わ」:2.1, 「ば」:1.5, 「が」:0.8
↓ softmaxで変換
確率: 「は」:89%, 「わ」:7%, 「ば」:3%, 「が」:1%
```

**Step 4: 確率に基づいて1つの文字を選ぶ**
```python
idx_next = torch.multinomial(probs, num_samples=1)  # 確率的に選択
```

**なぜ最も確率の高い文字を選ばない？**
- **創造性**: 毎回同じ文章にならない
- **自然さ**: 人間も完全に予測可能ではない
- **多様性**: 面白い文章が生成される

**例**: 89%で「は」、7%で「わ」を選ぶ。たまに「わ」が選ばれるから面白い！

**Step 5: 新しい文字を文章に追加**
```python
idx = torch.cat((idx, idx_next), dim=1)  # 新しい文字を追加
```

**完了！次のループに進む**
```
元の文章: 「こんにち」
↓ 新しい文字「は」を追加  
新しい文章: 「こんにちは」
↓ これが次のループの入力になる
```

#### 全体の流れをまとめて理解

**1回のループで起こること**:
```
入力: 「こんにち」
Step 1: 最新8文字に制限 → 「こんにち」（8文字以下なのでそのまま）  
Step 2: GPTで予測 → 各文字の確率を計算
Step 3: 確率変換 → は:89%, わ:7%, ば:3%, が:1%
Step 4: 確率的選択 → 「は」を選択
Step 5: 文章更新 → 「こんにちは」

次のループ:
入力: 「こんにちは」  
→ 同じ処理を繰り返して次の文字を生成...
```

これを指定回数（例：500回）繰り返すことで、長い文章を生成できます！

#### より良いテキストを生成するには？

**1. 温度（Temperature）の調整** - 今回は未実装ですが重要
```python
logits = logits / temperature  # temperatureで調整
probs = F.softmax(logits, dim=-1)
```
- **温度が低い（例：0.3）**: 安全で予測しやすい文章
- **温度が高い（例：1.5）**: 創造的で予測しにくい文章

**2. より長いコンテキスト**
- `block_size`を大きくすると、より長い文脈を覚えられる
- より一貫性のある文章が生成される

**3. より大きなモデル**
- パラメータ数が多いほど、より賢い文章を生成
- 実際のChatGPTは数十億〜数千億パラメータ

**4. より多くの訓練データ**
- 多様な文章で学習するほど、多様な表現ができる

!!! tip "実際の応用では"
    ChatGPTなどでは、KV-Cacheという高速化技術や、より洗練されたサンプリング手法（Top-p sampling等）が使用されています。


## 8. 参考資料

**使用したデータソース**:
- [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

**参考文献**:
- Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017)
- Radford, A., et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9

**関連リンク**:
- [Andrej Karpathy's YouTube Channel](https://www.youtube.com/@AndrejKarpathy)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

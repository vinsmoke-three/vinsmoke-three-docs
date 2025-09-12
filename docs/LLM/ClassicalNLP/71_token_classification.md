---
title: "Transformersを使ったトークン分類：固有表現認識（NER）の実践ガイド"
description: "HuggingFaceのTransformersライブラリを使用してBERTモデルを固有表現認識タスクにファインチューニングする方法を詳しく解説。CoNLL-2003データセットを使った実践的なチュートリアル。"
date: "2025-09-12"
tags: ["自然言語処理", "機械学習", "BERT", "固有表現認識", "HuggingFace", "Transformers"]
---

# Transformersを使ったトークン分類：固有表現認識（NER）の実践ガイド

## 概要

この記事では、トークン分類（Token Classification）について詳しく解説します。特に固有表現認識（Named Entity Recognition, NER）に焦点を当て、HuggingFaceのTransformersライブラリを使ってBERTモデルをファインチューニングする方法を実践的に学習します。

### 学習目標

- トークン分類タスクの理解
- CoNLL-2003データセットの処理方法
- BERTモデルのファインチューニング手法
- カスタム訓練ループの実装

## 前提知識

- Pythonプログラミングの基礎知識
- 機械学習の基本概念
- Transformersアーキテクチャの基礎理解
- PyTorchの基本的な使用方法

## トークン分類とは

トークン分類は、「文中の各トークンにラベルを割り当てる」問題として定義される汎用的なタスクです。主な応用例は以下の通りです：

- **固有表現認識（Named Entity Recognition, NER）**: 文中の実体（人名、地名、組織名など）を特定します。これは各トークンに対してエンティティクラスまたは「エンティティなし」のラベルを割り当てることで実現されます。
- **品詞タグ付け（Part-of-Speech Tagging, POS）**: 文中の各単語を特定の品詞（名詞、動詞、形容詞など）に分類します。
- **チャンキング（Chunking）**: 同じエンティティに属するトークンのグループを識別します。このタスクは、チャンクの開始を示すラベル（通常`B-`）、チャンク内部を示すラベル（通常`I-`）、どのチャンクにも属さないトークンを示すラベル（通常`O`）を割り当てることで実現されます。

## データセットの準備

### CoNLL-2003データセット

まず、トークン分類タスクに適したデータセットを準備する必要があります。今回は、ロイター通信のニュース記事を含む[CoNLL-2003データセット](https://huggingface.co/datasets/conll2003)を使用します。

HuggingFace Datasetsライブラリの`load_dataset()`関数を使ってCoNLL-2003データセットを読み込みます：

```python
from datasets import load_dataset

# CoNLL-2003データセットを読み込み
raw_datasets = load_dataset("eriktks/conll2003", revision="convert/parquet")
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3453
    })
})
```

このデータセットには、先ほど説明した3つのタスク（NER、POS、チャンキング）のラベルが含まれています。

他のデータセットとの大きな違いは、入力テキストが文や文書として提示されるのではなく、単語のリスト（事前にトークン化された入力）として提供される点です。

訓練セットの最初の要素を確認してみましょう：

```python
# 最初のサンプルのトークンを表示
raw_datasets["train"][0]["tokens"]
```

**実行結果:**
```
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
```

固有表現認識を実行したいので、NERタグを確認します：

```python
# 対応するNERタグを表示
raw_datasets["train"][0]["ner_tags"]
```

**実行結果:**
```
[3, 0, 7, 0, 0, 0, 7, 0, 0]
```

これらは訓練用の整数ラベルですが、データを確認する際にはあまり理解しやすくありません。テキスト分類と同様に、データセットの`features`属性を調べることで、これらの整数とラベル名の対応関係を確認できます：

```python
# NERタグの特徴量情報を取得
ner_feature = raw_datasets["train"].features["ner_tags"]
print(ner_feature)
```

**実行結果:**
```
List(ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']))
```

この列の要素は`ClassLabel`のシーケンスです。ラベル名のリストは以下のようにアクセスできます：

```python
# ラベル名のリストを取得
label_names = ner_feature.feature.names
print(label_names)
```

**実行結果:**
```
['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

このラベリング方式は**BIOタグ**と呼ばれ、各ラベルの意味は以下の通りです：

- `O`：単語がどのエンティティにも対応しない
- `B-PER`/`I-PER`：単語が*人名*エンティティの開始/内部に対応
- `B-ORG`/`I-ORG`：単語が*組織*エンティティの開始/内部に対応
- `B-LOC`/`I-LOC`：単語が*地名*エンティティの開始/内部に対応
- `B-MISC`/`I-MISC`：単語が*その他*エンティティの開始/内部に対応

先ほど確認したラベルをデコードすると、以下のようになります：

```python
# ラベルとトークンを整列表示する関数
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
```

**実行結果:**
```
EU    rejects German call to boycott British lamb . 
B-ORG O       B-MISC O    O  O       B-MISC  O    O 
```

`B-`と`I-`ラベルが混在する例として、訓練セットのインデックス4の要素を見てみましょう：

```python
# インデックス4の例を表示
words = raw_datasets["train"][4]["tokens"]
labels = raw_datasets["train"][4]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
```

**実行結果:**
```
Germany 's representative to the European Union 's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer . 
B-LOC   O  O              O  O   B-ORG    I-ORG O  O          O         B-PER  I-PER     O    O  O         O         O      O   O         O    O         O     O    B-LOC   O     O   O          O      O   O       O 
```

### その他のラベル（POS、Chunking）

データセットには他にも品詞タグ付け（POS）とチャンキングのラベルが含まれています。

**品詞タグ（POS Tags）**：

```python
# POSタグのラベル名を確認
pos_label_names = raw_datasets["train"].features["pos_tags"].feature.names
print(pos_label_names)
```

**実行結果:**
```
['"', "''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
```

```python
# 最初のサンプルのPOSタグを表示
words = raw_datasets["train"][0]["tokens"]
pos_labels = raw_datasets["train"][0]["pos_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, pos_labels):
    full_label = pos_label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
```

**実行結果:**
```
EU  rejects German call to boycott British lamb . 
NNP VBZ     JJ     NN   TO VB      JJ      NN   . 
```

**チャンクタグ（Chunk Tags）**：

```python
# チャンクタグのラベル名を確認
chunk_label_names = raw_datasets["train"].features["chunk_tags"].feature.names
print(chunk_label_names)
```

**実行結果:**
```
['O', 'B-ADJP', 'I-ADJP', 'B-ADVP', 'I-ADVP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'B-LST', 'I-LST', 'B-NP', 'I-NP', 'B-PP', 'I-PP', 'B-PRT', 'I-PRT', 'B-SBAR', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-VP', 'I-VP']
```

```python
# 最初のサンプルのチャンクタグを表示
words = raw_datasets["train"][0]["tokens"]
chunk_labels = raw_datasets["train"][0]["chunk_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, chunk_labels):
    full_label = chunk_label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
```

**実行結果:**
```
EU   rejects German call to   boycott British lamb . 
B-NP B-VP    B-NP   I-NP B-VP I-VP    B-NP    I-NP O 
```

### データの前処理

通常のタスクと同様に、モデルが理解できるようにテキストをトークンIDに変換する必要があります。トークン分類タスクの特徴は、事前にトークン化された入力が与えられることです。幸い、tokenizerのAPIは特別なフラグを使って簡単に対処できます。

まず、`tokenizer`オブジェクトを作成します。BERTの事前訓練済みモデルを使用するので、関連するtokenizerをダウンロードしてキャッシュします：

```python
from transformers import AutoTokenizer

# 使用するモデルのチェックポイントを指定
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

`model_checkpoint`は[Hugging Face Hub](https://huggingface.co/models)の任意のモデルに変更したり、事前訓練済みモデルとtokenizerを保存したローカルフォルダのパスに変更したりできます。唯一の制約は、tokenizerがHuggingFace Tokenizersライブラリによってサポートされている必要があることです。

tokenizerオブジェクトが実際にHuggingFace Tokenizersライブラリによってサポートされている（Fast Tokenizerである）かは、`is_fast`属性で確認できます：

```python
# Fasttokenizerかどうかを確認
print(tokenizer.is_fast)
```

**実行結果:**
```
True
```

事前にトークン化された入力をトークン化するには、通常通りtokenizerを使用し、`is_split_into_words=True`を追加します：

```python
# 事前トークン化された入力をトークン化
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
print(inputs.tokens())
```

**実行結果:**
```
['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']
```

ご覧のとおり、tokenizerはモデルで使用する特殊トークン（最初の`[CLS]`と最後の`[SEP]`）を追加し、ほとんどの単語はそのまま残しました。

しかし、`lamb`という単語は`la`と`##mb`の2つのサブワードにトークン化されました。これにより、入力とラベルの間に不一致が生じます。ラベルのリストは9つの要素しかありませんが、入力は12のトークンになりました。

特殊トークンは位置が明確なため簡単に処理できますが、すべてのラベルを適切な単語と整列させる必要があります。

幸い、Fast Tokenizerを使用しているので、HuggingFace Tokenizersの強力な機能にアクセスでき、各トークンを対応する単語に簡単にマッピングできます：

```python
# 各トークンの対応する単語IDを確認
print(inputs.word_ids())
```

**実行結果:**
```
[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
```

少しの工夫で、ラベルリストをトークンに対応するように拡張できます。

適用するルールは以下の通りです：
1. **特殊トークン**：`-100`のラベルを与える（損失関数で無視されるため）
2. **単語の最初のトークン**：元の単語のラベルをそのまま使用
3. **単語内の後続トークン**：`B-`を`I-`に変更（エンティティの開始ではないため）

```python
def align_labels_with_tokens(labels, word_ids):
    """ラベルをトークンに整列させる関数"""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 新しい単語の開始
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊トークン
            new_labels.append(-100)
        else:
            # 前のトークンと同じ単語
            label = labels[word_id]
            # ラベルがB-XXXの場合、I-XXXに変更
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    
    return new_labels
```

この関数をテストしてみましょう：

```python
# ラベル整列のテスト
labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print("元のラベル:", labels)
print("整列後ラベル:", align_labels_with_tokens(labels, word_ids))
```

**実行結果:**
```
元のラベル: [3, 0, 7, 0, 0, 0, 7, 0, 0]
整列後ラベル: [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
```

ご覧のとおり、関数は最初と最後の2つの特殊トークンに`-100`を追加し、2つのトークンに分割された単語に新しい`0`を追加しました。

### 代替的なアプローチ

一部の研究者は、単語ごとに1つのラベルのみを割り当て、その単語内の他のサブトークンに`-100`を割り当てる手法を好みます。これは、多くのサブトークンに分割される長い単語が損失計算に過度に影響することを避けるためです。

この規則に従ってラベルを入力IDに整列させる関数の代替バージョンを以下に示します：

```python
def align_labels_with_tokens_v2(labels, word_ids):
    """単語ごとに1つのラベルのみを割り当てる代替バージョン"""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 新しい単語の開始
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊トークン
            new_labels.append(-100)
        else:
            # 前のトークンと同じ単語（サブトークン）
            new_labels.append(-100)
    
    return new_labels
```

```python
# 代替バージョンのテスト
labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print("元のラベル:", labels)
print("代替整列:", align_labels_with_tokens_v2(labels, word_ids))
```

**実行結果:**
```
元のラベル: [3, 0, 7, 0, 0, 0, 7, 0, 0]
代替整列: [-100, 3, 0, 7, 0, 0, 0, 7, 0, -100, 0, -100]
```

データセット全体を前処理するには、すべての入力をトークン化し、すべてのラベルに`align_labels_with_tokens()`関数を適用する必要があります。

Fast Tokenizerの速度を活用するため、一度に多数のテキストをトークン化する関数を作成し、`batched=True`オプションで`Dataset.map()`メソッドを使用します。以前の例との違いは、tokenizerへの入力がテキストのリスト（今回の場合は単語のリストのリスト）の場合、`word_ids()`関数に単語IDを取得したい例のインデックスを渡す必要があることです：

```python
def tokenize_and_align_labels(examples):
    """データセット全体をトークン化してラベルを整列させる関数"""
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
```

なお、この段階では入力をパディングしていません。後でデータコレーター（Data Collator）を使用してバッチを作成する際にパディングを行います。

データセットの他の分割に対して、この前処理をまとめて適用できます：

```python
# 全データセットに前処理を適用
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
print(tokenized_datasets)
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 14041
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 3453
    })
})
```

## Trainer APIを使用したモデルのファインチューニング

`Trainer`を使用した実際のコードは基本的に以前と同じです。主な変更点は、データをバッチにまとめる方法と評価指標計算関数のみです。

### データの照合

第3章の`DataCollatorWithPadding`は使用できません。それは入力（入力ID、アテンションマスク、トークンタイプID）のみをパディングするためです。

トークン分類では、ラベルも入力と同じ方法でパディングし、同じサイズを維持する必要があります。対応する予測が損失計算で無視されるように、パディング値として`-100`を使用します。

これらの処理はすべて[`DataCollatorForTokenClassification`](https://huggingface.co/transformers/main_classes/data_collator.html#datacollatorfortokenclassification)によって自動的に行われます。`DataCollatorWithPadding`と同様に、入力の前処理に使用した`tokenizer`を渡します：

```python
from transformers import DataCollatorForTokenClassification

# トークン分類用のデータコレーターを作成
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

これをいくつかのサンプルでテストするため、トークン化済み訓練セットからの例のリストで呼び出します：

```python
# データコレーターをテスト
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print(batch["labels"])
```

**実行結果:**
```
tensor([[-100,    3,    0,    7,    0,    0,    0,    7,    0,    0,    0, -100],
        [-100,    1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
```

これをデータセット内の最初と2番目の要素のラベルと比較してみましょう：

```python
# 元のラベルと比較
for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])
```

**実行結果:**
```
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
[-100, 1, 2, -100]
```

ご覧のとおり、短い方の2番目のラベルセットは`-100`を使用して最初のラベルセットの長さにパディングされています。

### 評価指標

`Trainer`がエポックごとに評価指標を計算するために、予測とラベルの配列を受け取り、評価指標名と値を含む辞書を返す`compute_metrics()`関数を定義する必要があります。

トークン分類予測を評価するために広く使用されている評価フレームワークは[**seqeval**](https://github.com/chakki-works/seqeval)です。

```python
# seqeval評価指標を読み込み
import evaluate

metric = evaluate.load("seqeval")
```

この評価指標は標準的な精度計算とは異なる動作をします。整数ではなく文字列としてラベルのリストを受け取るため、評価指標に渡す前に予測とラベルを完全にデコードする必要があります。

動作を確認してみましょう。まず、最初の訓練例のラベルを取得します：

```python
# 最初の訓練例のラベルを文字列に変換
labels = raw_datasets["train"][0]["ner_tags"]
label_names = raw_datasets["train"].features["ner_tags"].feature.names
labels = [label_names[i] for i in labels]
print(labels)
```

**実行結果:**
```
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

次に、インデックス2の値を変更して偽の予測を作成できます：

```python
# 偽の予測を作成してメトリクスをテスト
predictions = labels.copy()
predictions[2] = "O"
result = metric.compute(predictions=[predictions], references=[labels])
print(result)
```

**実行結果:**
```
{'MISC': {'precision': 1.0,
  'recall': 0.5,
  'f1': 0.6666666666666666,
  'number': 2},
 'ORG': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
 'overall_precision': 1.0,
 'overall_recall': 0.6666666666666666,
 'overall_f1': 0.8,
 'overall_accuracy': 0.8888888888888888}
```

豊富な評価情報が返されます！各エンティティタイプ別の精度（precision）、再現率（recall）、F1スコア、および全体的なスコアを取得できます。

実際のモデル予測を使用してスコアを計算する準備が整いました。評価時の予測結果を処理するための`compute_metrics`関数を定義します：

```python
import numpy as np

def compute_metrics(eval_preds):
    """評価指標を計算する関数"""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # 無視されるインデックス（特殊トークン）を削除してラベルに変換
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # seqevalメトリクスを計算
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```

### モデルの定義

トークン分類問題に取り組んでいるため、`AutoModelForTokenClassification`クラスを使用します。

このモデルを定義する際の重要なポイントは、ラベル数の情報を渡すことです。最も簡単な方法は`num_labels`引数でその数を指定することですが、推論ウィジェットが適切に機能するようにするには、ラベルの対応関係を明示的に設定する方が良いでしょう。

これは、IDからラベルへのマッピングとその逆方向のマッピングを含む2つの辞書`id2label`と`label2id`によって設定します：

```python
# ラベルマッピング辞書を作成
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
print(f"id2label: {id2label}")
print(f"label2id: {label2id}")
```

**実行結果:**
```
id2label: {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
label2id: {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
```

これらの辞書を`AutoModelForTokenClassification.from_pretrained()`メソッドに渡すことで、モデルの設定に反映されます：

```python
from transformers import AutoModelForTokenClassification

# トークン分類モデルを作成
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id
)

# ラベル数を確認
print(f"ラベル数: {model.config.num_labels}")
```

**実行結果:**
```
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
ラベル数: 9
```

### モデルのファインチューニング

これでモデルを訓練する準備が整いました！まず、訓練パラメータを定義します：

```python
from transformers import TrainingArguments

# 訓練引数を設定
args = TrainingArguments(
    "bert-finetuned-ner",           # 出力ディレクトリ
    eval_strategy="epoch",          # エポックごとに評価
    save_strategy="epoch",          # エポックごとに保存
    learning_rate=2e-5,            # 学習率
    num_train_epochs=3,            # 訓練エポック数
    weight_decay=0.01,             # 重み減衰
    push_to_hub=False,             # Hubにプッシュしない
)
```

最後に、すべてのコンポーネントを`Trainer`に渡して訓練を開始します：

```python
from transformers import Trainer

# Trainerを作成
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)

# 訓練を実行（コメントアウト）
# trainer.train()
```

## カスタム訓練ループ

より細かい制御が必要な場合に備えて、完全な訓練ループの実装を見てみましょう。

### 訓練の準備

まず、データセットから`DataLoader`を構築する必要があります。`data_collator`を`collate_fn`として再利用し、訓練セットをシャッフルしますが、検証セットはシャッフルしません：

```python
from torch.utils.data import DataLoader

# データローダーを作成
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=8
)
```

次に、モデルを再インスタンス化して、以前のファインチューニング結果から継続するのではなく、BERT事前訓練済みモデルから新たに開始するようにします：

```python
# モデルを再作成（新しいインスタンス）
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id
)
```

次にオプティマイザーが必要です。重み減衰の適用方法が改良された`Adam`である`AdamW`を使用します：

```python
from torch.optim import AdamW

# オプティマイザーを作成
optimizer = AdamW(model.parameters(), lr=2e-5)
```

これらのオブジェクトがすべて揃ったら、`accelerator.prepare()`メソッドに送信できます：

```python
from accelerate import Accelerator

# Acceleratorを初期化してオブジェクトを準備
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

`train_dataloader`を`accelerator.prepare()`に渡したので、その長さを使用して訓練ステップ数を計算できます。

**重要**: データローダーを準備した後に長さを計算する必要があります。`prepare()`メソッドがデータローダーの長さを変更する可能性があるためです。学習率を線形にゼロまで減衰させるスケジューラーを使用します：

```python
from transformers import get_scheduler

# 学習率スケジューラーを設定
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
```

### 訓練ループ

完全な訓練ループを書く準備が整いました。評価部分を簡潔にするため、予測とラベルを受け取り、`metric`オブジェクトが期待する文字列のリストに変換する`postprocess()`関数を定義します：

```python
def postprocess(predictions, labels):
    """後処理関数：予測とラベルを文字列リストに変換"""
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # 無視されるインデックスを削除してラベルに変換
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
```

次に訓練ループを実装します。訓練の進行状況を追跡するプログレスバーを定義した後、ループは3つの主要な部分から構成されます：

1. **訓練フェーズ**: `train_dataloader`の通常の反復処理、モデルを通した順伝播、逆伝播とオプティマイザーステップ
2. **評価フェーズ**: モデルのバッチ出力を取得した後、複数のプロセス間で異なる形状に入力とラベルがパディングされている可能性があるため、`gather()`メソッドを呼び出す前に`accelerator.pad_across_processes()`を使用して予測とラベルを同じ形状に揃える必要があります
3. **保存フェーズ**: モデルとtokenizerを保存します

完全な訓練ループのコードは以下の通りです：

```python
from tqdm import tqdm
import torch

# 訓練ループを実行
progress_bar = tqdm(range(num_training_steps))
output_dir = "bert-finetuned-ner-accelerate"

for epoch in range(num_train_epochs):
    # 訓練フェーズ
    model.train()
    for batch in train_dataloader:
        # 順伝播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 逆伝播
        accelerator.backward(loss)

        # パラメータ更新
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # 評価フェーズ
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # 予測とラベルを収集のためにパディング
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)
        
    # 評価指標を計算・表示
    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"] 
            for key in ["precision", "recall", "f1", "accuracy"]
        }
    )

    # モデルとtokenizerを保存
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
```

**実行結果:**
```
5%|▌         | 281/5268 [00:47<07:22, 11.28it/s]
```

HuggingFace Accelerateでモデルを保存するのが初めての場合は、関連する3行のコードについて説明します：

```python
# 全プロセスの同期を待つ
accelerator.wait_for_everyone()
# ベースモデルを取得
unwrapped_model = accelerator.unwrap_model(model)
# モデルを保存
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
```

**1行目**: 全プロセスの同期を待機します。保存前にすべてのプロセスがこの段階に到達するのを待ちます。

**2行目**: 元のモデル（`unwrapped_model`）を取得します。`accelerator.prepare()`メソッドは分散訓練用にモデルを変更するため、`save_pretrained()`メソッドがアクセスできなくなります。`accelerator.unwrap_model()`メソッドでこの変更を元に戻します。

**3行目**: `save_pretrained()`を呼び出しますが、標準の`torch.save()`の代わりに`accelerator.save()`を使用するように指定します。

これが完了すると、`Trainer`で訓練したモデルと非常に似た結果を得られるはずです。このコードを使用して訓練したモデルの例は[**huggingface-course/bert-finetuned-ner-accelerate**](https://huggingface.co/huggingface-course/bert-finetuned-ner-accelerate)で確認できます。

訓練ループをカスタマイズしたい場合は、上記のコードを編集して独自の実装を行うことができます！

## ファインチューニングしたモデルの使用

ローカル環境で`pipeline`を使用するには、適切なモデル識別子を指定するだけです：

```python
from transformers import pipeline

# ファインチューニングしたモデルでパイプラインを作成
# (実際のチェックポイントに置き換えてください)
model_checkpoint = "bert-finetuned-ner-accelerate"
token_classifier = pipeline(
    "token-classification", 
    model=model_checkpoint, 
    aggregation_strategy="simple"
)

# モデルをテスト
result = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)
```

**実行結果:**
```
Device set to use mps:0
[{'entity_group': 'PER', 'score': 0.9961755, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
{'entity_group': 'ORG', 'score': 0.98193836, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
{'entity_group': 'LOC', 'score': 0.99809605, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

素晴らしい結果です！モデルは文中の3つのエンティティを正確に識別しました：

- "Sylvain"を人名（PER）として
- "Hugging Face"を組織名（ORG）として  
- "Brooklyn"を地名（LOC）として

## 参考資料

- [CoNLL-2003データセット](https://huggingface.co/datasets/conll2003)
- [HuggingFace Transformersドキュメント](https://huggingface.co/transformers/)
- [seqeval評価ライブラリ](https://github.com/chakki-works/seqeval)
- [HuggingFace Accelerateライブラリ](https://huggingface.co/docs/accelerate/)
- [ファインチューニング済みモデル](https://huggingface.co/huggingface-course/bert-finetuned-ner)
    
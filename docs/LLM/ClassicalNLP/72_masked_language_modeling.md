---
title: "マスク言語モデルのファインチューニング - ドメイン適応による性能向上"
description: "DistilBERTを映画レビューデータでファインチューニングし、ドメイン適応によって言語モデルの性能を向上させる手法を詳しく解説します。"
date: "2025-09-20"
tags: ["自然言語処理", "ファインチューニング", "BERT", "DistilBERT", "マスク言語モデリング"]
---

# マスク言語モデルのファインチューニング

## 概要

この記事では、Transformerモデルのマスク言語モデリング（Masked Language Modeling）を使ったファインチューニングについて学習します。特に、事前学習済みのDistilBERTモデルを映画レビューデータでファインチューニングし、ドメイン適応（Domain Adaptation）を実現する手法を実践的に解説します。

!!! info "参考資料"
    本ドキュメントは [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter7/3) を参考に、日本語で学習内容をまとめた個人的な学習ノートです。詳細な内容や最新情報については、原文も併せてご参照ください。

## 前提知識

- Pythonプログラミングの基礎知識
- 機械学習とディープラーニングの基本概念
- Transformerアーキテクチャの理解
- PyTorchまたはTensorFlowの基本的な使い方

## ドメイン適応の必要性

多くのNLPアプリケーションでは、Hugging Face Hubの事前学習済みモデルを直接ファインチューニングするだけで良い結果が得られます。しかし、以下のような場合には、まず言語モデルをドメイン固有のデータでファインチューニングしてから、タスク固有のヘッドを学習させる必要があります。

- **法律文書**: 法的専門用語が多く含まれている
- **科学論文**: 専門的な学術用語が頻出している
- **医療記録**: 医療固有の略語や用語が使用されている

このような場合、BERTのような汎用Transformerモデルは、ドメイン固有の単語を稀少トークン（rare tokens）として扱い、期待する性能が得られない可能性があります。

ドメイン内データで言語モデルをファインチューニングすることで、多くの下流タスクの性能を向上させることができ、通常この処理は一度だけ実行すれば済みます。

この手法は**ドメイン適応**と呼ばれ、2018年にULMFiTによって普及しました。ULMFiTは、NLPにおける転移学習を実用化した最初のニューラルアーキテクチャ（LSTM基盤）の一つでした。

## 事前学習済みモデルの選択

マスク言語モデリング用の適切な事前学習済みモデルを選択しましょう。Hugging Face Hubでは、「Fill-Mask」フィルターを適用することで候補を見つけることができます。

BERTやRoBERTaファミリのモデルが最もダウンロードされていますが、今回は**DistilBERT**を使用します。このモデルは、下流タスクでの性能をほとんど損なうことなく、はるかに高速に学習できます。

### DistilBERTの特徴

DistilBERTは**知識蒸留（Knowledge Distillation）**という特別な技術を使って訓練されました。

- **教師モデル**: BERT（大規模モデル）
- **生徒モデル**: DistilBERT（パラメータ数が大幅に削減されたモデル）
- **結果**: 性能をほぼ維持しながら、約2倍の高速化を実現している

まず、`AutoModelForMaskedLM`クラスを使ってDistilBERTをダウンロードしましょう。

```python
from transformers import AutoModelForMaskedLM

# DistilBERTの事前学習済みモデルを読み込み
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

モデルのパラメータ数を確認してみましょう。

```python
# モデルのパラメータ数を計算（百万単位）
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT パラメータ数: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT パラメータ数: 110M'")
```

**実行結果:**
```
'>>> DistilBERT パラメータ数: 67M'
'>>> BERT パラメータ数: 110M'
```

約6700万のパラメータを持つDistilBERTは、BERTベースモデルの約2分の1のサイズで、学習時間も約2倍高速化されます。

### マスク予測のテスト

DistilBERTがどのようなトークンを予測するか確認してみましょう。

```python
# テスト用のサンプルテキスト
text = "This is a great [MASK]."
```

人間が考えると、`[MASK]`の位置には「day」、「ride」、「painting」など様々な可能性があります。事前学習済みモデルの予測は、学習に使用されたコーパスに依存します。DistilBERTは、BERTと同様に英語Wikipediaおよび BookCorpusデータセットで事前学習されているため、これらのドメインを反映した予測が期待されます。

マスクを予測するためには、DistilBERTのトークナイザーも必要です。

```python
from transformers import AutoTokenizer

# 対応するトークナイザーを読み込み
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

トークナイザーとモデルを使って、上位5つの候補を予測してみましょう。

```python
import torch

# テキストをトークン化してモデルに入力
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits

# [MASK]トークンの位置を特定し、そのロジットを抽出
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]

# 最も高いロジットを持つ上位5つの[MASK]候補を取得
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# 結果を表示
for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
```

**実行結果:**
```
'>>> This is a great deal.'
'>>> This is a great success.'
'>>> This is a great adventure.'
'>>> This is a great idea.'
'>>> This is a great feat.'
```

出力から分かるように、モデルの予測は日常的な用語を反映しており、これは英語Wikipediaという基盤を考えると驚くことではありません。次に、このドメインをより特化したもの（極めて偏った映画レビュー）に変更する方法を見ていきましょう。

## データセットの準備

ドメイン適応を実演するために、有名な**Large Movie Review Dataset**（IMDbデータセット）を使用します。これは映画レビューのコーパスで、感情分析モデルのベンチマークによく使用されます。

このコーパスでDistilBERTをファインチューニングすることで、言語モデルが事前学習で使用したWikipediaの事実的データから、映画レビューのより主観的な要素へと語彙を適応させることが期待されます。

Hugging Face Hubからデータを取得しましょう。

```python
from datasets import load_dataset

# IMDbデータセットを読み込み
imdb_dataset = load_dataset("imdb")
imdb_dataset
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
```

`train`と`test`分割にはそれぞれ25,000件のレビューが含まれ、`unsupervised`という名前のラベルなし分割には50,000件のレビューが含まれています。

データの内容を確認してみましょう。

```python
# ランダムに3つのサンプルを選択
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> レビュー: {row['text']}'")
    print(f"'>>> ラベル: {row['label']}'")
```

**実行結果:**
```

'>>> レビュー: There is no relation at all between Fortier and Profiler but the fact that both are police series about violent crimes. Profiler looks crispy, Fortier looks classic. Profiler plots are quite simple. Fortier's plot are far more complicated... Fortier looks more like Prime Suspect, if we have to spot similarities... The main character is weak and weirdo, but have "clairvoyance". People like to compare, to judge, to evaluate. How about just enjoying? Funny thing too, people writing Fortier looks American but, on the other hand, arguing they prefer American series (!!!). Maybe it's the language, or the spirit, but I think this series is more English than American. By the way, the actors are really good and funny. The acting is not superficial at all...'
'>>> ラベル: 1'

'>>> レビュー: This movie is a great. The plot is very true to the book which is a classic written by Mark Twain. The movie starts of with a scene where Hank sings a song with a bunch of kids called "when you stub your toe on the moon" It reminds me of Sinatra's song High Hopes, it is fun and inspirational. The Music is great throughout and my favorite song is sung by the King, Hank (bing Crosby) and Sir "Saggy" Sagamore. OVerall a great family movie or even a great Date movie. This is a movie you can watch over and over again. The princess played by Rhonda Fleming is gorgeous. I love this movie!! If you liked Danny Kaye in the Court Jester then you will definitely like this movie.'
'>>> ラベル: 1'

'>>> レビュー: George P. Cosmatos' "Rambo: First Blood Part II" is pure wish-fulfillment. The United States clearly didn't win the war in Vietnam. They caused damage to this country beyond the imaginable and this movie continues the fairy story of the oh-so innocent soldiers. The only bad guys were the leaders of the nation, who made this war happen. The character of Rambo is perfect to notice this. He is extremely patriotic, bemoans that US-Americans didn't appreciate and celebrate the achievements of the single soldier, but has nothing but distrust for leading officers and politicians. Like every film that defends the war (e.g. "We Were Soldiers") also this one avoids the need to give a comprehensible reason for the engagement in South Asia. And for that matter also the reason for every single US-American soldier that was there. Instead, Rambo gets to take revenge for the wounds of a whole nation. It would have been better to work on how to deal with the memories, rather than suppressing them. "Do we get to win this time?" Yes, you do.'
'>>> ラベル: 0'
```

これらは確実に映画レビューです。言語モデリングにはラベルは必要ありませんが、`0`がネガティブレビュー、`1`がポジティブレビューを表していることが分かります。

## データの前処理

自動回帰言語モデリングとマスク言語モデリングの両方において、一般的な前処理ステップは、すべての例を連結してから、全体のコーパスを等しいサイズのチャンクに分割することです。これは、単純に個別の例をトークン化する通常のアプローチとは大きく異なります。

なぜすべてを連結するのでしょうか。個別の例が長すぎると切り捨てられ、言語モデリングタスクに有用な情報が失われる可能性があるためです。

### トークン化の実装

まず、コーパスを通常通りトークン化しますが、トークナイザーで`truncation=True`オプションは設定**しません**。後で全単語マスキングに必要なword IDも取得します：

```python
def tokenize_function(examples):
    # テキストをトークン化（切り捨てなし）
    result = tokenizer(examples["text"])
    # 高速トークナイザーの場合、word IDsを取得
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# 高速マルチスレッド処理を有効化
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids'],
        num_rows: 50000
    })
})
```

DistilBERTはBERT系モデルなので、エンコードされたテキストは`input_ids`、`attention_mask`、および追加した`word_ids`から構成されています。

### チャンクサイズの決定

映画レビューをトークン化したので、次にそれらをすべてグループ化し、結果をチャンクに分割します。チャンクのサイズはどの程度にすべきでしょうか。これは最終的に利用可能なGPUメモリ量によって決まりますが、良い出発点はモデルの最大コンテキストサイズを確認することです。

```python
# モデルの最大コンテキストサイズを確認
tokenizer.model_max_length
```

**実行結果:**
```
512
```

この値は、チェックポイントに関連付けられた*tokenizer_config.json*ファイルから取得されています。この場合、BERTと同様にコンテキストサイズが512トークンであることが分かります。

### 連結処理のデモンストレーション

連結がどのように機能するかを示すために、トークン化された訓練セットからいくつかのレビューを取得し、レビューごとのトークン数を出力してみましょう。

```python
# スライシングにより各特徴のリストのリストを生成
tokenized_samples = tokenized_datasets["train"][:3]

for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> レビュー {idx} の長さ: {len(sample)}'")
```

**実行結果:**
```
'>>> レビュー 0 の長さ: 363'
'>>> レビュー 1 の長さ: 304'
'>>> レビュー 2 の長さ: 133'
```

これらすべての例を、次のような単純な辞書内包表記で連結できます。

```python
# すべての例を連結
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> 連結されたレビューの長さ: {total_length}'")
```

**実行結果:**
```
'>>> 連結されたレビューの長さ: 800'
```

素晴らしいです。総長さが合っています。次に、連結されたレビューを`chunk_size`で指定されたサイズのチャンクに分割しましょう。

```python
# チャンクサイズを設定
chunk_size = 128

# 連結された例の特徴ごとにチャンクを作成
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

# 各チャンクの長さを確認
for chunk in chunks["input_ids"]:
    print(f"'>>> チャンク長: {len(chunk)}'")
```

**実行結果:**
```
'>>> チャンク長: 128'
'>>> チャンク長: 128'
'>>> チャンク長: 128'
'>>> チャンク長: 128'
'>>> チャンク長: 128'
'>>> チャンク長: 128'
'>>> チャンク長: 32'
```

この例で見られるように、最後のチャンクは通常、最大チャンクサイズよりも小さくなります。これに対処する主な戦略は2つあります。

1. **削除**: 最後のチャンクが`chunk_size`より小さい場合は削除する
2. **パディング**: 最後のチャンクの長さが`chunk_size`と等しくなるまでパディングする

ここでは最初のアプローチを採用します。上記のロジックをすべて一つの関数にまとめて、トークン化されたデータセットに適用できるようにしましょう。

```python
def group_texts(examples):
    # すべてのテキストを連結
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 連結されたテキストの長さを計算
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 最後のチャンクがchunk_sizeより小さい場合は削除
    total_length = (total_length // chunk_size) * chunk_size
    # max_lenのチャンクに分割
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # 新しいlabelsカラムを作成
    result["labels"] = result["input_ids"].copy()
    return result
```

`group_texts()`の最後のステップで、`input_ids`のコピーである新しい`labels`カラムを作成していることに注目してください。これは、マスク言語モデリングでは、入力バッチ内のランダムにマスクされたトークンを予測することが目的であり、`labels`カラムを作成することで、言語モデルが学習するための正解を提供するためです。

信頼できる`Dataset.map()`関数を使用して、`group_texts()`をトークン化されたデータセットに適用しましょう：

```python
# グループ化されたテキストに変換
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
lm_datasets
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids', 'labels'],
        num_rows: 61291
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids', 'labels'],
        num_rows: 59904
    })
    unsupervised: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids', 'labels'],
        num_rows: 122957
    })
})
```

テキストをグループ化してからチャンク化することで、元の`train`および`test`分割の25,000よりもはるかに多くの例が生成されました。これは、元のコーパスの複数の例にまたがる**連続するトークン**を含む例ができたためです。

チャンクの一つで特別な`[SEP]`と`[CLS]`トークンを探すことで、これを明示的に確認できます：

```python
# チャンクの内容を確認
tokenizer.decode(lm_datasets["train"][1]["input_ids"])
```

**実行結果:**
```
"as the vietnam war and race issues in the united states. in between asking politicians and ordinary denizens of stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men. < br / > < br / > what kills me about i am curious - yellow is that 40 years ago, this was considered pornographic. really, the sex and nudity scenes are few and far between, even then it ' s not shot like some cheaply made porno. while my countrymen mind find it shocking, in reality sex and nudity are a major staple in swedish cinema. even ingmar bergman,"
```

この例では、高校映画に関するレビューとホームレスに関するレビューという、2つの重複する映画レビューが確認できます。

マスク言語モデリング用のラベルがどのようになっているかも確認してみましょう：

```python
# ラベルの内容を確認
tokenizer.decode(lm_datasets["train"][1]["labels"])
```

**実行結果:**
```
"as the vietnam war and race issues in the united states. in between asking politicians and ordinary denizens of stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men. < br / > < br / > what kills me about i am curious - yellow is that 40 years ago, this was considered pornographic. really, the sex and nudity scenes are few and far between, even then it ' s not shot like some cheaply made porno. while my countrymen mind find it shocking, in reality sex and nudity are a major staple in swedish cinema. even ingmar bergman,"
```

上記の`group_texts()`関数から期待されるように、これはデコードされた`input_ids`と同じに見えます。しかし、どうやってモデルが何かを学習できるのでしょうか？重要なステップが欠けています：入力のランダムな位置に`[MASK]`トークンを挿入することです！ファインチューニング中に特別なデータコレーターを使用してこれをその場で行う方法を見ていきましょう。

## Trainer APIを使ったDistilBERTのファインチューニング

マスク言語モデルのファインチューニングは、第3章で行った系列分類モデルのファインチューニングとほぼ同じです。唯一の違いは、各テキストバッチでいくつかのトークンをランダムにマスクできる特別なデータコレーターが必要なことです。

幸い、Hugging Face Transformersには、まさにこのタスク用の専用`DataCollatorForLanguageModeling`が用意されています。トークナイザーと、マスクするトークンの割合を指定する`mlm_probability`引数を渡すだけです。BERTで使用され、文献でも一般的な15%を選択します：

```python
from transformers import DataCollatorForLanguageModeling

# マスク言語モデリング用のデータコレーターを作成
# 15%の確率でトークンをマスク
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

### ランダムマスキングの動作確認

ランダムマスキングがどのように機能するかを確認するために、いくつかの例をデータコレーターに供給してみましょう。これは`dict`のリストを期待し、各`dict`は連続するテキストの単一のチャンクを表します。このデータコレーターは`"word_ids"`キーを期待しないため、削除してからバッチをコレーターに供給します：

```python
# サンプルデータを準備
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

# データコレーターを適用してマスキング結果を確認
for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
```

**実行結果:**
```

'>>> [CLS] i rented i am curious - yellow from my [MASK] store because of all the controversy that surrounded it when it was first released in 1967. i also heard that at first it was [MASK] byÂ² 1915 s. customs if [MASK] ever tried to enter this country [MASK] therefore [MASK] heiress fan of films considered " controversial " i [MASK] had to see this for myself. < br collier > [MASK] br / > the plot [MASK] centered around a [MASK] swedish drama student named lena who wants to learn everything she can about life. in [MASK] she wants to focus her attention 407 to [MASK] some sort of documentary on what the [MASK] totalede thought about certain political issues such'

'>>> as the vietnam war and race issues in the united states. in between asking politicians and [MASK] den [MASK]ns of stockholm about [MASK] opinions on politics, she namesake sex with her drama teacher, classmates, and married men. < br / > < br / [MASK] [MASK] kills me [MASK] i am [MASK] - yellow is that 40 years ago, this was considered pornographic. really, the [MASK] and nudity scenes are few and far between, even then it [MASK] s not [MASK] like some cheaply made porno. while my countrymen [MASK] find [MASK] shocking, in reality [MASK] and nudity are a major staple in swedish [MASK]. even ing caucasian bergman,'
```

素晴らしい！うまく機能しています。`[MASK]`トークンがテキストの様々な場所にランダムに挿入されているのが確認できます。これらが、訓練中にモデルが予測しなければならないトークンになります。データコレーターの美しさは、すべてのバッチで`[MASK]`の挿入をランダム化することです！

### 全単語マスキング（Whole Word Masking）

マスク言語モデリングで使用できる一つの技術は、個別のトークンではなく、単語全体を一緒にマスクすることです。このアプローチは**全単語マスキング**と呼ばれます。

全単語マスキングを使用したい場合は、データコレーターを自分で構築する必要があります。データコレーターは、サンプルのリストを受け取ってバッチに変換する関数です：

```python
import collections
import numpy as np
from transformers import default_data_collator

# 全単語マスキングの確率
wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # 単語と対応するトークンインデックス間のマップを作成
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # 単語をランダムにマスク
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)
```

次に、前と同じサンプルで試してみましょう：

```python
# 全単語マスキングの動作確認
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
```

**実行結果:**
```

'>>> [CLS] i [MASK] i am curious - yellow from [MASK] video store because [MASK] all the controversy that surrounded it [MASK] [MASK] [MASK] first [MASK] in 1967 [MASK] [MASK] [MASK] heard that at first [MASK] [MASK] seized by u [MASK] s. [MASK] if it ever [MASK] [MASK] enter this country, therefore being a fan [MASK] films considered " controversial [MASK] i really [MASK] to see this for [MASK]. [MASK] br / > < [MASK] / > [MASK] plot is [MASK] around a young swedish drama student named lena [MASK] wants to [MASK] everything [MASK] can [MASK] life. in particular she wants to focus her attentions to making some sort of documentary on what the average [MASK] [MASK] thought about certain political issues [MASK]'

'>>> as [MASK] vietnam [MASK] and race issues in the united states. in between asking [MASK] and ordinary denizens of [MASK] about [MASK] opinions on politics, she has sex with her drama teacher, classmates, and [MASK] men. < [MASK] / > [MASK] br / > [MASK] kills me [MASK] [MASK] am curious - [MASK] [MASK] that 40 years ago, this was considered pornographic. really, the sex and nudity scenes are few and [MASK] between, even then it [MASK] [MASK] not shot like some cheaply made [MASK] [MASK]. while my countrymen [MASK] find it shocking, [MASK] [MASK] sex and nudity are a major staple in [MASK] cinema. even ingmar [MASK],'
```

### データセットのダウンサンプリング

2つのデータコレーターができたので、残りのファインチューニングステップは標準的です。心配しないでください、それでもかなり良い言語モデルが得られます！

Hugging Face Datasetsでデータセットをダウンサンプリングする簡単な方法は、`Dataset.train_test_split()`関数を使用することです：

```python
# データセットサイズを設定
train_size = 10_000
test_size = int(0.1 * train_size)

# ダウンサンプリングを実行
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
downsampled_dataset
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids', 'labels'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask', 'word_ids', 'labels'],
        num_rows: 1000
    })
})
```

これにより、訓練セットのサイズが10,000例、検証セットが1,000例に設定された新しい`train`と`test`分割が自動的に作成されました。

### 訓練引数の設定

```python
from transformers import TrainingArguments

batch_size = 64
# エポックごとに訓練損失を表示
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    logging_steps=logging_steps,
)
```

ここでは、エポックごとに訓練損失を追跡するために`logging_steps`を含め、いくつかのデフォルトオプションを調整しました。

### Trainerの初期化と実行

必要な材料がすべて揃ったので、`Trainer`をインスタンス化できます。ここでは標準の`data_collator`を使用していますが、演習として全単語マスキングコレーターを試して結果を比較することもできます：

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
)
```

**実行結果:**
```
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
```

## 言語モデルの困惑度（Perplexity）評価

テキスト分類や質問応答のような他のタスクとは異なり、言語モデリングではラベル付きコーパスが与えられません。では、良い言語モデルとは何を決定するのでしょうか？

スマートフォンの自動修正機能のように、良い言語モデルは文法的に正しい文に高い確率を割り当て、意味のない文に低い確率を割り当てるものです。これをより良く理解するために、オンラインで「自動修正の失敗」のセット全体を見つけることができ、人の電話のモデルがかなり面白い（そしてしばしば不適切な）補完を生成した例が示されています。

テストセットが主に文法的に正しい文で構成されていると仮定すると、言語モデルの品質を測定する一つの方法は、テストセット内のすべての文の次の単語に割り当てる確率を計算することです。

高い確率は、モデルが未見の例に対して「驚いていない」または「困惑していない」ことを示し、言語の基本的な文法パターンを学習していることを示唆します。

困惑度には様々な数学的定義がありますが、ここで使用するものは**クロスエントロピー損失の指数**として定義されます。したがって、`Trainer.evaluate()`関数を使用してテストセットでクロスエントロピー損失を計算し、その結果の指数を取ることで、事前学習済みモデルの困惑度を計算できます：

```python
import math

# 事前学習済みモデルの困惑度を評価
# eval_results = trainer.evaluate()
# print(f">>> 困惑度: {math.exp(eval_results['eval_loss']):.2f}")
# >>> 困惑度: 21.75
```

困惑度スコアが低いほど、言語モデルが優れていることを意味します。ここでは、開始モデルがやや大きな値を持っていることが分かります。ファインチューニングによってこれを下げることができるか見てみましょう。

まず、訓練ループを実行します：

```python
# 訓練の実行
# trainer.train()
```

その後、前と同様にテストセットで結果の困惑度を計算します：

```python
# ファインチューニング後の困惑度を評価
# eval_results = trainer.evaluate()
# print(f">>> 困惑度: {math.exp(eval_results['eval_loss']):.2f}")
# >>> 困惑度: 11.32
```

素晴らしいです。これは困惑度の大幅な削減で、モデルが映画レビューのドメインについて何かを学習したことを示しています。

## Hugging Face Accelerateを使ったDistilBERTのファインチューニング

`Trainer`で見たように、マスク言語モデルのファインチューニングは、第3章のテキスト分類例と非常に似ています。実際、唯一の微妙な点は特別なデータコレーターの使用であり、このセクションの前半で既にそれをカバーしました！

しかし、`DataCollatorForLanguageModeling`は各評価でもランダムマスキングを適用するため、各訓練実行で困惑度スコアにいくらかの変動が見られました。

このランダム性の源を排除する一つの方法は、テストセット全体に**一度**マスキングを適用し、評価中にHugging Face Transformersのデフォルトデータコレーターを使用することです。

### 固定マスキング関数の実装

これがどのように機能するかを確認するために、`DataCollatorForLanguageModeling`との最初の遭遇に似た、バッチにマスキングを適用する簡単な関数を実装しましょう：

```python
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # データセットの各カラムに新しい「masked」カラムを作成
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
```

次に、この関数をテストセットに適用し、マスクされていないカラムを削除してマスクされたものに置き換えることができます。全単語マスキングを使用する場合は、上記の`data_collator`を適切なものに置き換え、ここで最初の行を削除する必要があります：

```python
# データセットからword_idsカラムを削除
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])

# テストセットに固定マスキングを適用
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)

# カラム名を変更してマスクされたバージョンを標準名に
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
```

### データローダーの設定

通常通りデータローダーを設定できますが、評価セットには Hugging Face Transformersの`default_data_collator`を使用します：

```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64

# 訓練用データローダー（ランダムマスキング付き）
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

# 評価用データローダー（固定マスキング）
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
```

### モデルとオプティマイザーの準備

ここから、Hugging Face Accelerateを使った標準的なステップに従います。まず、事前学習済みモデルの新しいバージョンを読み込みます：

```python
# 新しいモデルインスタンスを読み込み
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

次にオプティマイザーを指定します。標準的な`AdamW`を使用します：

```python
from torch.optim import AdamW

# オプティマイザーを設定
optimizer = AdamW(model.parameters(), lr=5e-5)
```

これらのオブジェクトを使って、`Accelerator`オブジェクトで訓練のためのすべてを準備できます：

```python
from accelerate import Accelerator

# Acceleratorを初期化し、すべてのコンポーネントを準備
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

### 学習率スケジューラーの設定

モデル、オプティマイザー、データローダーが設定されたので、学習率スケジューラーを以下のように指定できます：

```python
from transformers import get_scheduler

# 訓練パラメータの設定
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# 線形学習率スケジューラーを設定
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

### 完全な訓練・評価ループ

```python
from tqdm.auto import tqdm
import torch
import math

# プログレスバーと出力ディレクトリを設定
progress_bar = tqdm(range(num_training_steps))
output_dir = "distilbert-base-uncased-finetuned-imdb-accelerate"

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
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    # 困惑度の計算
    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> エポック {epoch}: 困惑度: {perplexity}")

    # モデルの保存とアップロード
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
```

**実行結果:**
```
>>> エポック 0: 困惑度: 11.560117477655108
>>> エポック 1: 困惑度: 11.106818026989123
>>> エポック 2: 困惑度: 10.901680692261271
```

## ファインチューニング済みモデルの使用

ファインチューニング済みモデルは、Hubのウィジェットを使用するか、Hugging Face Transformersの`pipeline`を使用してローカルで操作できます。後者を使用して、`fill-mask`パイプラインでモデルをダウンロードしましょう：

```python
from transformers import pipeline

# ファインチューニング済みモデルでパイプラインを作成
mask_filler = pipeline(
    "fill-mask", model="distilbert-base-uncased-finetuned-imdb-accelerate"
)
```

**実行結果:**
```
Device set to use mps:0
```

「This is a great [MASK]」というサンプルテキストをパイプラインに供給し、上位5つの予測を確認してみましょう：

```python
# マスク予測を実行
preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
```

**実行結果:**
```
>>> this is a great film.
>>> this is a great movie.
>>> this is a great idea.
>>> this is a great one.
>>> this is a great show.
```

素晴らしいです。ファインチューニング前は「deal」、「success」、「adventure」などの一般的な用語が予測されていましたが、ファインチューニング後は「film」、「movie」、「show」といった映画レビューのドメインに特化した用語が予測されるようになりました。これは、ドメイン適応が成功したことを明確に示しています。

## 参考資料

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [DistilBERT: a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [Universal Language Model Fine-tuning for Text Classification (ULMFiT)](https://arxiv.org/abs/1801.06146)
- [IMDb Dataset](https://huggingface.co/datasets/imdb)
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)
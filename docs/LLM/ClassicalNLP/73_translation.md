---
title: "Hugging Face Transformersを使った機械翻訳モデルのファインチューニング実践ガイド"
description: "Helsinki-NLPのOpus-100データセットを使用して英日翻訳モデルをファインチューニングする方法を、TrainerAPIとカスタム学習ループの両方で詳しく解説します。"
date: "2024-01-15"
tags: ["機械翻訳", "Transformers", "ファインチューニング", "NLP", "Python", "英日翻訳"]
---

# 機械翻訳モデルのファインチューニング

## 概要

この記事では、**sequence-to-sequence（系列対系列）** タスクの一つである機械翻訳について学習します。機械翻訳は、ある言語の文章を別の言語に変換する問題で、要約問題と同様のアプローチが適用できます。

本記事で学習する内容は、以下のような他の系列対系列問題にも応用可能です：

- **スタイル変換**: あるスタイルで書かれたテキストを別のスタイルに変換するモデル（例：フォーマルな文章をカジュアルに、シェイクスピア英語を現代英語に）
- **生成的質問応答**: 文脈を与えられた質問に対する回答を生成するモデル

!!! info "参考資料"
    本ドキュメントは [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter7/4) を参考に、日本語で学習内容をまとめた個人的な学習ノートです。詳細な内容や最新情報については、原文も併せてご参照ください。

## 前提知識

この記事を理解するために必要な知識：

- Python の基本的な文法
- Transformers ライブラリの基本的な使い方
- 機械学習の基礎概念（学習、評価、ファインチューニング）
- PyTorch の基本的な操作

## データの準備

機械翻訳モデルをファインチューニングまたはゼロから学習するには、タスクに適したデータセットが必要です。この記事では [opus-100 dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100) を使用しますが、翻訳したい言語ペアの文章があれば、コードを簡単に適応させることができます。

### opus-100 データセット

いつものように、`load_dataset()` 関数を使用してデータセットをダウンロードします。

```python
from datasets import load_dataset

# 英日翻訳用データセットを読み込み
split_datasets = load_dataset("Helsinki-NLP/opus-100", "en-ja")
split_datasets
```

**実行結果:**
```
DatasetDict({
    test: Dataset({
        features: ['translation'],
        num_rows: 2000
    })
    train: Dataset({
        features: ['translation'],
        num_rows: 1000000
    })
    validation: Dataset({
        features: ['translation'],
        num_rows: 2000
    })
})
```

データセットの要素を一つ見てみましょう。

```python
# データセットのサンプルを確認
split_datasets["train"][1]["translation"]
```

**実行結果:**
```
{'en': "I'm being held in a basement. I've been abducted with two other girls.",
 'ja': 'いま地下に居ます 他の2人と一緒に誘拐されたんです！'}
```

リクエストした言語ペアの2つの文章を含む辞書が得られます。

事前訓練済みモデルを使って翻訳を試してみましょう。このモデルは、より大きなフランス語と英語の文章コーパスで事前訓練されており、簡単な翻訳を提供します。

```python
from transformers import pipeline

# 英日翻訳用の事前訓練済みモデルを読み込み
model_checkpoint = "Helsinki-NLP/opus-mt-en-jap"
translator = pipeline("translation", model=model_checkpoint)
translator("I'm being held in a basement. I've been abducted with two other girls.")
```

**実行結果:**
```
Device set to use mps:0

[{'translation_text': 'あたし は 不平 を 負う 者 と な り , ほか の ふたり の 女 と 婚約 し た こと が ある が ,'}]
```

### データの前処理

お馴染みの手順ですが、すべてのテキストをトークンIDのセットに変換して、モデルが理解できるようにする必要があります。このタスクでは、入力とターゲットの両方をトークン化する必要があります。最初のタスクは、`tokenizer` オブジェクトを作成することです。前述のとおり、Marian英日事前訓練済みモデルを使用します。別の言語ペアでこのコードを試す場合は、モデルチェックポイントを適応させてください。[Helsinki-NLP](https://huggingface.co/Helsinki-NLP) 組織では、複数言語で1000以上のモデルを提供しています。

```python
from transformers import AutoTokenizer

# 英日翻訳モデルのTokenizerを初期化
model_checkpoint = "Helsinki-NLP/opus-mt-en-jap"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors = "pt")
```

データの準備は非常に簡単です。覚えておくべきことが一つあります：tokenizerが出力言語（ここでは日本語）でターゲットを処理することを確実にする必要があります。これは、tokenizerの `__call__` メソッドの `text_targets` 引数にターゲットを渡すことで行えます。

この仕組みを確認するために、訓練セットから各言語のサンプルを1つずつ処理してみましょう。

```python
# 英語と日本語の文章を取得
en_sentence = split_datasets["train"][1]["translation"]["en"]
ja_sentence = split_datasets["train"][1]["translation"]["ja"]

# 入力文章とターゲット文章をトークン化
inputs = tokenizer(en_sentence, text_target=ja_sentence)
inputs
```

**実行結果:**
```
{'input_ids': [31, 62, 2315, 616, 2513, 20, 33, 3761, 6359, 4, 31, 62, 9099, 428, 1, 44823, 203, 48, 253, 422, 24303, 4, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': [2249, 18925, 115, 14755, 652, 6, 2338, 1, 570, 2666, 4550, 1, 1382, 1845, 572, 315, 0]}
```

ご覧のとおり、出力には英語の文章に関連付けられた入力IDが含まれ、日本語に関連付けられたIDは `labels` フィールドに格納されます。ラベルをトークン化していることを示すのを忘れると、入力tokenizerによってトークン化されることになり、Marianモデルの場合はうまくいきません：

```python
# 間違った方法：日本語文章を英語tokenizerで処理
wrong_targets = tokenizer(ja_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))
```

**実行結果:**
```
['▁', '<unk>', '▁', 'ä»–ã®', '<unk>', '<unk>', '!', '</s>']
['▁いま', '地下', 'に', '居', 'ます', '▁', 'ä»–ã®', '<unk>', '人', 'と一緒に', '誘', '<unk>', 'された', 'んで', 'す', '!', '</s>']
```

ご覧のとおり、英語tokenizerを使ってフランス語（日本語）の文章を前処理すると、tokenizerが日本語の単語を知らないため（英語にも現れる単語を除いて、例えば「discussion」など）、はるかに多くのトークンが生成されます。

`inputs` は通常のキー（入力ID、attention mask など）を持つ辞書なので、最後のステップは、データセットに適用する前処理関数を定義することです：

```python
# 最大長を設定
max_length = 128

def preprocess_function(examples):
    # 英語と日本語の文章を分離
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["ja"] for ex in examples["translation"]]
    
    # 入力とターゲットを同時にトークン化
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs
```

入力と出力に同じ最大長を設定していることに注意してください。扱っているテキストは非常に短いようなので、128を使用します。

**💡 注意点:**

ターゲットのattention maskには注意を払いません。代わりに、パディングトークンに対応するラベルは、損失計算で無視されるように `-100` に設定する必要があります。これは、動的パディングを適用しているため、後でデータコレクターによって行われますが、ここでパディングを使用する場合は、パディングトークンに対応するすべてのラベルを `-100` に設定するように前処理関数を適応させる必要があります。

データセットのすべての分割で前処理を一度に適用できます。

```python
# データセット全体に前処理を適用
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names
)
```

データが前処理されたので、事前訓練済みモデルをファインチューニングする準備が整いました。

## Trainer API を使用したモデルのファインチューニング

`Trainer` を使用した実際のコードは以前と同じで、小さな変更が一つだけあります：ここでは [`Seq2SeqTrainer`](https://huggingface.co/transformers/main_classes/trainer.html#seq2seqtrainer) を使用します。これは `Trainer` のサブクラスで、`generate()` メソッドを使用して入力から出力を予測することで、評価を適切に処理できます。メトリック計算について話すときに、これについてより詳細に説明します。

まず第一に、ファインチューニングする実際のモデルが必要です。通常の `AutoModel` API を使用します：

```python
from transformers import AutoModelForSeq2SeqLM
import torch

# Sequence-to-Sequence 学習用モデルを読み込み
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# デバイスの設定（MPS, GPU, CPUの順で利用可能なものを使用）
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)
```

### データコレクション

動的バッチ処理のパディングを処理するために、データコレクターが必要です。`DataCollatorWithPadding` だけは使用できません。これは入力（入力ID、attention mask、トークンタイプID）のみをパッドするからです。ラベルも、ラベル内で遭遇した最大長にパッドされる必要があります。そして、前述のとおり、ラベルをパッドするために使用されるパディング値は、損失計算でパッドされた値が無視されるように、tokenizerのパディングトークンではなく `-100` である必要があります。

これはすべて [`DataCollatorForSeq2Seq`](https://huggingface.co/transformers/main_classes/data_collator.html#datacollatorforseq2seq) によって行われます。`DataCollatorWithPadding` と同様に、入力を前処理するために使用される `tokenizer` を取りますが、`model` も取ります。このデータコレクターは、特別なトークンを先頭に持つラベルのシフトバージョンであるデコーダー入力IDの準備も担当するからです。このシフトは異なるアーキテクチャで若干異なって行われるため、`DataCollatorForSeq2Seq` は `model` オブジェクトを知る必要があります：

```python
from transformers import DataCollatorForSeq2Seq

# Seq2Seq用のデータコレクターを初期化
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

いくつかのサンプルでテストするために、トークン化された訓練セットのサンプルリストで呼び出します。

```python
# データコレクターのテスト実行
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
```

ラベルがバッチの最大長にパッドされ、`-100` が使用されていることを確認できます。

```python
# パディングされたラベルを確認
batch["labels"]
```

**実行結果:**
```
tensor([[ 2249, 18925,   115, 14755,   652,     6,  2338,     1,   570,  2666,
          4550,     1,  1382,  1845,   572,   315,     0],
        [    6,  1695,   621,   261,   315,     0,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100]])
```

また、デコーダー入力IDを見ることもできます。これらがラベルのシフトされたバージョンであることがわかります。

```python
# デコーダー入力IDを確認
batch["decoder_input_ids"]
```

**実行結果:**
```
tensor([[46275,  2249, 18925,   115, 14755,   652,     6,  2338,     1,   570,
          2666,  4550,     1,  1382,  1845,   572,   315],
        [46275,     6,  1695,   621,   261,   315,     0, 46275, 46275, 46275,
         46275, 46275, 46275, 46275, 46275, 46275, 46275]])
```

データセットの最初と2番目の要素のラベルは次のとおりです。

```python
# 元のラベルを確認
for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])
```

**実行結果:**
```
[2249, 18925, 115, 14755, 652, 6, 2338, 1, 570, 2666, 4550, 1, 1382, 1845, 572, 315, 0]
[6, 1695, 621, 261, 315, 0]
```

この `data_collator` を `Seq2SeqTrainer` に渡します。次に、メトリックを見てみましょう。

### 評価メトリック

`Seq2SeqTrainer` がスーパークラス `Trainer` に追加する機能は、評価や予測中に `generate()` メソッドを使用する能力です。訓練中、モデルは予測しようとしているトークンの後のトークンを使用しないようにするattention maskを持つ `decoder_input_ids` を使用して、訓練を高速化します。推論中はラベルがないため、これらを使用することはできないので、同じセットアップでモデルを評価することは良いアイデアです。

デコーダーは、一度に一つずつトークンを予測することで推論を実行します。これは、Hugging Face Transformersの `generate()` メソッドによって舞台裏で実装されています。`Seq2SeqTrainer` では、`predict_with_generate=True` を設定すると、評価にそのメソッドを使用できます。

翻訳に使用される伝統的なメトリックは、Kishore Papineni らによる [2002年の論文](https://aclanthology.org/P02-1040.pdf) で紹介された [BLEU スコア](https://en.wikipedia.org/wiki/BLEU) です。BLEUスコアは、翻訳がラベルにどれだけ近いかを評価します。モデルが生成した出力の理解可能性や文法的正確性を測定するのではなく、統計的ルールを使用して、生成された出力内のすべての単語がターゲット内にも現れることを確認します。さらに、ターゲットでも繰り返されていない場合に同じ単語の繰り返しを罰するルール（モデルが `"the the the the the"` のような文章を出力することを避けるため）と、ターゲット内のものより短い出力文章を罰するルール（モデルが `"the"` のような文章を出力することを避けるため）があります。

BLEUの弱点の一つは、テキストが既にトークン化されていることを期待することで、異なるtokenizerを使用するモデル間でスコアを比較することが困難になります。そのため、今日の翻訳モデルのベンチマークに最もよく使用されるメトリックは [SacreBLEU](https://github.com/mjpost/sacrebleu) です。これはトークン化ステップを標準化することで、この弱点（および他の弱点）を対処します。このメトリックを使用するには、まずSacreBLEUライブラリをインストールする必要があります：

```python
# !pip install sacrebleu
```

その後、`evaluate.load()` を介して読み込むことができます。

```python
import evaluate

# SacreBLEUメトリックを読み込み
metric = evaluate.load("sacrebleu")
```

このメトリックは、入力とターゲットとしてテキストを受け取ります。複数の受け入れ可能なターゲットを受け入れるように設計されています。同じ文章に対して複数の受け入れ可能な翻訳があることが多いからです。使用しているデータセットは1つしか提供していませんが、NLPでラベルとして複数の文章を提供するデータセットを見つけることは珍しくありません。したがって、予測は文章のリストである必要がありますが、参照は文章のリストのリストである必要があります。

```python
# BLEUスコアの計算例
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
```

**実行結果:**
```
{'score': 46.750469682990186,
 'counts': [11, 6, 4, 3],
 'totals': [12, 11, 10, 9],
 'precisions': [91.66666666666667,
  54.54545454545455,
  40.0,
  33.333333333333336],
 'bp': 0.9200444146293233,
 'sys_len': 12,
 'ref_len': 13}
```

これは46.75のBLEUスコアを取得します。これは非常に良いスコアです。参考として、["Attention Is All You Need" 論文](https://arxiv.org/pdf/1706.03762.pdf) の元のTransformerモデルは、英語とフランス語間の同様の翻訳タスクで41.8のBLEUスコアを達成しました！（`counts` や `bp` などの個別のメトリックの詳細については、[SacreBLEU リポジトリ](https://github.com/mjpost/sacrebleu/blob/078c440168c6adc89ba75fe6d63f0d922d42bcfe/sacrebleu/metrics/bleu.py#L74) を参照してください。）

一方、翻訳モデルからよく出てくる2つの悪いタイプの予測（大量の繰り返しや短すぎるもの）を試すと、かなり悪いBLEUスコアが得られます：

```python
# 悪い予測例1：繰り返しが多い場合
predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
```

**実行結果:**
```
{'score': 1.683602693167689,
 'counts': [1, 0, 0, 0],
 'totals': [4, 3, 2, 1],
 'precisions': [25.0, 16.666666666666668, 12.5, 12.5],
 'bp': 0.10539922456186433,
 'sys_len': 4,
 'ref_len': 13}
```

```python
# 悪い予測例2：短すぎる場合
predictions = ["This plugin"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
```

**実行結果:**
```
{'score': 0.0,
 'counts': [2, 1, 0, 0],
 'totals': [2, 1, 0, 0],
 'precisions': [100.0, 100.0, 0.0, 0.0],
 'bp': 0.004086771438464067,
 'sys_len': 2,
 'ref_len': 13}
```

スコアは0から100の範囲で、高いほど良いです。

モデルの出力からメトリックが使用できるテキストに変換するには、`tokenizer.batch_decode()` メソッドを使用します。ラベル内のすべての `-100` をクリーンアップする必要があります（tokenizerはパディングトークンに対して自動的に同じことを行います）。

```python
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # モデルが予測logit以上を返す場合
    if isinstance(preds, tuple):
        preds = preds[0]

    # 予測をデコードしてテキストに変換
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # ラベルの-100をパディングトークンIDに置き換えてデコード
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 簡単な後処理：空白の削除
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # BLEUスコアを計算
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}
```

### モデルのファインチューニング

`Seq2SeqTrainingArguments` を定義できます。`Trainer` と同様に、いくつかのフィールドが追加された `TrainingArguments` のサブクラスを使用します。

```python
from transformers import Seq2SeqTrainingArguments

# 訓練パラメータの設定
args = Seq2SeqTrainingArguments(
    f"marian-finetuned-opus-100-en-to-ja",  # 出力ディレクトリ
    eval_strategy="no",  # 評価戦略
    save_strategy="epoch",  # 保存戦略
    learning_rate=2e-5,  # 学習率
    per_device_train_batch_size=16,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=32,   # 評価時のバッチサイズ
    weight_decay=0.01,  # 重み減衰
    save_total_limit=3,  # 保存するチェックポイントの最大数
    num_train_epochs=3,  # エポック数
    predict_with_generate=True,  # 評価時にgenerate()メソッドを使用
    push_to_hub=False,  # Hubにアップロードしない
)
```

前のセクションで見たものと比較して、通常のハイパーパラメータ（学習率、エポック数、バッチサイズ、重み減衰など）に加えて、いくつかの変更があります：

- 評価には時間がかかるため、定期的な評価は設定しません。訓練前と訓練後に一度だけモデルを評価します
- 上記で説明したように、`predict_with_generate=True` を設定します

最後に、すべてを `Seq2SeqTrainer` に渡します。

```python
from transformers import Seq2SeqTrainer

# Seq2SeqTrainerの初期化
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
```

```python
# 実際の訓練実行（コメントアウト）
# trainer.train()
```

## カスタム訓練ループ

### 訓練の準備

これまでに何度か見てきたので、コードを素早く進めます。まず、データセットを `"torch"` 形式に設定してPyTorchテンソルを取得した後、データセットから `DataLoader` を構築します。

```python
from torch.utils.data import DataLoader

# データセットをPyTorch形式に設定
tokenized_datasets.set_format("torch")

# 訓練用データローダーの作成
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,  # データをシャッフル
    collate_fn=data_collator,
    batch_size=8,
)

# 評価用データローダーの作成
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    collate_fn=data_collator, 
    batch_size=8
)
```

次に、以前のファインチューニングから続行するのではなく、事前訓練済みモデルから再開することを確認するために、モデルを再インスタンス化します。

```python
# モデルを再度読み込み（前回の訓練状態をリセット）
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

次に、オプティマイザーが必要です。

```python
from torch.optim import AdamW

# Adam optimizerの初期化
optimizer = AdamW(model.parameters(), lr=2e-5)
```

これらのオブジェクトがすべて揃ったら、`accelerator.prepare()` メソッドに送信できます。ColabノートブックでTPUで訓練したい場合は、このコードをすべて訓練関数に移動する必要があり、`Accelerator` をインスタンス化するセルは実行すべきではないことを忘れないでください。

```python
from accelerate import Accelerator

# 分散訓練のためのAcceleratorを初期化
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

`train_dataloader` を `accelerator.prepare()` に送信したので、その長さを使用して訓練ステップ数を計算できます。データローダーを準備した後は、その方法が `DataLoader` の長さを変更するため、常にこれを行う必要があることを忘れないでください。学習率から0への古典的な線形スケジュールを使用します：

```python
from transformers import get_scheduler

# 訓練ステップ数の計算
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# 学習率スケジューラーの設定
lr_scheduler = get_scheduler(
    "linear",  # 線形減衰
    optimizer=optimizer,
    num_warmup_steps=0,  # ウォームアップなし
    num_training_steps=num_training_steps,
)
```

```python
# 出力ディレクトリの設定
output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"
```

### 訓練ループ

完全な訓練ループを書く準備が整いました。評価部分を簡素化するために、予測とラベルを受け取り、`metric` オブジェクトが期待する文字列のリストに変換する `postprocess()` 関数を定義します。

```python
def postprocess(predictions, labels):
    # CPUにデータを移動してnumpy配列に変換
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    # 予測をテキストにデコード
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # ラベルの-100をパディングトークンIDに置き換えてデコード
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 簡単な後処理：空白の削除とリスト形式への変換
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels
```

注意すべき最初の点は、予測を計算するために `generate()` メソッドを使用することですが、これはHugging Face Accelerateが `prepare()` メソッドで作成したラップされたモデルではなく、ベースモデルのメソッドです。そのため、最初にモデルをアンラップしてから、このメソッドを呼び出します。

2番目の点は、**トークン分類**と同様に、2つのプロセスが入力とラベルを異なる形状にパディングしている可能性があるため、`gather()` メソッドを呼び出す前に、`accelerator.pad_across_processes()` を使用して予測とラベルを同じ形状にします。これを行わないと、評価がエラーアウトするか、永続的にハングします。

```python
from tqdm.auto import tqdm
import torch

# 進捗バーの設定
progress_bar = tqdm(range(num_training_steps))

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
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            # generate()メソッドを使用してテキスト生成
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # 予測とラベルを各プロセス間で同じ形状にパディング
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        # 全プロセスから予測とラベルを収集
        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        # テキストに変換してメトリックに追加
        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    # BLEUスコアを計算して表示
    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    # モデルの保存とアップロード
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
```

**実行結果:**
```
epoch 0, BLEU score: 33.26
epoch 1, BLEU score: 38.19
epoch 2, BLEU score: 41.56
```

## 参考資料

**使用したデータセットとモデル:**
- [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100): 大規模多言語翻訳データセット
- [Helsinki-NLP/opus-mt-en-jap](https://huggingface.co/Helsinki-NLP/opus-mt-en-jap): 英日翻訳用事前訓練済みモデル

**評価メトリック:**
- [SacreBLEU](https://github.com/mjpost/sacrebleu): 標準化されたBLEU評価ツール
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf): BLEUスコアの原論文

**関連技術資料:**
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf): Transformerアーキテクチャの原論文
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/): 公式ドキュメント
- [Helsinki-NLP Organization](https://huggingface.co/Helsinki-NLP): 1000以上の翻訳モデルを提供

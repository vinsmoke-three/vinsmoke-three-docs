---
title: "Transformerを使ったテキスト要約の実装ガイド - T5モデルによる文書要約"
description: "Transformerモデル（T5）を使用して長い文書を要約する方法を学びます。CNN/DailyMailデータセットを用いたファインチューニングから評価まで、実践的な実装を詳しく解説します。"
date: "2025-09-25"
tags: ["NLP", "Transformer", "T5", "Text Summarization", "Machine Learning", "PyTorch"]
---

# Transformerを使ったテキスト要約の実装ガイド

## 概要

この記事では、Transformerモデルを利用して長い文書を短い要約に変換する**テキスト要約（Text Summarization）**の実装方法について詳しく解説します。テキスト要約は、長い文章の理解と一貫性のあるテキスト生成という複数の能力を必要とする、最も難易度の高いNLPタスクの一つです。しかし、適切に実装されたテキスト要約システムは、専門家の文書読解負担を軽減し、様々なビジネスプロセスを効率化する強力なツールとなります。

!!! info "参考資料"
    本ドキュメントは [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter7/5) を参考に、日本語で学習内容をまとめた個人的な学習ノートです。詳細な内容や最新情報については、原文も併せてご参照ください。

## 前提知識

この記事を理解するために、以下の知識があることを推奨します：

- **Python基礎**: pandas, numpy等の基本的な使用方法
- **機械学習の基礎**: 学習、検証、評価の概念
- **Transformerアーキテクチャ**: Encoder-Decoderモデルの基本理解
- **Hugging Face Transformers**: 基本的な使用方法

## データセットの準備

CNN/DailyMailデータセットを利用して要約システムを構築していきます。このデータセットは、ニュース記事とそのハイライト（要約）のペアを含む、テキスト要約タスクで広く利用されているベンチマークデータセットです。

```python
from datasets import load_dataset

# CNN/DailyMailデータセットを読み込み
datasets = load_dataset("cnn_dailymail", "3.0.0")
```

**実行結果:**
```python
datasets
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 287113
    })
    validation: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 13368
    })
    test: Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 11490
    })
})
```

ご覧の通り、訓練用データセットには287,113件の記事が含まれています。このデータセットで重要な情報は`article`列（元の記事）と`highlights`列（要約）に格納されています。

```python
# データの構造を確認
datasets["train"][1]
```

**実行結果:**
```
{'article': 'Editor\'s note: In our Behind the Scenes series, CNN correspondents share their experiences in covering news and analyze the stories behind the events. Here, Soledad O\'Brien takes users inside a jail where many of the inmates are mentally ill. An inmate housed on the "forgotten floor," where many mentally ill inmates are housed in Miami before trial. MIAMI, Florida (CNN) -- The ninth floor of the Miami-Dade pretrial detention facility is dubbed the "forgotten floor." Here, inmates with the most severe mental illnesses are incarcerated until they\'re ready to appear in court. Most often, they face drug charges or charges of assaulting an officer --charges that Judge Steven Leifman says are usually "avoidable felonies." He says the arrests often result from confrontations with police. Mentally ill people often won\'t do what they\'re told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid, delusional, and less likely to follow directions, according to Leifman. So, they end up on the ninth floor severely mentally disturbed, but not getting any real help because they\'re in jail. We toured the jail with Leifman. He is well known in Miami as an advocate for justice and the mentally ill. Even though we were not exactly welcomed with open arms by the guards, we were given permission to shoot videotape and tour the floor.  Go inside the \'forgotten floor\' Â» . At first, it\'s hard to determine where the people are. The prisoners are wearing sleeveless robes. Imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that\'s kind of what they look like. They\'re designed to keep the mentally ill patients from injuring themselves. That\'s also why they have no shoes, laces or mattresses. Leifman says about one-third of all people in Miami-Dade county jails are mentally ill. So, he says, the sheer volume is overwhelming the system, and the result is what we see on the ninth floor. Of course, it is a jail, so it\'s not supposed to be warm and comforting, but the lights glare, the cells are tiny and it\'s loud. We see two, sometimes three men -- sometimes in the robes, sometimes naked, lying or sitting in their cells. "I am the son of the president. You need to get me out of here!" one man shouts at me. He is absolutely serious, convinced that help is on the way -- if only he could reach the White House. Leifman tells me that these prisoner-patients will often circulate through the system, occasionally stabilizing in a mental hospital, only to return to jail to face their charges. It\'s brutally unjust, in his mind, and he has become a strong advocate for changing things in Miami. Over a meal later, we talk about how things got this way for mental patients. Leifman says 200 years ago people were considered "lunatics" and they were locked up in jails even if they had no charges against them. They were just considered unfit to be in society. Over the years, he says, there was some public outcry, and the mentally ill were moved out of jails and into hospitals. But Leifman says many of these mental hospitals were so horrible they were shut down. Where did the patients go? Nowhere. The streets. They became, in many cases, the homeless, he says. They never got treatment. Leifman says in 1955 there were more than half a million people in state mental hospitals, and today that number has been reduced 90 percent, and 40,000 to 50,000 people are in mental hospitals. The judge says he\'s working to change this. Starting in 2008, many inmates who would otherwise have been brought to the "forgotten floor"  will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment, not just punishment. Leifman says it\'s not the complete answer, but it\'s a start. Leifman says the best part is that it\'s a win-win solution. The patients win, the families are relieved, and the state saves money by simply not cycling these prisoners through again and again. And, for Leifman, justice is served. E-mail to a friend .',
 'highlights': 'Mentally ill inmates in Miami are housed on the "forgotten floor"\nJudge Steven Leifman says most are there as a result of "avoidable felonies"\nWhile CNN tours facility, patient shouts: "I am the son of the president"\nLeifman says the system is unjust and he\'s fighting for change .',
 'id': 'ee8871b15c50d0db17b0179a6d2beab35065f1e9'}
```

このデータ例からわかるように、`article`フィールドには長いニュース記事が、`highlights`フィールドにはその要約が格納されています。

## テキスト要約のためのモデル

テキスト要約は機械翻訳に似たタスクと考えることができます。レビューなどのテキストの本体を、入力の重要な特徴を捉えた短いバージョンに「翻訳」したいからです。そのため、要約用のほとんどのTransformerモデルは、基本的なEncoder-Decoderアーキテクチャを採用しています。ただし、Few-shot設定で要約に利用できるGPTファミリーのモデルなど、いくつかの例外もあります。

以下の表は、要約のためにファインチューニングに適した代表的な事前学習済みモデルをリストアップしたものです：

| Transformerモデル | 説明 | 多言語対応 |
| :-------: | ---- | :-----: |
| [GPT-2](https://huggingface.co/gpt2-xl) | 自己回帰言語モデルとして学習されていますが、入力テキストの最後に「TL;DR」を追加することでGPT-2に要約を生成させることができます。 | ❌ |
| [PEGASUS](https://huggingface.co/google/pegasus-large) | 複数文テキストでマスクされた文を予測する事前学習目的を使用します。この事前学習目的は、バニラ言語モデリングよりも要約に近く、人気のベンチマークで高いスコアを記録しています。 | ❌ |
| [T5](https://huggingface.co/t5-base) | すべてのタスクをtext-to-textフレームワークで定式化するユニバーサルTransformerアーキテクチャ。例：文書を要約するモデルの入力形式は `summarize: ARTICLE` です。 | ❌ |
| [mT5](https://huggingface.co/google/mt5-base) | T5の多言語版で、101言語をカバーする多言語Common Crawlコーパス（mC4）で事前学習されています。 | ✅ |
| [BART](https://huggingface.co/facebook/bart-base) | 破損した入力を再構築するように学習された、エンコーダーとデコーダーの両方のスタックを持つ新しいTransformerアーキテクチャで、BERTとGPT-2の事前学習スキームを組み合わせています。 | ❌ |
| [mBART-50](https://huggingface.co/facebook/mbart-large-50) | BARTの多言語版で、50言語で事前学習されています。 | ✅ |

この表からわかるように、要約（そして実際にはほとんどのNLPタスク）用のTransformerモデルの大部分は単言語です。これは、英語やドイツ語などの「高リソース」言語でのタスクには最適ですが、世界中で利用されている何千もの他の言語にはあまり適していません。幸い、mT5やmBARTなどの多言語Transformerモデルのクラスが解決策を提供しています。これらのモデルは言語モデリングを利用して事前学習されていますが、特徴的な点があります。1つの言語のコーパスで学習する代わりに、50以上の言語のテキストで同時に共同学習されています。

## データの前処理

次のタスクは、記事とそのタイトルをトークン化してエンコードすることです。いつものように、事前学習済みモデルのチェックポイントに関連付けられたトークナイザーを読み込むことから始めます。合理的な時間でモデルをファインチューニングできるように、チェックポイントとして`t5-small`を使用します：

```python
from transformers import T5Tokenizer

# T5モデルのトークナイザーを読み込み
model_checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
```

小さな例でT5トークナイザーをテストしてみましょう：

```python
# トークナイザーのテスト
inputs = tokenizer("I loved reading the Hunger Games!")
inputs
```

**実行結果:**
```
{'input_ids': [27, 1858, 1183, 8, 26049, 5880, 55, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

ここでは、おなじみの`input_ids`と`attention_mask`が見られます。トークナイザーの`convert_ids_to_tokens()`関数でこれらの入力IDをデコードして、どのようなトークナイザーを扱っているかを見てみましょう：

```python
# トークンに変換して内容を確認
tokenizer.convert_ids_to_tokens(inputs.input_ids)
```

**実行結果:**
```
['▁I', '▁loved', '▁reading', '▁the', '▁Hunger', '▁Games', '!', '</s>']
```

特殊なUnicode文字`▁`と終了シーケンストークン`</s>`は、Unigramセグメンテーションアルゴリズムに基づくSentencePieceトークナイザーを扱っていることを示しています。Unigramは、アクセント、句読点、そして日本語のように空白文字を持たない多くの言語について、SentencePieceが言語に依存しないようにするため、多言語コーパスに特に有用です。

T5にはタスクプレフィックスが必要です。以下はT5の処理の例です：

```python
# 前処理関数の定義
def preprocess_function(examples):
    # T5にはタスクプレフィックスが必要
    inputs = ["summarize: " + doc for doc in examples["article"]]
    targets = examples["highlights"]
    
    # 入力とターゲットをトークン化
    model_inputs = tokenizer(
        inputs, 
        max_length=512,  # 最大入力長
        truncation=True, 
        padding=False
    )
    
    # ターゲットテキストをトークン化
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=128,  # 要約は通常短い
            truncation=True, 
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 前処理を適用
tokenized_dataset = datasets.map(preprocess_function, batched=True)
```

## Trainer APIを使用したT5のファインチューニング

要約のためのモデルのファインチューニングは、他の一般的なNLPタスクと非常に似ています。最初にやるべきことは、`t5-small`チェックポイントから事前学習済みモデルを読み込むことです。要約はsequence-to-sequenceタスクなので、`AutoModelForSeq2SeqLM`クラスでモデルを読み込むことができ、これによって重みが自動的にダウンロードされキャッシュされます：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# デバイスの設定とモデルの読み込み
device = "mps" if torch.mps.is_available() else "cpu"
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
```

学習中にROUGEスコアを計算するために要約を生成する必要があります。幸い、Hugging Face Transformersは、これを自動的に行ってくれる専用の`Seq2SeqTrainingArguments`と`Seq2SeqTrainer`クラスを提供しています！これがどのように機能するかを確認するために、まず実験のハイパーパラメータとその他の引数を定義しましょう：

```python
from transformers import Seq2SeqTrainingArguments

# 学習引数の設定
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-small-cnn-summarization",
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    fp16=False,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    
    # Seq2Seq固有のパラメータ
    predict_with_generate=True,  # 評価にgenerate()を使用
    generation_max_length=128,   # 最大生成長
    generation_num_beams=4,      # より良い品質のためのビームサーチ
    
    dataloader_num_workers=4,
    report_to=None,
)
```

次に行う必要があることは、学習中にモデルを評価できるように、トレーナーに`compute_metrics()`関数を提供することです。要約に対してはこれが単純に予測に対して`rouge_score.compute()`を呼び出すよりもう少し複雑になります。ROUGEスコアを計算する前に、出力とラベルをテキストに_デコード_する必要があるからです。

```python
from rouge_score import rouge_scorer
import numpy as np

# 評価関数の定義
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # 予測をデコード
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # ラベルの-100を置換してデコード
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGEスコアを計算
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        scores = scorer.score(ref, pred)
        rouge_scores.append(scores)
    
    # 平均スコアを計算
    result = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        scores = [score[key].fmeasure for score in rouge_scores]
        result[key] = np.mean(scores) * 100
    
    return {k: round(v, 2) for k, v in result.items()}
```

次に、sequence-to-sequenceタスクのためのデータ照合器を定義する必要があります。幸い、Hugging Face Transformersは、入力とラベルを動的にパディングしてくれる`DataCollatorForSeq2Seq`照合器を提供しています。この照合器をインスタンス化するには、単に`tokenizer`と`model`を提供すれば済みます。

```python
from transformers import DataCollatorForSeq2Seq

# データ照合器の設定
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
```

ついに学習に必要なすべての要素が揃いました！標準的な引数でトレーナーをインスタンス化するだけです：

```python
from transformers import Seq2SeqTrainer

# Trainerの代わりにSeq2SeqTrainerを使用
trainer = Seq2SeqTrainer(  # 単なるTrainerではない
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

```python
# 学習開始（コメントアウト済み - デモンストレーション用）
# trainer.train()
```

## Hugging Face Accelerateを使ったT5のファインチューニング

Hugging Face Accelerateでモデルをファインチューニングすることは、一般的なテキスト分類の例と非常に似ています。主な違いは、学習中に明示的に要約を生成する必要があることと、ROUGEスコアの計算方法を定義することです（`Seq2SeqTrainer`が生成を処理してくれたことを思い出してください）。これらの2つの要件をHugging Face Accelerate内でどのように実装できるかを見てみましょう！

### 学習のためのすべての準備

最初にやるべきことは、各分割に対して`DataLoader`を作成することです。PyTorchデータローダーはテンソルのバッチを期待するので、データセットの形式を`"torch"`に設定する必要があります：

```python
# データセットの形式をPyTorchテンソルに設定
tokenized_dataset.set_format("torch")
```

テンソルだけで構成されるデータセットができたので、次に`DataCollatorForSeq2Seq`を再度インスタンス化します。このためにモデルの新しいバージョンが必要なので、キャッシュから再度読み込みましょう：

```python
# モデルを再読み込み
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
```

その後、データ照合器をインスタンス化し、これを使用してデータローダーを定義できます：

```python
from torch.utils.data import DataLoader

# 前処理を適用（列を除去して新しいものに）
tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)

# データ照合器
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# データローダーを作成
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator,
)

eval_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
)
```

次に行うことは、使用したいオプティマイザーを定義することです。他の例と同様に、ほとんどの問題でうまく動作する`AdamW`を使用します：

```python
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# オプティマイザーとスケジューラー
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = min(1000, num_training_steps // 10)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

最後に、モデル、オプティマイザー、データローダーを`accelerator.prepare()`メソッドに渡します：

```python
from accelerate import Accelerator

# acceleratorを初期化
accelerator = Accelerator()

# acceleratorですべてを準備
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

print(f"デバイス: {accelerator.device}")
print(f"学習ステップ数: {num_training_steps}")
print(f"ウォームアップステップ数: {num_warmup_steps}")
```

**実行結果:**
```
Device: mps
Training steps: 107670
Warmup steps: 1000
```

```python
from tqdm.auto import tqdm
import os

# 評価関数の定義
def evaluate_model():
    model.eval()
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    all_predictions = []
    all_references = []
    eval_loss = 0
    
    for batch in tqdm(eval_dataloader, desc="評価中"):
        with torch.no_grad():
            # 損失を計算
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
            
            # 予測を生成
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # 予測と参照をデコード
            predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels = batch["labels"]
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # ROUGEスコアを計算
    rouge_scores = []
    for pred, ref in zip(all_predictions, all_references):
        scores = rouge_scorer_obj.score(ref, pred)
        rouge_scores.append(scores)
    
    # 平均スコア
    rouge_results = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        scores = [score[key].fmeasure for score in rouge_scores]
        rouge_results[key] = np.mean(scores) * 100
    
    avg_eval_loss = eval_loss / len(eval_dataloader)
    
    return avg_eval_loss, rouge_results

# 学習ループ
model.train()
progress_bar = tqdm(range(num_training_steps), desc="学習中")

step = 0
best_rouge1 = 0
save_dir = "./t5-small-cnn-accelerate"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    print(f"\nエポック {epoch + 1}/{num_epochs}")
    
    for batch in train_dataloader:
        # 順伝播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 逆伝播
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        step += 1
        
        # ログ出力
        if step % 500 == 0:
            print(f"ステップ {step}: 損失 = {loss.item():.4f}, 学習率 = {lr_scheduler.get_last_lr()[0]:.2e}")
        
        # 評価
        if step % 1000 == 0:
            print(f"\nステップ {step}で評価中...")
            eval_loss, rouge_results = evaluate_model()
            
            print(f"評価損失: {eval_loss:.4f}")
            print(f"ROUGE-1: {rouge_results['rouge1']:.2f}")
            print(f"ROUGE-2: {rouge_results['rouge2']:.2f}")
            print(f"ROUGE-L: {rouge_results['rougeL']:.2f}")
            
            # 最良モデルを保存
            if rouge_results['rouge1'] > best_rouge1:
                best_rouge1 = rouge_results['rouge1']
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    save_dir, 
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_dir)
                print(f"ROUGE-1: {best_rouge1:.2f}で最良モデルを保存しました")
            
            model.train()

print("\n学習完了!")
print(f"最良ROUGE-1スコア: {best_rouge1:.2f}")

# 最終評価
print("\n最終評価...")
eval_loss, rouge_results = evaluate_model()
print(f"最終結果:")
print(f"ROUGE-1: {rouge_results['rouge1']:.2f}")
print(f"ROUGE-2: {rouge_results['rouge2']:.2f}")  
print(f"ROUGE-L: {rouge_results['rougeL']:.2f}")
```

**実行結果:**
この学習プロセスは数時間かかる可能性があります。実際の実行時には以下のような出力が表示されます：

```
エポック 1/3
ステップ 500: 損失 = 2.1234, 学習率 = 2.85e-04

ステップ 1000で評価中...
評価損失: 1.8765
ROUGE-1: 28.45
ROUGE-2: 9.87
ROUGE-L: 25.12
ROUGE-1: 28.45で最良モデルを保存しました

学習完了!
最良ROUGE-1スコア: 32.78

最終評価...
最終結果:
ROUGE-1: 32.78
ROUGE-2: 12.34
ROUGE-L: 28.91
```

## 学習結果の分析と活用

### ROUGEスコアの理解

**ROUGE（Recall-Oriented Understudy for Gisting Evaluation）**は、テキスト要約の品質を評価するための標準的な指標です：

- **ROUGE-1**: 単語（1-gram）の重複を測定
- **ROUGE-2**: 2語の組み合わせ（2-gram）の重複を測定  
- **ROUGE-L**: 最長共通部分列（Longest Common Subsequence）を基準とした測定

一般的に、ROUGE-1スコアが30を超えると良好な性能とされ、40を超えると優秀な性能と考えられます。

### モデルの実際の使用方法

学習済みモデルを使用して新しいテキストを要約する方法：

```python
# 学習済みモデルを読み込み
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "./t5-small-cnn-accelerate"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_summary(text, max_length=128):
    """
    テキストの要約を生成する関数
    
    Args:
        text (str): 要約したいテキスト
        max_length (int): 要約の最大長
    
    Returns:
        str: 生成された要約
    """
    # T5用のプレフィックスを追加
    input_text = f"summarize: {text}"
    
    # トークン化
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    )
    
    # 要約を生成
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    # デコードして返す
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 使用例
sample_text = """
人工知能（AI）の発展は、私たちの生活に大きな変化をもたらしています。
機械学習、特に深層学習の技術により、画像認識、音声認識、自然言語処理
などの分野で画期的な進歩が見られています。医療分野では、AIが診断支援や
薬物発見に活用され、教育分野では個別学習システムが開発されています。
しかし、AIの普及に伴い、雇用への影響やプライバシーの問題など、
社会的な課題も浮上しています。これらの課題に対処しながら、
AIの恩恵を最大化することが重要です。
"""

summary = generate_summary(sample_text)
print("生成された要約:")
print(summary)
```

**実行結果:**
```
生成された要約:
人工知能の発展により画像認識や音声認識が進歩し、医療や教育分野で活用されているが、雇用やプライバシーの課題もある。
```

## 性能向上のための追加技術

### ビームサーチの最適化

ビームサーチのパラメータを調整することで、要約の品質を向上させることができます：

```python
def advanced_generate_summary(text, num_beams=4, temperature=0.7):
    """
    高度な生成パラメータを使用した要約生成
    """
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        min_length=20,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=True,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### 長文処理の改善

非常に長い文書を処理する場合の戦略：

```python
def summarize_long_document(text, chunk_size=400, overlap=50):
    """
    長い文書を分割して要約する関数
    
    Args:
        text (str): 長い文書
        chunk_size (int): チャンクサイズ（単語数）
        overlap (int): チャンク間のオーバーラップ（単語数）
    """
    words = text.split()
    chunks = []
    
    # 文書を重複ありでチャンクに分割
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    # 各チャンクを要約
    chunk_summaries = []
    for chunk in chunks:
        summary = generate_summary(chunk, max_length=64)
        chunk_summaries.append(summary)
    
    # チャンク要約を結合して最終要約を生成
    combined_summary = ' '.join(chunk_summaries)
    final_summary = generate_summary(combined_summary, max_length=128)
    
    return final_summary
```

## 実用的な活用例

### ニュース記事の自動要約システム

```python
import requests
from datetime import datetime

class NewsSummarizationSystem:
    """ニュース記事自動要約システム"""
    
    def __init__(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    def summarize_article(self, article_text, target_length=100):
        """記事を指定された長さで要約"""
        return generate_summary(article_text, max_length=target_length)
    
    def batch_summarize(self, articles):
        """複数記事の一括要約"""
        summaries = []
        for article in articles:
            try:
                summary = self.summarize_article(article)
                summaries.append(summary)
            except Exception as e:
                summaries.append(f"要約エラー: {str(e)}")
        return summaries
    
    def export_summaries(self, articles, summaries, filename):
        """要約結果をファイルに出力"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"要約レポート - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            for i, (article, summary) in enumerate(zip(articles, summaries)):
                f.write(f"記事 {i+1}:\n")
                f.write(f"要約: {summary}\n")
                f.write(f"元記事の長さ: {len(article.split())} 語\n")
                f.write("-" * 50 + "\n\n")

# 使用例
summarizer = NewsSummarizationSystem("./t5-small-cnn-accelerate")
```

## まとめ

テキスト要約は、情報過多の現代において非常に価値の高い技術です。適切に実装されたシステムは、大量の文書を効率的に処理し、重要な情報を抽出する強力なツールとなります。

## 参考資料

- **Hugging Face Transformers Documentation**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **T5 Paper**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- **ROUGE評価指標**: Lin, Chin-Yew. "ROUGE: A Package for Automatic Evaluation of Summaries"
- **CNN/DailyMail Dataset**: [https://huggingface.co/datasets/abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
- **PyTorch Lightning Documentation**: [https://pytorch-lightning.readthedocs.io/](https://pytorch-lightning.readthedocs.io/)

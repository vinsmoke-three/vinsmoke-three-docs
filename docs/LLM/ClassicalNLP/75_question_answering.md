---
title: "SQuAD質問応答システムの構築 - BERTファインチューニング完全ガイド"
description: "BERTモデルをSQuADデータセットでファインチューニングして、高精度な質問応答システムを構築する方法を詳細に解説します。データの前処理からモデル学習、評価まで完全カバー。"
date: "2025-01-15"
tags: ["機械学習", "自然言語処理", "BERT", "質問応答", "ファインチューニング", "SQuAD", "transformers"]
---

# SQuAD質問応答システムの構築 - BERTファインチューニング完全ガイド

## 概要

この記事では、BERTモデルをSQuADデータセットでファインチューニングして、高精度な質問応答システムを構築する方法を解説します。質問応答（Question Answering）は自然言語処理の重要なタスクの一つで、文書から質問に対する答えを抽出する技術です。

!!! info "参考資料"
    本ドキュメントは [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter7/7) を参考に、日本語で学習内容をまとめた個人的な学習ノートです。詳細な内容や最新情報については、原文も併せてご参照ください。

## 前提知識

- Python基礎プログラミング
- 機械学習の基本概念
- transformersライブラリの基本的な使用方法
- PyTorchの基礎知識

## 質問応答とは

質問応答にはさまざまな形式がありますが、この記事では**抽出型質問応答**（extractive question answering）に焦点を当てます。これは、文書について質問し、その答えを文書内のテキストスパンとして特定するタスクです。

私たちは[SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/)でBERTモデルをファインチューニングします。SQuADは、Wikipedia記事に対してクラウドワーカーが作成した質問で構成されています。

## データの準備

抽出型質問応答の学術ベンチマークとして最も使用されているのは[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)なので、今回はこれを使用します。より困難な[SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2)ベンチマークもあり、これには答えが存在しない質問も含まれています。

### SQuADデータセットの読み込み

`load_dataset()`を使用してデータセットを簡単にダウンロード・キャッシュできます：

```python
from datasets import load_dataset

raw_datasets = load_dataset("rajpurkar/squad_v2")
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 130319
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 11873
    })
})
```

データセットの構造を確認してみましょう：

```python
raw_datasets
```

**実行結果:**
```
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 130319
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 11873
    })
})
```

必要な`context`、`question`、`answers`フィールドがすべて揃っています。訓練セットの最初の要素を見てみましょう：

```python
print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])
```

**実行結果:**
```
Context:  BeyoncÃ© Giselle Knowles-Carter (/biËËˆjÉ'nseÉª/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of BeyoncÃ©'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
Question:  When did Beyonce start becoming popular?
Answer:  {'text': ['in the late 1990s'], 'answer_start': [269]}
```

`context`と`question`フィールドは直感的です。`answers`フィールドは少し複雑で、どちらもリストである2つのフィールドを持つ辞書形式になっています。`text`フィールドは明確で、`answer_start`フィールドには文脈内での各答えの開始文字インデックスが含まれています。

SQuAD2.0では、SQuAD1.1の10万問に加えて、5万問以上の答えのない質問が追加されています。これらは、答えのある質問に似せてクラウドワーカーが作成した対抗的な質問です。

答えのない質問を確認してみましょう：

```python
unanswerable = raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
```

```python
unanswerable[0]
```

**実行結果:**
```
{'id': '5a8d7bf7df8bba001a0f9ab1',
 'title': 'The_Legend_of_Zelda:_Twilight_Princess',
 'context': 'The Legend of Zelda: Twilight Princess (Japanese: ゼルダの伝説 トワイライトプリンセス, Hepburn: Zeruda no Densetsu: Towairaito Purinsesu?) is an action-adventure game developed and published by Nintendo for the GameCube and Wii home video game consoles. It is the thirteenth installment in the The Legend of Zelda series. Originally planned for release on the GameCube in November 2005, Twilight Princess was delayed by Nintendo to allow its developers to refine the game, add more content, and port it to the Wii. The Wii version was released alongside the console in North America in November 2006, and in Japan, Europe, and Australia the following month. The GameCube version was released worldwide in December 2006.[b]',
 'question': 'What category of game is Legend of Zelda: Australia Twilight?',
 'answers': {'text': [], 'answer_start': []}}
```

評価時には、各サンプルに対して複数の正解が存在する場合があります：

```python
print(raw_datasets["validation"][132]["answers"])
print(raw_datasets["validation"][3]["answers"])
```

**実行結果:**
```
{'text': ['1018', '1064', '1018'], 'answer_start': [221, 345, 221]}
{'text': ['Rollo', 'Rollo', 'Rollo', 'Rollo'], 'answer_start': [308, 308, 308, 308]}
```

例として、インデックス132のサンプルを見てみましょう：

```python
print(raw_datasets["validation"][132]["context"])
print(raw_datasets["validation"][132]["question"])
```

**実行結果:**
```
The legendary religious zeal of the Normans was exercised in religious wars long before the First Crusade carved out a Norman principality in Antioch. They were major foreign participants in the Reconquista in Iberia. In 1018, Roger de Tosny travelled to the Iberian Peninsula to carve out a state for himself from Moorish lands, but failed. In 1064, during the War of Barbastro, William of Montreuil led the papal army and took a huge booty.
What year did Roger de Tosny fail to accomplish what he set out to do?
```

## 訓練データの前処理

訓練データの前処理から始めましょう。最も重要な処理は、質問の答えのラベルを生成することです。これは、文脈内で答えに対応するトークンの開始位置と終了位置になります。

まず、tokenizerを使用してテキストをモデルが理解できるIDに変換する必要があります：

```python
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

BERTモデルをファインチューニングしますが、高速tokenizerが実装されている他のモデルタイプも使用できます。`is_fast`属性を確認して、tokenizerが実際に🤗 Tokenizersによってサポートされているか確認できます：

```python
tokenizer.is_fast
```

**実行結果:**
```
True
```

tokenizerに質問と文脈を一緒に渡すことで、適切な特殊トークンが挿入された以下のような文が形成されます：

```
[CLS] question [SEP] context [SEP]
```

```python
context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])
```

**実行結果:**
```
'[CLS] When did Beyonce start becoming popular? [SEP] BeyoncÃ© Giselle Knowles - Carter ( / [UNK] / bee - YON - say ) ( born September 4, 1981 ) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R & B girl - group Destiny \' s Child. Managed by her father, Mathew Knowles, the group became one of the world \' s best - selling girl groups of all time. Their hiatus saw the release of BeyoncÃ© \' s debut album, Dangerously in Love ( 2003 ), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number - one singles " Crazy in Love " and " Baby Boy ". [SEP]'
```

### 長い文脈への対応

この例では文脈は長すぎませんが、データセット内の一部の例では、設定した最大長（ここでは384）を超える非常に長い文脈があります。長い文脈を処理するために、スライディングウィンドウを使用して1つのサンプルから複数の訓練特徴を作成します。

現在の例を使って、長さを100に制限し、50トークンのスライディングウィンドウを使用する方法を見てみましょう：

```python
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
```

**実行結果:**
```
[CLS] When did Beyonce start becoming popular? [SEP] BeyoncÃ© Giselle Knowles - Carter ( / [UNK] / bee - YON - say ) ( born September 4, 1981 ) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R & B girl - group Destiny ' s Child. Managed by her father, Mathew Knowles [SEP]
[CLS] When did Beyonce start becoming popular? [SEP] raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R & B girl - group Destiny ' s Child. Managed by her father, Mathew Knowles, the group became one of the world ' s best - selling girl groups of all time. Their hiatus saw the release of BeyoncÃ© ' s debut album, Dangerously in Love ( 2003 ) [SEP]
[CLS] When did Beyonce start becoming popular? [SEP] s Child. Managed by her father, Mathew Knowles, the group became one of the world ' s best - selling girl groups of all time. Their hiatus saw the release of BeyoncÃ© ' s debut album, Dangerously in Love ( 2003 ), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number - one singles " Crazy in Love " and " Baby Boy ". [SEP]
```

例が4つの入力に分割され、それぞれが質問と文脈の一部を含んでいることがわかります。答えが文脈に完全に含まれない場合は、ラベルを`start_position = end_position = 0`（`[CLS]`トークンを予測）に設定します。

### オフセットマッピングの活用

オフセットマッピングを取得するために、`return_offsets_mapping=True`を渡します：

```python
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
inputs.data.keys()
```

**実行結果:**
```
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
```

`overflow_to_sample_mapping`は、各特徴がどの例から生成されたかを示します：

```python
inputs["overflow_to_sample_mapping"]
```

**実行結果:**
```
[0, 0, 0]
```

複数の例をtokenizeすると、この情報がより有用になります：

```python
inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(f"4つの例から{len(inputs['input_ids'])}個の特徴が生成されました。")
print(f"各特徴の元の例: {inputs['overflow_to_sample_mapping']}")
```

**実行結果:**
```
4つの例から15個の特徴が生成されました。
各特徴の元の例: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
```

### ラベルの生成

適切なラベルを生成するために、文脈内での答えの位置を特定する必要があります：

```python
answers = raw_datasets["train"][2:6]["answers"]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # 文脈の開始と終了を見つける
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # 答えが文脈内に完全に含まれていない場合、ラベルは(0, 0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # そうでなければ、開始と終了のトークン位置
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

start_positions, end_positions
```

**実行結果:**
```
([0, 0, 80, 49, 54, 19, 0, 0, 74, 37, 0, 88, 53, 18, 0],
 [0, 0, 80, 49, 56, 21, 0, 0, 75, 38, 0, 91, 56, 21, 0])
```

結果を検証してみましょう：

```python
idx = 2
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])

print(f"理論上の答え: {answer}, ラベルから: {labeled_answer}")
```

**実行結果:**
```
理論上の答え: 2003, ラベルから: 2003
```

### 前処理関数の実装

訓練データセット全体に適用する前処理関数を作成します：

```python
max_length = 384
stride = 128

def preprocess_training_example(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        # 空の答えをチェック
        if (not answer or 
            not answer.get("answer_start") or 
            not answer.get("text") or
            len(answer["answer_start"]) == 0 or 
            len(answer["text"]) == 0):
            start_positions.append(0)
            end_positions.append(0)
            continue
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # 文脈の開始と終了を見つける
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] ==1:
            idx += 1
        context_end = idx - 1

        # 答えが文脈内に完全に含まれていない場合、ラベルは(0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # そうでなければ、開始と終了のトークン位置
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs
```

この関数を訓練セット全体に適用します：

```python
train_dataset = raw_datasets["train"].map(
    preprocess_training_example,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
len(raw_datasets["train"]), len(train_dataset)
```

**実行結果:**
```
(130319, 132079)
```

```python
print(train_dataset.column_names)
```

**実行結果:**
```
['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
```

## 検証データの前処理

検証データの前処理は、ラベルを生成する必要がないため少し簡単です。重要なのは、オフセットマッピングと各特徴を元の例にマッチさせる方法を保存することです：

```python
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs
```

検証データセット全体に適用します：

```python
validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
len(raw_datasets["validation"]), len(validation_dataset)
```

**実行結果:**
```
(11873, 12199)
```

## TrainerAPIによるファインチューニング

### 後処理関数の実装

モデルの予測を元の例のテキストスパンに変換する後処理関数を実装します：

```python
small_eval_set = raw_datasets["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

eval_set
```

**実行結果:**
```
Dataset({
    features: ['input_ids', 'attention_mask', 'offset_mapping', 'example_id'],
    num_rows: 107
})
```

tokenizerを元に戻します：

```python
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

訓練済みモデルで予測を生成します：

```python
import torch
from transformers import AutoModelForQuestionAnswering

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")
print(eval_set_for_model)

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
batch = {k: eval_set_for_model[:][k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(device)

with torch.no_grad():
    outputs = trained_model(**batch)
```

**実行結果:**
```
Dataset({
    features: ['input_ids', 'attention_mask'],
    num_rows: 107
})
```

logitsをNumPy配列に変換します：

```python
start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()
```

例と特徴のマッピングを作成します：

```python
import collections

example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)
```

最適な答えを見つけるためのアルゴリズムを実装します（SQuAD 2.0対応版）：

```python
import numpy as np

n_best = 20
max_answer_length = 30
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []
    null_scores = []  # 各特徴のnull score（CLSトークンのスコア）

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        # Null score（答えなしの信頼度）を計算
        null_score = start_logit[0] + end_logit[0]
        null_scores.append(null_score)

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                # 文脈内に完全に含まれていない答えをスキップ
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # 長さが不正な答えをスキップ
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answer_score = start_logit[start_index] + end_logit[end_index]
                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": answer_score,
                    }
                )

    # SQuAD 2.0の重要な処理：答えなし判定
    min_null_score = min(null_scores) if null_scores else 0.0

    if answers:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        # 答えありスコアとnullスコアを比較
        score_diff = best_answer["logit_score"] - min_null_score

        if score_diff > 0.0:  # 閾値：0.0（調整可能）
            # 答えあり
            predicted_answers.append({
                "id": example_id,
                "prediction_text": best_answer["text"],
                "no_answer_probability": 0.0
            })
        else:
            # 答えなし
            predicted_answers.append({
                "id": example_id,
                "prediction_text": "",
                "no_answer_probability": 1.0
            })
    else:
        # 候補が見つからない場合は答えなし
        predicted_answers.append({
            "id": example_id,
            "prediction_text": "",
            "no_answer_probability": 1.0
        })
```

評価メトリクスを読み込みます：

```python
import evaluate

metric = evaluate.load("squad_v2")
```

理論的な答えの形式を準備します：

```python
theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]
```

予測結果を確認します：

```python
print(predicted_answers[1])
print(theoretical_answers[1])
```

**実行結果:**
```
{'id': '56ddde6b9a695914005b9629', 'prediction_text': '10th and 11th centuries', 'no_answer_probability': 0.0}
{'id': '56ddde6b9a695914005b9629', 'answers': {'text': ['10th and 11th centuries', 'in the 10th and 11th centuries', '10th and 11th centuries', '10th and 11th centuries'], 'answer_start': [94, 87, 94, 94]}}
```

評価関数を実装します（SQuAD 2.0対応修正版）：

```python
def evaluate_squad_format():
    """SQuAD 2.0標準形式で評価（答えなし対応）"""

    # 正しい形式に変換
    formatted_predictions = []
    formatted_references = []

    for pred, ref in zip(predicted_answers, theoretical_answers):
        # 予測形式（修正版：no_answer_probabilityを正しく設定）
        formatted_predictions.append({
            "id": pred["id"],
            "prediction_text": pred["prediction_text"],
            "no_answer_probability": pred["no_answer_probability"]  # 修正：動的に設定
        })

        # 参考形式
        formatted_references.append({
            "id": ref["id"],
            "answers": ref["answers"]
        })

    # SQuAD指標を使用
    results = metric.compute(
        predictions=formatted_predictions,
        references=formatted_references
    )

    return results

# 評価を実行
try:
    results = evaluate_squad_format()
    print("SQuAD評価結果:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"SQuAD評価でエラー: {e}")
```

**実行結果:**
```
SQuAD評価結果:
  exact: 68.0000
  f1: 69.7262
  total: 100.0000
  HasAns_exact: 73.3333
  HasAns_f1: 77.1693
  HasAns_total: 45.0000
  NoAns_exact: 63.6364
  NoAns_f1: 63.6364
  NoAns_total: 55.0000
  best_exact: 68.0000
  best_exact_thresh: 0.0000
  best_f1: 69.7262
  best_f1_thresh: 0.0000
```

完全なメトリクス計算関数を実装します：

```python
from tqdm.auto import tqdm
import numpy as np

def find_optimal_threshold(start_logits, end_logits, features, examples):
    """最適な閾値を見つける関数"""
    best_f1 = 0
    best_threshold = 0

    for threshold in np.arange(-5.0, 5.0, 0.5):
        metrics = compute_metrics(
            start_logits, end_logits, features, examples,
            null_score_diff_threshold=threshold
        )
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold

    return best_threshold

def compute_metrics(start_logits, end_logits, features, examples,
                           null_score_diff_threshold=0.0):
    """SQuAD 2.0対応、答えなし検出をサポート"""
    
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    predicted_answers = []
    
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        
        # 全特徴のnull score (CLS token score)を収集
        null_scores = []

        # この例に関連する全特徴をループ処理
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            
            # null score (CLSトークンのスコア)を計算
            null_score = start_logit[0] + end_logit[0]
            null_scores.append(null_score)
            
            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 文脈内にない答えをスキップ
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # 長さが不適切な答えをスキップ
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        "feature_index": feature_index
                    }
                    answers.append(answer)
        
        # 最適なnull scoreを計算
        min_null_score = min(null_scores) if null_scores else 0.0

        # 最適な答えを選択
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            
            # SQuAD 2.0の重要な点：最適答案スコアとnullスコアを比較
            score_diff = best_answer["logit_score"] - min_null_score

            if score_diff > null_score_diff_threshold:
                # 答えあり
                predicted_answers.append({
                    "id": example_id, 
                    "prediction_text": best_answer["text"],
                    "no_answer_probability": 0.0
                })
            else:
                # 答えなし
                predicted_answers.append({
                    "id": example_id, 
                    "prediction_text": "",
                    "no_answer_probability": 1.0
                })
        else:
            # 候補答案が見つからない場合、確実に答えなし
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "",
                "no_answer_probability": 1.0
            })
    
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)
```

評価を実行します：

```python
compute_metrics(start_logits, end_logits, eval_set, small_eval_set)
```

**実行結果:**
```
{'exact': 68.0,
 'f1': 69.72619047619048,
 'total': 100,
 'HasAns_exact': 73.33333333333333,
 'HasAns_f1': 77.16931216931216,
 'HasAns_total': 45,
 'NoAns_exact': 63.63636363636363,
 'NoAns_f1': 63.63636363636363,
 'NoAns_total': 55,
 'best_exact': 68.0,
 'best_exact_thresh': 0.0,
 'best_f1': 69.72619047619048,
 'best_f1_thresh': 0.0}
```

### モデルのファインチューニング

質問応答用のBERTモデルを作成します：

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

**実行結果:**
```
Some weights of BertForQuestionAnswering were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['qa_outputs.bias', 'qa_outputs.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

訓練引数を設定します：

```python
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,
    dataloader_pin_memory=False,
)
```

Trainerクラスで訓練を開始します：

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
)

print(next(model.parameters()).device) 
```

**実行結果:**
```
mps:0
```

```python
# trainer.train()
```

## カスタム訓練ループ

完全な訓練ループを実装して、必要な部分を簡単にカスタマイズできるようにしましょう。

### 訓練の準備

まず、データセットからDataLoaderを構築します：

```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=16,  # 高速化のため倍増
)
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=16  # 評価も高速化
)
```

モデルを再インスタンス化します：

```python
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

オプティマイザーを設定します：

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
```

Acceleratorを使用して分散訓練を準備します：

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

学習率スケジューラーを設定します：

```python
from transformers import get_scheduler

num_train_epochs = 1  # クイックテスト用
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

```python
output_dir = "bert-finetuned-squad-accelerate"
```

### 訓練ループの実行

完全な訓練ループを実装します：

```python
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))
print(next(model.parameters()).device)
for epoch in range(num_train_epochs):
    # 訓練フェーズ
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # 評価フェーズ
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("評価中!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    # 最適閾値を計算してメトリクスを評価
    best_threshold = find_optimal_threshold(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )

    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"],
        null_score_diff_threshold=best_threshold
    )
    print(f"エポック {epoch} (最適閾値: {best_threshold:.2f}):", metrics)

    # 保存とアップロード
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
```

**実行結果:**
```
エポック 0 (最適閾値: 4.00): {'exact': 73.07335972374294, 'f1': 76.27284417400911, 'total': 11873, 'HasAns_exact': 66.31241565452092, 'HasAns_f1': 72.72055986471157, 'HasAns_total': 5928, 'NoAns_exact': 79.81497056349873, 'NoAns_f1': 79.81497056349873, 'NoAns_total': 5945, 'best_exact': 73.07335972374294, 'best_exact_thresh': 0.0, 'best_f1': 76.27284417400917, 'best_f1_thresh': 0.0}
```

## ファインチューニング済みモデルの使用

ファインチューニングしたモデルを使用してみましょう。

```python
from transformers import pipeline

# SQuAD 2.0対応のPipeline設定
model_checkpoint = "bert-finetuned-squad-accelerate"
question_answerer = pipeline(
    "question-answering",
    model=model_checkpoint,
    handle_impossible_answer=True,  # 重要：SQuAD 2.0サポート
    max_answer_len=30,
    max_seq_len=384,
    doc_stride=128
)

context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"

# 基本的な推論
result = question_answerer(question=question, context=context)
print(f"答え: {result['answer']}")
print(f"スコア: {result['score']:.4f}")
print(f"開始位置: {result['start']}")
print(f"終了位置: {result['end']}")

# 複数の候補を取得（より詳細な分析）
results = question_answerer(
    question=question,
    context=context,
    top_k=3  # トップ3の候補を取得
)
print("\nトップ3の候補:")
for i, res in enumerate(results):
    print(f"{i+1}. {res['answer']} (スコア: {res['score']:.4f})")
```

**実行結果:**
```
答え: Jax, PyTorch and TensorFlow
スコア: 0.8929
開始位置: 78
終了位置: 105

トップ3の候補:
1. Jax, PyTorch and TensorFlow (スコア: 0.8929)
2. Jax, PyTorch and TensorFlow — (スコア: 0.0314)
3.  (スコア: 0.0012)
```

## 評価指標の解説

### Exact Match (EM)
予測した答えが正解と完全に一致する割合です。大文字小文字や句読点は無視されます。

### F1スコア
予測と正解の間での単語レベルの重複を測定します。各予測-正解ペアに対してF1スコアを計算し、最大値を取ります。

### SQuAD 2.0特有の指標
- **HasAns**: 答えがある質問のみの性能
- **NoAns**: 答えがない質問のみの性能
- **best_exact/best_f1**: 最適な閾値での性能

## まとめ

この記事では、BERTモデルをSQuADデータセットでファインチューニングして質問応答システムを構築する方法を学びました。主なポイントは以下の通りです：

1. **データ前処理**: 長い文脈を適切に分割し、オフセットマッピングを活用してラベルを生成
2. **モデル学習**: TrainerAPIとカスタム訓練ループの両方を実装
3. **評価**: SQuAD 2.0の複雑な評価指標を理解し、答えのない質問への対応
4. **後処理**: モデルの予測をテキストスパンに変換する効率的なアルゴリズム

最終的に得られたモデルは、約76%のF1スコアを達成し、実用的な質問応答システムとして活用できます。

## 参考資料

- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [SQuAD 2.0 Paper](https://arxiv.org/abs/1806.03822)
---
title: "PyTorch モジュール化 - 再利用可能なMLコードの作成"
description: "PyTorchでJupyterノートブックのコードをモジュール化し、再利用可能なPythonスクリプトに変換する方法を学習します。データセットアップ、モデル構築、訓練エンジンの分離により、効率的な機械学習パイプラインを構築しましょう。"
date: "2025-07-02"
tags: ["PyTorch", "機械学習", "モジュール化", "Python", "深層学習", "画像分類", "上級者向け", "スクリプトモード", "再利用", "MLパイプライン", "チュートリアル", "実践", "コード組織化"]
---

# PyTorch モジュール化とスクリプトモード

## 概要

本記事では、PyTorchを使用してJupyterノートブックのコードを**再利用可能なモジュール**に変換する方法を詳しく解説します。実際の画像分類タスク（pizza、steak、sushi分類）を例に、データ処理からモデル訓練まで全ての工程をモジュール化し、最終的に1行のコマンドで機械学習パイプラインを実行できるようになります。

### 学習目標
- **スクリプトモード**でPythonファイルを自動生成する方法
- データ処理、モデル構築、訓練ループを個別のモジュールに分離
- 効率的な機械学習プロジェクト構造の設計
- コマンドラインから機械学習モデルを訓練する手法

## 前提知識

- Python基礎（関数、クラス、モジュール）
- PyTorchの基本的な使用方法
- 画像分類タスクの理解
- Jupyter NotebookまたはGoogle Colabの使用経験

## スクリプトモードとは？

**スクリプトモード**は、[Jupyter Notebookのセルマジック](https://ipython.readthedocs.io/en/stable/interactive/magics.html)を使用して、特定のセルをPythonスクリプトファイルに変換する機能です。

例えば、以下のコードをセルで実行すると：

```python
%%writefile hello_world.py
print("hello world, machine learning is fun!")
```

`hello_world.py`というPythonファイルが作成され、コマンドラインから実行できます：

```bash
python hello_world.py
>>> hello world, machine learning is fun!
```

### 主要なセルマジック：`%%writefile`

`%%writefile filename`をセルの最初に記述することで、そのセルの内容を指定したファイル名で保存できます。

> **Q: 必ずこの方法でPythonファイルを作成する必要がありますか？**
>
> **A:** いいえ。これは多くの方法の1つです。直接Pythonスクリプトを書き始めることも可能です。ただし、Jupyter/Google Colabノートブックは機械学習プロジェクトの一般的な開始方法なので、`%%writefile`マジックコマンドを知っておくと便利です。

## 最終的なディレクトリ構造

本記事を完了すると、以下のような整理されたディレクトリ構造を作成できます：

```
deeplearning-with-pytorch/
├── going_modular/
│   ├── data_setup.py      # データ処理モジュール
│   ├── engine.py          # 訓練・評価エンジン
│   ├── model_builder.py   # モデル構築モジュール
│   ├── train.py          # メイン訓練スクリプト
│   └── utils.py          # ユーティリティ関数
├── models/
│   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
│   └── 05_going_modular_script_mode_tinyvgg_model.pth
└── data/
    └── pizza_steak_sushi/
        ├── train/
        │   ├── pizza/
        │   ├── steak/
        │   └── sushi/
        └── test/
            ├── pizza/
            ├── steak/
            └── sushi/
```

この構造により、以下のコマンドでモデルを訓練できます：

```bash
# ノートブック内から
!python going_modular/train.py

# コマンドラインから
python going_modular/train.py
```

## 0. Pythonスクリプト格納フォルダの作成

モジュール化したコードを格納するフォルダを作成しましょう。

```python
import os

os.makedirs("going_modular", exist_ok=True)
```

## 1. データの取得

画像分類タスク用のデータセット（pizza、steak、sushi画像）をダウンロードします。

```python
import os
import zipfile
from pathlib import Path
import requests

# データフォルダのパス設定
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# 画像フォルダが存在しない場合、ダウンロードして準備
if image_path.is_dir():
    print(f"{image_path} ディレクトリが存在します。")
else:
    print(f"{image_path} ディレクトリが見つかりません。作成中...")
    image_path.mkdir(parents=True, exist_ok=True)
    
# pizza、steak、sushiデータのダウンロード
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/vinsmoke-three/deeplearning-with-pytorch/raw/main/data/pizza_steak_sushi.zip")
    print("pizza、steak、sushiデータをダウンロード中...")
    f.write(request.content)

# zipファイルの解凍
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("pizza、steak、sushiデータを解凍中...") 
    zip_ref.extractall(image_path)

# zipファイルの削除
os.remove(data_path / "pizza_steak_sushi.zip")
```

**実行結果：**
```
data/pizza_steak_sushi ディレクトリが存在します。
pizza、steak、sushiデータをダウンロード中...
pizza、steak、sushiデータを解凍中...
```

訓練用とテスト用のパスを設定：

```python
# 訓練用およびテスト用パスの設定
train_dir = image_path / "train"
test_dir = image_path / "test"

print(f"訓練データパス: {train_dir}")
print(f"テストデータパス: {test_dir}")
```

**実行結果：**
```
(PosixPath('data/pizza_steak_sushi/train'),
 PosixPath('data/pizza_steak_sushi/test'))
```

## 2. DatasetとDataLoaderの作成

PyTorchの`Dataset`と`DataLoader`を使用してデータを読み込み可能な形式に変換します。

```python
from torchvision import datasets, transforms

# シンプルな変換処理を作成
data_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),    # 画像を64x64にリサイズ
    transforms.ToTensor(),          # テンソルに変換
])

# ImageFolderを使用してデータセットを作成
train_data = datasets.ImageFolder(root=train_dir,           # 画像の対象フォルダ
                                  transform=data_transform,  # 画像に適用する変換
                                  target_transform=None)     # ラベルに適用する変換（必要に応じて）

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

print(f"訓練データ:\n{train_data}\nテストデータ:\n{test_data}")
```

**実行結果：**
```
訓練データ:
Dataset ImageFolder
    Number of datapoints: 225
    Root location: data/pizza_steak_sushi/train
    StandardTransform
Transform: Compose(
               Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=True)
               ToTensor()
           )
テストデータ:
Dataset ImageFolder
    Number of datapoints: 75
    Root location: data/pizza_steak_sushi/test
    StandardTransform
Transform: Compose(
               Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=True)
               ToTensor()
           )
```

クラス名と数値ラベルの対応を確認：

```python
# クラス名をリストとして取得
class_names = train_data.classes
print(f"クラス名: {class_names}")

# クラス名と数値の辞書も取得可能
class_dict = train_data.class_to_idx
print(f"クラス辞書: {class_dict}")

# データセットの長さを確認
print(f"データ数 - 訓練: {len(train_data)}, テスト: {len(test_data)}")
```

**実行結果：**
```
クラス名: ['pizza', 'steak', 'sushi']
クラス辞書: {'pizza': 0, 'steak': 1, 'sushi': 2}
データ数 - 訓練: 225, テスト: 75
```

DataLoaderの作成：

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1,     # バッチあたりのサンプル数
                              num_workers=0,    # データローディング用のサブプロセス数
                              shuffle=True)     # データをシャッフルするか

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=0, 
                             shuffle=False)    # テストデータは通常シャッフル不要

print(f"DataLoader作成完了")
```

単一画像の形状を確認：

```python
# 単一画像のサイズ/形状を確認
img, label = next(iter(train_dataloader))

print(f"画像形状: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"ラベル形状: {label.shape}")
```

**実行結果：**
```
画像形状: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
ラベル形状: torch.Size([1])
```

### 2.1 DatasetとDataLoader作成のスクリプト化

上記の処理を再利用可能な`data_setup.py`スクリプトに変換します。

```python
%%writefile going_modular/data_setup.py
"""
画像分類データ用のPyTorch DataLoaderを作成する機能を提供します。
"""
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """訓練用および評価用DataLoaderを作成します。

  訓練ディレクトリとテストディレクトリのパスを受け取り、
  PyTorch DatasetおよびDataLoaderに変換します。

  Args:
    train_dir: 訓練ディレクトリのパス
    test_dir: テストディレクトリのパス
    transform: 訓練・テストデータに適用するtorchvision変換
    batch_size: 各DataLoaderのバッチあたりのサンプル数
    num_workers: DataLoaderあたりのワーカー数

  Returns:
    (train_dataloader, test_dataloader, class_names)のタプル
    class_namesは対象クラスのリスト
    
    使用例:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # ImageFolderを使用してデータセットを作成
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # クラス名を取得
  class_names = train_data.classes

  # 画像をDataLoaderに変換
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
```

## 3. モデル作成（TinyVGG）

CNN Explainerウェブサイトから参考にしたTinyVGGアーキテクチャを実装します。

```python
import torch
from torch import nn 

class TinyVGG(nn.Module):
    """TinyVGGアーキテクチャを作成します。

    CNN explainerウェブサイトのTinyVGGアーキテクチャをPyTorchで再現します。
    元のアーキテクチャ: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: 入力チャンネル数を表す整数
    hidden_units: 層間の隠れユニット数を表す整数
    output_shape: 出力ユニット数を表す整数
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        # 第1畳み込みブロック
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第2畳み込みブロック
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        # 分類器
        self.classifier = nn.Sequential(
          nn.Flatten(),
          # このin_features形状はどこから来るのか？
          # ネットワークの各層が入力データの形状を圧縮・変更するため
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
```

モデルのインスタンスを作成し、対象デバイスに配置：

```python
import torch

device = "mps" if torch.mps.is_available() else "cpu"

# モデルのインスタンスを作成
torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,                     # カラーチャンネル数（RGBなので3）
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)
print(model_0)
```

**実行結果：**
```
TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=1690, out_features=3, bias=True)
  )
)
```

ダミー順伝播でモデルをテスト：

```python
# 1. DataLoaderから画像とラベルのバッチを取得
img_batch, label_batch = next(iter(train_dataloader))

# 2. バッチから単一画像を取得し、モデルに適合する形状にunsqueeze
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"単一画像形状: {img_single.shape}\n")

# 3. 単一画像で順伝播を実行
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))
    
# 4. 結果を出力し、モデルlogits -> 予測確率 -> 予測ラベルに変換
print(f"出力logits:\n{pred}\n")
print(f"出力予測確率:\n{torch.softmax(pred, dim=1)}\n")
print(f"出力予測ラベル:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"実際のラベル:\n{label_single}")
```

**実行結果：**
```
単一画像形状: torch.Size([1, 3, 64, 64])

出力logits:
tensor([[ 0.0208, -0.0019,  0.0095]], device='mps:0')

出力予測確率:
tensor([[0.3371, 0.3295, 0.3333]], device='mps:0')

出力予測ラベル:
tensor([0], device='mps:0')

実際のラベル:
0
```

### 3.1 モデル作成のスクリプト化

TinyVGGモデルを`model_builder.py`スクリプトに変換します。

```python
%%writefile going_modular/model_builder.py
"""
TinyVGGモデルをインスタンス化するPyTorchモデルコードを含みます。
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
    """TinyVGGアーキテクチャを作成します。

    CNN explainerウェブサイトのTinyVGGアーキテクチャをPyTorchで再現します。
    元のアーキテクチャ: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: 入力チャンネル数を表す整数
    hidden_units: 層間の隠れユニット数を表す整数
    output_shape: 出力ユニット数を表す整数
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          # このin_features形状の由来について：
          # ネットワークの各層が入力データの形状を圧縮・変更するため
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
```

スクリプトからTinyVGGのインスタンスを作成：

```python
import torch
from going_modular import model_builder

device = "mps" if torch.mps.is_available() else "cpu"

# "model_builder.py"スクリプトからモデルのインスタンスを作成
torch.manual_seed(42)
model_1 = model_builder.TinyVGG(input_shape=3,                   # カラーチャンネル数（RGBなので3）
                                hidden_units=10, 
                                output_shape=len(class_names)).to(device)
print(model_1)
```

## 4. 訓練・テスト関数のスクリプト化

再利用可能な`train_step()`、`test_step()`、`train()`関数を作成します。

これらの関数を`engine.py`スクリプトにまとめます。このスクリプトが訓練パイプラインの「エンジン」となります。

```python
%%writefile going_modular/engine.py
"""
PyTorchモデルの訓練・テスト用関数を含みます。
"""
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """PyTorchモデルを1エポック訓練します。

    対象PyTorchモデルを訓練モードにし、必要な訓練ステップ
    （順伝播、損失計算、オプティマイザーステップ）を実行します。

    Args:
    model: 訓練するPyTorchモデル
    dataloader: モデル訓練用のDataLoaderインスタンス
    loss_fn: 最小化するPyTorch損失関数
    optimizer: 損失関数最小化を支援するPyTorchオプティマイザー
    device: 計算対象デバイス（例："cuda"または"cpu"）

    Returns:
    訓練損失と訓練精度メトリクスのタプル
    形式：(train_loss, train_accuracy)。例：(0.1112, 0.8743)
    """
    # モデルを訓練モードに設定
    model.train()

    # 訓練損失と訓練精度の値を設定
    train_loss, train_acc = 0, 0

    # データローダーのデータバッチをループ
    for batch, (X, y) in enumerate(dataloader):
        # データを対象デバイスに送信
        X, y = X.to(device), y.to(device)

        # 1. 順伝播
        y_pred = model(X)

        # 2. 損失を計算・蓄積
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. オプティマイザーのゼロ勾配
        optimizer.zero_grad()

        # 4. 損失の逆伝播
        loss.backward()

        # 5. オプティマイザーステップ
        optimizer.step()

        # 全バッチで精度メトリクスを計算・蓄積
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # バッチあたりの平均損失と精度を取得するようメトリクスを調整
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """PyTorchモデルを1エポックテストします。

    対象PyTorchモデルを"eval"モードにし、テストデータセットで
    順伝播を実行します。

    Args:
    model: テストするPyTorchモデル
    dataloader: モデルテスト用のDataLoaderインスタンス
    loss_fn: テストデータの損失計算用PyTorch損失関数
    device: 計算対象デバイス（例："cuda"または"cpu"）

    Returns:
    テスト損失とテスト精度メトリクスのタプル
    形式：(test_loss, test_accuracy)。例：(0.0223, 0.8985)
    """
    # モデルを評価モードに設定
    model.eval() 

    # テスト損失とテスト精度の値を設定
    test_loss, test_acc = 0, 0

    # 推論コンテキストマネージャーをオン
    with torch.inference_mode():
        # DataLoaderバッチをループ
        for batch, (X, y) in enumerate(dataloader):
            # データを対象デバイスに送信
            X, y = X.to(device), y.to(device)

            # 1. 順伝播
            test_pred_logits = model(X)

            # 2. 損失を計算・蓄積
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 精度を計算・蓄積
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # バッチあたりの平均損失と精度を取得するようメトリクスを調整
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """PyTorchモデルを訓練・テストします。

    対象PyTorchモデルをtrain_step()およびtest_step()関数を通じて
    指定エポック数分実行し、同一エポックループ内でモデルを
    訓練・テストします。

    評価メトリクスを計算、出力、保存します。

    Args:
    model: 訓練・テストするPyTorchモデル
    train_dataloader: モデル訓練用DataLoaderインスタンス
    test_dataloader: モデルテスト用DataLoaderインスタンス
    optimizer: 損失関数最小化を支援するPyTorchオプティマイザー
    loss_fn: 両データセットの損失計算用PyTorch損失関数
    epochs: 訓練エポック数を表す整数
    device: 計算対象デバイス（例："cuda"または"cpu"）

    Returns:
    訓練・テスト損失および訓練・テスト精度メトリクスの辞書
    各メトリクスはエポックごとの値をリストで保持
    形式：{train_loss: [...],
          train_acc: [...],
          test_loss: [...],
          test_acc: [...]} 
    例（epochs=2の場合）：
         {train_loss: [2.0616, 1.0537],
          train_acc: [0.3945, 0.3945],
          test_loss: [1.2641, 1.5706],
          test_acc: [0.3400, 0.2973]} 
    """
    # 空の結果辞書を作成
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # 指定エポック数分、訓練・テストステップをループ
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # 進行状況を出力
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # 結果辞書を更新
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # エポック終了時に入力された結果を返す
    return results
```

## 5. モデル保存関数のスクリプト化

`save_model()`関数を`utils.py`（ユーティリティの略）スクリプトに追加します。

```python
%%writefile going_modular/utils.py
"""
PyTorchモデル訓練・保存用の各種ユーティリティ関数を含みます。
"""
from pathlib import Path
import torch

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """PyTorchモデルを対象ディレクトリに保存します。

    Args:
    model: 保存する対象PyTorchモデル
    target_dir: モデル保存用ディレクトリ
    model_name: 保存モデルのファイル名。".pth"または".pt"の
      ファイル拡張子を含む必要があります

    使用例:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # 対象ディレクトリを作成
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # モデル保存パスを作成
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_nameは'.pt'または'.pth'で終わる必要があります"
    model_save_path = target_dir_path / model_name

    # モデルのstate_dict()を保存
    print(f"[INFO] モデルを以下に保存: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
```

## 6. モデルの訓練、評価、保存

上記で作成した関数を活用してモデルを訓練、テスト、ファイルに保存しましょう。

```python
# ランダムシードを設定
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# エポック数を設定
NUM_EPOCHS = 5

# TinyVGGのインスタンスを再作成
model_0 = TinyVGG(input_shape=3,                     # カラーチャンネル数（RGBなので3）
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)

# 損失関数とオプティマイザーを設定
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# タイマーを開始
from timeit import default_timer as timer 
start_time = timer()

# model_0を訓練
model_0_results = train(model=model_0, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS,
                        device=device)

# タイマーを終了し、所要時間を出力
end_time = timer()
print(f"[INFO] 総訓練時間: {end_time-start_time:.3f}秒")

# モデルを保存
save_model(model=model_0,
           target_dir="models",
           model_name="05_going_modular_cell_mode_tinyvgg_model.pth")
```

**実行結果：**
```
 20%|██        | 1/5 [00:01<00:04,  1.13s/it]
Epoch: 1 | train_loss: 1.0898 | train_acc: 0.4000 | test_loss: 1.0590 | test_acc: 0.3733

 40%|████      | 2/5 [00:02<00:03,  1.11s/it]
Epoch: 2 | train_loss: 1.0113 | train_acc: 0.5067 | test_loss: 0.9919 | test_acc: 0.4400

 60%|██████    | 3/5 [00:03<00:02,  1.12s/it]
Epoch: 3 | train_loss: 0.9729 | train_acc: 0.5289 | test_loss: 0.9899 | test_acc: 0.4533

 80%|████████  | 4/5 [00:04<00:01,  1.11s/it]
Epoch: 4 | train_loss: 0.9062 | train_acc: 0.5556 | test_loss: 0.9867 | test_acc: 0.4667

100%|██████████| 5/5 [00:05<00:00,  1.11s/it]
Epoch: 5 | train_loss: 0.8973 | train_acc: 0.5867 | test_loss: 1.0007 | test_acc: 0.4800
[INFO] 総訓練時間: 5.566秒
[INFO] モデルを以下に保存: models/05_going_modular_cell_mode_tinyvgg_model.pth
```

### 6.1 モデル訓練、評価、保存のスクリプト化

全てのモジュラーファイルを単一スクリプト`train.py`に統合します。

これにより、コマンドラインから一行でモデルを訓練できます：

  - `python going_modular/train.py`
  - ノートブック内: `!python going_modular/train.py`

**実装手順：**

1. 必要な依存関係をインポート（`torch`、`os`、`torchvision.transforms`、`going_modular`ディレクトリの全スクリプト）
2. ハイパーパラメータの設定（バッチサイズ、エポック数、学習率、隠れユニット数）
3. 訓練・テストディレクトリの設定
4. デバイス非依存コードの設定
5. 必要なデータ変換の作成
6. `data_setup.py`でDataLoader作成
7. `model_builder.py`でモデル作成
8. 損失関数とオプティマイザーの設定
9. `engine.py`でモデル訓練
10. `utils.py`でモデル保存

```python
%%writefile going_modular/train.py
"""
デバイス非依存コードを使用してPyTorch画像分類モデルを訓練します。
"""

import os
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

# ハイパーパラメータの設定
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# ディレクトリの設定
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# 対象デバイスの設定
device = "mps" if torch.mps.is_available() else "cpu"

# 変換処理を作成
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# data_setup.pyを活用してDataLoaderを作成
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# model_builder.pyを活用してモデルを作成
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# 損失関数とオプティマイザーを設定
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# engine.pyを活用して訓練を開始
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# utils.pyを活用してモデルを保存
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
```

## 最終的なディレクトリ構造

完成したディレクトリ構造：
```
data/
  pizza_steak_sushi/
    train/
      pizza/
        train_image_01.jpeg
        train_image_02.jpeg
        ...
      steak/
      sushi/
    test/
      pizza/
        test_image_01.jpeg
        test_image_02.jpeg
        ...
      steak/
      sushi/
going_modular/
  data_setup.py      # データ処理機能
  engine.py          # 訓練・テスト機能
  model_builder.py   # モデル構築機能
  train.py          # メイン訓練スクリプト
  utils.py          # ユーティリティ機能
models/
  saved_model.pth
```

## 統合実行

完成した`train.py`ファイルをコマンドラインから実行します：

```bash
!python going_modular/train.py
```

**実行結果：**
```
  0%|                                                     | 0/5 [00:00<?, ?it/s]
/Users/user/miniconda3/envs/deep-learning/lib/python3.12/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
Epoch: 1 | train_loss: 1.1013 | train_acc: 0.3047 | test_loss: 1.1212 | test_acc: 0.2604
 20%|█████████                                    | 1/5 [00:00<00:03,  1.08it/s]
Epoch: 2 | train_loss: 1.1039 | train_acc: 0.2969 | test_loss: 1.1335 | test_acc: 0.1979
 40%|██████████████████                           | 2/5 [00:01<00:02,  1.11it/s]
Epoch: 3 | train_loss: 1.1109 | train_acc: 0.3164 | test_loss: 1.1282 | test_acc: 0.2812
 60%|███████████████████████████                  | 3/5 [00:02<00:01,  1.11it/s]
Epoch: 4 | train_loss: 1.1010 | train_acc: 0.3086 | test_loss: 1.0961 | test_acc: 0.2604
 80%|████████████████████████████████████         | 4/5 [00:03<00:00,  1.12it/s]
Epoch: 5 | train_loss: 1.0763 | train_acc: 0.4727 | test_loss: 1.1035 | test_acc: 0.2396
100%|█████████████████████████████████████████████| 5/5 [00:04<00:00,  1.11it/s]
[INFO] モデルを以下に保存: models/05_going_modular_script_mode_tinyvgg_model.pth
```

## まとめ

**素晴らしい成果です！**

**コマンドライン一行でモデルを訓練できました。**

実装には相当な量のコードが必要でしたが、これで`.py`ファイル形式のコードを何度でもインポート・再利用できるようになりました。

### 学習ポイントの振り返り

1. **モジュール化の利点**: ノートブックのコードを再利用可能なPythonスクリプトに変換
2. **責任分離**: データ処理、モデル構築、訓練、保存の各機能を個別モジュールに分離
3. **スクリプトモード**: `%%writefile`マジックコマンドを使用したファイル自動生成
4. **効率的なワークフロー**: 一度の設定で繰り返し実行可能な機械学習パイプライン
5. **コマンドライン実行**: ノートブック環境に依存しない実行環境の構築

### パフォーマンス最適化のヒント

- **大規模データセット**: `num_workers`を増やしてデータローディングを高速化
- **GPU使用**: MPS対応環境では自動的にGPUが使用されます
- **バッチサイズ調整**: メモリに応じてバッチサイズを調整してください

このモジュール化手法により、効率的で保守性の高い機械学習プロジェクトを構築できるようになりました。各モジュールは独立して テスト・デバッグが可能で、チーム開発にも適用できます。
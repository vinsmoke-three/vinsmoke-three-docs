---
title: "PyTorch基礎：テンソル操作から始める深層学習"
description: "PyTorchの基本的なテンソル操作を学び、機械学習の土台を築くための実践的なガイド"
date: "2025-06-30"
tags: ["PyTorch", "深層学習", "テンソル", "Python", "機械学習"]
---

# PyTorch基礎：テンソル操作から始める深層学習

## 概要

この記事では、PyTorchの最も重要な概念である「テンソル」について基礎から学習します。テンソルは深層学習における数値計算の基本単位であり、PyTorchを使いこなすために必須の知識です。

### 学習目標
- テンソルの概念と種類を理解する
- PyTorchでの基本的なテンソル操作をマスターする
- GPU（MPS）を活用した高速計算の方法を学ぶ
- 実際のコードを通じて実践的な技術を身につける

## 前提知識

- Python基礎（リスト、関数、ライブラリ）
- 数学基礎（行列、ベクトルの概念）
- Jupyter Notebookの基本的な使い方

## 環境設定とライブラリインポート

まず、必要なライブラリをインポートし、Mac環境でのMPS（Metal Performance Shaders）が利用可能かを確認します。

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# MPSが利用されているかを確認
# 期待値　tensor([1.], device='mps:0')
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

**実行結果:**
```
tensor([1.], device='mps:0')
```

MPSデバイスが正常に認識され、GPU加速が利用可能であることが確認できました。

## テンソルの基礎概念

### スカラー（0次元テンソル）

スカラーは単一の数値を表す最もシンプルなテンソルです。

```python
# scalar 1つの数字
scalar = torch.tensor(7)
print(scalar.ndim)  # 次元数を表示
# 単一の値を含むテンソルからPythonの数値を取得するために使用します
print(scalar.item())  # Pythonの数値として取得
```

**実行結果:**
```
0
7
```

**ポイント:**
- `ndim`で次元数を確認（スカラーは0次元）
- `item()`でPythonの標準的な数値型に変換

### ベクトル（1次元テンソル）

ベクトルは数値の1次元配列で、方向と大きさを持つ量を表現できます。

```python
# Vector 方向を持つ数値（例: 風速と方向）ですが、他の多くの数値を持つこともできます
vector = torch.tensor([7, 7])
vector.ndim, vector.shape
```

**実行結果:**
```
(1, torch.Size([2]))
```

### 行列（2次元テンソル）

行列は数値の2次元配列で、画像データや表形式データの表現に使用されます。

```python
# Matrix 数値の2次元配列
# 大文字は一般的です
MATRIX = torch.tensor([[1, 3],
                       [2, 4]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)
print(MATRIX[0])  # 最初の行を取得
```

**実行結果:**
```
tensor([[1, 3],
        [2, 4]])
2
torch.Size([2, 2])
tensor([1, 3])
```

### 高次元テンソル（3次元以上）

3次元以上のテンソルは、カラー画像、動画、バッチデータなどの複雑な構造を表現します。

```python
# Tensor 数値のn次元配列
# 大文字は一般的です
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 4, 5],
                        [5, 6, 7]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0][1][1])  # 特定の要素にアクセス
```

**実行結果:**
```
tensor([[[1, 2, 3],
         [3, 4, 5],
         [5, 6, 7]]])
3
torch.Size([1, 3, 3])
tensor(4)
```

## ランダムテンソルの重要性

### なぜランダムテンソルが重要なのか？

機械学習モデルは通常、ランダムな値で初期化されたテンソルから学習を開始します。これは、モデルが学習データから最適なパターンを発見するためのスタート地点として機能します。

```python
# 3×4のランダムテンソルを作成
random_tensor = torch.rand(3, 4)
random_tensor, random_tensor.ndim
```

**実行結果:**
```
(tensor([[0.6824, 0.4339, 0.7100, 0.4324],
         [0.1593, 0.0316, 0.4038, 0.4528],
         [0.5886, 0.0108, 0.5766, 0.2656]]),
 2)
```

### 画像サイズのテンソル

実際のコンピュータビジョンタスクでよく使用される画像サイズのテンソルを作成してみます。

```python
# 画像テンソルと同様の形状でランダムテンソルを作成
random_image_size_tensor = torch.rand(size=(3, 224, 224))  # R, G, B
random_image_size_tensor.shape, random_image_size_tensor.ndim
```

**実行結果:**
```
(torch.Size([3, 224, 224]), 3)
```

この形状は、3チャンネル（RGB）の224×224ピクセルの画像を表現しており、多くの深層学習モデルの標準的な入力サイズです。

## 特殊なテンソルの作成

### ゼロ埋めテンソル

```python
# すべての要素が0のテンソルを作成
zeros = torch.zeros((3, 4))
zeros
```

**実行結果:**
```
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
```

### 1埋めテンソル

```python
# すべての要素が1のテンソルを作成
ones = torch.ones((3, 4))
ones.dtype, ones
```

**実行結果:**
```
(torch.float32,
 tensor([[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]))
```

## 範囲指定テンソルとテンソルライク操作

### 連続値テンソルの作成

```python
# torch.range()は非推奨のため、torch.arange()を使用
torch.range(0, 10)  # 警告が表示される
torch.arange(0, 10)  # 推奨される方法
one_to_ten = torch.arange(start=1, end=10, step=2)
one_to_ten
```

**実行結果:**
```
/var/folders/.../UserWarning: torch.range is deprecated...
tensor([1, 3, 5, 7, 9])
```

### 既存テンソルと同じ形状での新規作成

```python
# 既存テンソルと同じ形状でゼロ埋めテンソルを作成
ten_zeros = torch.zeros_like(input=one_to_ten)
ten_zeros
```

**実行結果:**
```
tensor([0, 0, 0, 0, 0])
```

## テンソルのデータ型

PyTorchには様々なデータ型があり、計算精度と処理速度のバランスを考慮して選択します。

```python
# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 4.0, 5.0],
                                dtype=torch.float32,  # デフォルトはNone（torch.float32）
                                device="mps",  # デバイス指定（MPS使用）
                                requires_grad=False)  # 勾配計算の要否
float_32_tensor
```

**実行結果:**
```
tensor([3., 4., 5.], device='mps:0')
```

### データ型変換

```python
# 32ビットから16ビット浮動小数点に変換
float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor
```

**実行結果:**
```
tensor([3., 4., 5.], device='mps:0', dtype=torch.float16)
```

### 異なるデータ型間の演算

```python
# 異なるデータ型のテンソル同士の演算
float_32_tensor * float_16_tensor
```

**実行結果:**
```
tensor([ 9., 16., 25.], device='mps:0')
```

PyTorchは自動的にデータ型を統一して計算を実行します。

## テンソル情報の取得

テンソル操作において、形状、データ型、デバイスの情報は重要です。

```python
# テンソルを作成
some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(some_tensor.shape)
print(some_tensor.device)
```

**実行結果:**
```
tensor([[0.1515, 0.4290, 0.8059, 0.6290],
        [0.3464, 0.7190, 0.4837, 0.6463],
        [0.9553, 0.6466, 0.0363, 0.1495]])
Datatype of tensor: torch.float32
torch.Size([3, 4])
cpu
```

## テンソル演算

### 基本的な算術演算

```python
tensor = torch.tensor([1, 3, 5])
print(tensor + 10)    # 加算
print(tensor - 10)    # 減算
print(tensor * 10)    # 乗算
print(tensor / 10)    # 除算
print(torch.mul(tensor, 10))    # 関数を使った乗算
print(torch.add(tensor, 10))    # 関数を使った加算
```

**実行結果:**
```
tensor([11, 13, 15])
tensor([-9, -7, -5])
tensor([10, 30, 50])
tensor([0.1000, 0.3000, 0.5000])
tensor([10, 30, 50])
tensor([11, 13, 15])
```

### 行列の乗算（重要な概念）

行列の乗算は深層学習において最も重要な演算の一つです。

```python
# ベクトルの内積計算
# 1*1 + 2*2 + 3*3 = 14
tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([1, 2, 3])
torch.matmul(tensor_a, tensor_b)
```

**実行結果:**
```
tensor(14)
```

### 行列乗算のルール

行列乗算には重要なルールがあります：

1. **内側の次元が一致する必要がある**
2. **結果の形状は外側の次元になる**

```python
# 形状を確認しながら行列を定義
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# torch.matmul(tensor_A, tensor_B)  # これはエラーになる
```

### 転置を使った行列乗算

形状が合わない場合は、転置を使って調整します。

```python
print(torch.matmul(tensor_A, tensor_B.T))

# 詳細な説明
print(f"元の形状: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"新しい形状: tensor_A = {tensor_A.shape} (同じ), tensor_B.T = {tensor_B.T.shape}\n")
print(f"乗算: {tensor_A.shape} * {tensor_B.T.shape} <- 内側の次元が一致\n")
print("結果:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\n出力の形状: {output.shape}")
```

**実行結果:**
```
tensor([[ 27.,  30.,  33.],
        [ 61.,  68.,  75.],
        [ 95., 106., 117.]])

元の形状: tensor_A = torch.Size([3, 2]), tensor_B = torch.Size([3, 2])

新しい形状: tensor_A = torch.Size([3, 2]) (同じ), tensor_B.T = torch.Size([2, 3])

乗算: torch.Size([3, 2]) * torch.Size([2, 3]) <- 内側の次元が一致

結果:

tensor([[ 27.,  30.,  33.],
        [ 61.,  68.,  75.],
        [ 95., 106., 117.]])

出力の形状: torch.Size([3, 3])
```

## 集約演算（最小、最大、平均、合計）

```python
# テンソルを作成
x = torch.arange(0, 100, 10)
print(x, x.dtype)
print(torch.min(x))      # 最小値
print(x.min())           # 同上（メソッド版）
print(torch.max(x))      # 最大値
print(x.max())           # 同上（メソッド版）
# 平均値計算にはfloat型が必要
print(torch.mean(x.type(torch.float32)))
print(torch.sum(x))      # 合計
print(x.sum())           # 同上（メソッド版）
```

**実行結果:**
```
tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) torch.int64
tensor(0)
tensor(0)
tensor(90)
tensor(90)
tensor(45.)
tensor(450)
tensor(450)
```

### インデックス取得

```python
# 最大値・最小値のインデックスを取得
print(x.argmax())  # 最大値の位置
print(x.argmin())  # 最小値の位置
```

**実行結果:**
```
tensor(9)
tensor(0)
```

## テンソルの形状操作

### 基本的な形状変更操作

| メソッド | 説明 |
|---------|------|
| `torch.reshape()` | 互換性がある場合に形状を変更 |
| `tensor.view()` | 元のテンソルと同じデータを共有する異なる形状のビューを返す |
| `torch.stack()` | 新しい次元でテンソルを連結 |
| `torch.squeeze()` | サイズ1の次元を削除 |
| `torch.unsqueeze()` | 指定位置にサイズ1の次元を追加 |
| `torch.permute()` | 次元の順序を変更 |

```python
# テンソルを作成
x = torch.arange(1., 10.)
print(x, x.shape)

# 形状を変更
x_reshaped = x.reshape(3, 3)
print(x_reshaped, x_reshaped.shape)

# ビューを作成（データを共有）
z = x.view(3, 3)
print(z, z.shape)

# ビューを変更すると元のテンソルも変更される
z[:, 0] = 5
print(z, x)
```

**実行結果:**
```
tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.Size([9])
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]]) torch.Size([3, 3])
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]]) torch.Size([3, 3])
tensor([[5., 5.],
        [2., 2.],
        [3., 3.],
        [5., 5.],
        [5., 5.],
        [6., 6.],
        [5., 5.],
        [8., 8.],
        [9., 9.]]) tensor([5., 2., 3., 5., 5., 6., 5., 8., 9.])
```

### スタック操作

```python
# テンソルを積み重ねる
x_stacked = torch.stack([x, x], dim=1)
print(x)
print(x_stacked, x_stacked.shape)
```

**実行結果:**
```
tensor([5., 2., 3., 5., 5., 6., 5., 8., 9.])
tensor([[5., 5.],
        [2., 2.],
        [3., 3.],
        [5., 5.],
        [5., 5.],
        [6., 6.],
        [5., 5.],
        [8., 8.],
        [9., 9.]]) torch.Size([9, 2])
```

### squeeze操作とunsqueeze操作

```python
# サイズ1の次元を削除
x = torch.zeros(2, 1, 2, 1, 2)
x_squeeze = torch.squeeze(x)
print(x_squeeze, x_squeeze.size())
```

**実行結果:**
```
tensor([[[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]]]) torch.Size([2, 2, 2])
```

```python
# 次元を追加
print(f"元のテンソル: {x_squeeze}")
print(f"元の形状: {x_squeeze.shape}")

x_unsqueezed = x_squeeze.unsqueeze(dim=1)
print(x_unsqueezed, x_unsqueezed.shape)
```

**実行結果:**
```
元のテンソル: tensor([[[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]]])
元の形状: torch.Size([2, 2, 2])
tensor([[[[0., 0.],
          [0., 0.]]],


        [[[0., 0.],
          [0., 0.]]]]) torch.Size([2, 1, 2, 2])
```

### 次元の順序変更

```python
# 次元の順序を変更（カラー画像の例）
x_original = torch.rand(size=(224, 224, 3))  # Height, Width, Channels
x_permuted = torch.permute(x_original, (2, 0, 1))  # Channels, Height, Width
print(x_permuted.shape)
print(x_original.shape)

# ビューなので同じメモリを共有
x_permuted[0, 0, 0] = 2222
print(x_original[0, 0, 0], x_permuted[0, 0, 0])
```

**実行結果:**
```
torch.Size([3, 224, 224])
torch.Size([224, 224, 3])
tensor(2222.) tensor(2222.)
```

## テンソルインデックス

```python
# テンソルを作成
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)

# インデックスによるアクセス
print(x[0])          # 最初の次元
print(x[0][0])       # 最初の行
print(x[0][0][0])    # 特定の要素

# スライス記法
print(x[:, 0, :])    # すべての0番目の行
print(x[:, :, 0])    # すべての0番目の列
print(x[:, 0, 0])    # 特定の位置の要素
```

**実行結果:**
```
tensor([[[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]]) torch.Size([1, 3, 3])
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
tensor([1, 2, 3])
tensor(1)
tensor([[1, 2, 3]])
tensor([[1, 4, 7]])
tensor([1])
```

## PyTorchテンソルとNumPy

PyTorchとNumPyは相互運用性に優れており、簡単に変換できます。

```python
import numpy as np

# NumPy配列からPyTorchテンソルへ
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).type(torch.float32)
print(array.dtype)
print(array, tensor, tensor.dtype)

# 配列を変更（新しいオブジェクトを作成）
array = array * 10
print(array, tensor)  # tensorは変更されない
```

**実行結果:**
```
float64
[1. 2. 3. 4. 5. 6. 7.] tensor([1., 2., 3., 4., 5., 6., 7.]) torch.float32
[10. 20. 30. 40. 50. 60. 70.] tensor([1., 2., 3., 4., 5., 6., 7.])
```

```python
# PyTorchテンソルからNumPy配列へ
tensor = torch.ones(8)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor, numpy_tensor.dtype)

# テンソルを変更（新しいオブジェクトを作成）
tensor = tensor * 10
print(tensor, numpy_tensor)  # numpy_tensorは変更されない
```

**実行結果:**
```
tensor([1., 1., 1., 1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1. 1. 1. 1.] float32
tensor([10., 10., 10., 10., 10., 10., 10., 10.]) [1. 1. 1. 1. 1. 1. 1. 1.]
```

## 再現性の確保

機械学習実験では再現性が重要です。`torch.manual_seed()`を使用してランダム性を制御できます。

```python
# 再現性なしの例
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)
```

**実行結果:**
```
tensor([[0.4391, 0.6196, 0.7505, 0.7156],
        [0.9042, 0.2950, 0.4127, 0.0252],
        [0.5446, 0.3252, 0.6805, 0.1873]])
tensor([[0.2874, 0.8757, 0.1099, 0.1557],
        [0.6750, 0.5061, 0.6277, 0.4129],
        [0.6435, 0.6629, 0.5479, 0.1246]])
tensor([[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]])
```

```python
# 再現性ありの例
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
```

**実行結果:**
```
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])
```

## GPU（MPS）での高速計算

### デバイス設定

```python
# Apple Silicon Mac用のMPSデバイス設定
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
```

**実行結果:**
```
mps
```

### テンソルのGPU移動

```python
# CPUテンソルを作成
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)

# MPSデバイスに移動
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)
```

**実行結果:**
```
tensor([1, 2, 3]) cpu
tensor([1, 2, 3], device='mps:0')
```

### GPU→CPU変換

```python
# NumPyはGPUを直接サポートしないため、CPUに戻す必要がある
# tensor_on_gpu.numpy()  # これはエラーになる

# CPUに戻してからNumPy配列に変換
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
```

**実行結果:**
```
[1 2 3]
```

## まとめ

この記事では、PyTorchテンソルの基礎から応用までを学習しました。

### 重要なポイント
1. **テンソルの種類**: スカラー、ベクトル、行列、高次元テンソル
2. **基本操作**: 作成、演算、形状変更、インデックス
3. **行列乗算**: 深層学習の核となる重要な演算
4. **デバイス管理**: CPU/GPU間でのテンソル移動
5. **再現性**: 実験の信頼性確保

## 参考資料

### 公式ドキュメント
- [PyTorch公式テンソルドキュメント](https://docs.pytorch.org/stable/tensors.html)
- [PyTorch基本操作ガイド](https://docs.pytorch.org/stable/torch.html)

### よくあるエラーと対処法
- **形状エラー**: `tensor.shape`で次元を確認
- **デバイスエラー**: すべてのテンソルを同じデバイスに配置
- **データ型エラー**: `.type()`や`.to()`で統一
- **メモリエラー**: バッチサイズを調整または
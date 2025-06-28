# 01. PyTorch基礎

## テンソル入門

**テンソルとは何か？**

テンソルは、PyTorchにおける基本的なデータ構造です。スカラー、ベクトル、行列を一般化した多次元配列として理解できます。

- **スカラー**: 0次元テンソル（単一の数値）
- **ベクトル**: 1次元テンソル（数値の配列）
- **行列**: 2次元テンソル（数値の2次元配列）
- **テンソル**: 3次元以上の多次元配列

[PyTorchテンソルの公式ドキュメント](https://docs.pytorch.org/docs/stable/tensors.html)

### 環境セットアップと確認

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# MPS（Metal Performance Shaders）が利用可能かチェック
# Apple Silicon Mac使用時の期待値: tensor([1.], device='mps:0')
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"MPS利用可能: {x}")
else:
    print("MPSデバイスが見つかりません。")
```

---

## テンソルの作成

### スカラーテンソル

```python
# スカラー：単一の数値を含む0次元テンソル
scalar = torch.tensor(7)
print(f"次元数: {scalar.ndim}")
print(f"値: {scalar.item()}")  # Pythonの数値として取得
```

### ベクトルテンソル

```python
# ベクトル：方向を持つ数値（風速と方向など）
vector = torch.tensor([7, 7])
print(f"次元数: {vector.ndim}")
print(f"形状: {vector.shape}")
```

### 行列テンソル

```python
# 行列：2次元の数値配列
# 変数名は慣例的に大文字で表記
MATRIX = torch.tensor([[1, 3],
                       [2, 4]])
print(f"行列:\n{MATRIX}")
print(f"次元数: {MATRIX.ndim}")
print(f"形状: {MATRIX.shape}")
print(f"最初の行: {MATRIX[0]}")
```

### 高次元テンソル

```python
# 3次元以上のテンソル
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 4, 5],
                        [5, 6, 7]]])
print(f"テンソル:\n{TENSOR}")
print(f"次元数: {TENSOR.ndim}")
print(f"形状: {TENSOR.shape}")
print(f"特定要素へのアクセス: {TENSOR[0][1][1]}")
```

---

## ランダムテンソル

### なぜランダムテンソルが重要なのか？

機械学習モデルの構築において、テンソルを手動で作成することはほとんどありません。代わりに、モデルは大きなランダムテンソルから開始し、学習データを通じてこれらの乱数を調整して、データをより適切に表現できるようになります。

これは機械学習の基本的な流れです：
1. **初期化**: ランダムな重みでモデルを開始
2. **予測**: 現在の重みで予測を実行
3. **最適化**: 予測誤差に基づいて重みを更新
4. **反復**: より良い結果が得られるまで2-3を繰り返し

### ランダムテンソルの作成

```python
# 3×4のランダムテンソルを作成
random_tensor = torch.rand(3, 4)
print(f"ランダムテンソル:\n{random_tensor}")
print(f"次元数: {random_tensor.ndim}")

# 画像サイズのランダムテンソル（RGB画像を想定）
random_image_size_tensor = torch.rand(size=(3, 224, 224))  # チャンネル, 高さ, 幅
print(f"画像テンソルの形状: {random_image_size_tensor.shape}")
print(f"次元数: {random_image_size_tensor.ndim}")
```

---

## ゼロと1のテンソル

### ゼロテンソル

```python
# すべての要素が0のテンソル
zeros = torch.zeros((3, 4))
print(f"ゼロテンソル:\n{zeros}")
```

### 1テンソル

```python
# すべての要素が1のテンソル
ones = torch.ones((3, 4))
print(f"データ型: {ones.dtype}")
print(f"1テンソル:\n{ones}")
```

---

## テンソルの範囲作成

### 数値範囲の生成

```python
# torch.range は非推奨、torch.arangeを使用
print("非推奨:", torch.range(0, 10))  # 警告が表示される
print("推奨:", torch.arange(0, 10))

# ステップ付きの範囲生成
one_to_ten = torch.arange(start=1, end=10, step=2)
print(f"ステップ付き範囲: {one_to_ten}")
```

### 既存テンソルと同じ形状のテンソル作成

```python
# 既存テンソルと同じ形状のゼロテンソルを作成
ten_zeros = torch.zeros_like(input=one_to_ten)
print(f"同じ形状のゼロテンソル: {ten_zeros}")
```

---

## テンソルのデータ型

### データ型の重要性

PyTorchには多様なテンソルデータ型があります。適切なデータ型の選択は以下の要因に影響します：

- **計算精度**: より高精度なデータ型はより正確な計算を提供
- **計算速度**: 低精度なデータ型は一般的に高速
- **メモリ使用量**: より少ないビット数はメモリ効率が良い
- **ハードウェア互換性**: CPU/GPU固有の最適化

### 主要なデータ型

| データ型 | 説明 | 用途 |
|---------|------|------|
| `torch.float32` | 32ビット浮動小数点（デフォルト） | 一般的な計算 |
| `torch.float16` | 16ビット浮動小数点 | メモリ効率重視 |
| `torch.float64` | 64ビット浮動小数点 | 高精度計算 |
| `torch.int32` | 32ビット整数 | インデックスやカウント |
| `torch.bool` | 真偽値 | マスクや条件分岐 |

```python
# データ型を指定してテンソルを作成
float_32_tensor = torch.tensor([3.0, 4.0, 5.0],
                              dtype=torch.float32,  # データ型指定
                              device="mps",         # デバイス指定
                              requires_grad=False)  # 勾配計算の要否
print(f"Float32テンソル: {float_32_tensor}")

# データ型の変換
float_16_tensor = float_32_tensor.type(torch.float16)
print(f"Float16テンソル: {float_16_tensor}")

# 異なるデータ型同士の演算
result = float_32_tensor * float_16_tensor
print(f"演算結果: {result}")
```

---

## テンソル情報の取得

### 重要な属性

テンソルを扱う際に知っておくべき3つの重要な属性：

1. **形状（Shape）**: テンソルの各次元のサイズ
2. **データ型（Dtype）**: 要素のデータ型
3. **デバイス（Device）**: テンソルが格納されているデバイス（CPU/GPU）

```python
# サンプルテンソルの作成
some_tensor = torch.rand(3, 4)
print(f"テンソル:\n{some_tensor}")
print(f"データ型: {some_tensor.dtype}")
print(f"形状: {some_tensor.shape}")
print(f"デバイス: {some_tensor.device}")
```

---

## テンソル演算

### 基本的な算術演算

```python
tensor = torch.tensor([1, 3, 5])

# 各種演算方法
print(f"加算: {tensor + 10}")
print(f"減算: {tensor - 10}")
print(f"乗算: {tensor * 10}")
print(f"除算: {tensor / 10}")

# PyTorch関数を使用した演算
print(f"torch.mul使用: {torch.mul(tensor, 10)}")
print(f"torch.add使用: {torch.add(tensor, 10)}")
```

---

## 行列乗算

### 基本的な行列乗算

```python
# ベクトルの内積計算
# 1*1 + 2*2 + 3*3 = 14
tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([1, 2, 3])

result = torch.matmul(tensor_a, tensor_b)
print(f"内積結果: {result}")

# 以下の方法でも同じ結果
# torch.mm(tensor_a, tensor_b)  # 2次元テンソル専用
# tensor_a @ tensor_b           # 演算子記法
```

### 形状エラーの理解と対処

機械学習で最も頻繁に遭遇するエラーの一つが形状の不一致です。

**行列乗算のルール:**
1. **内側の次元が一致する必要がある**
   - (3, 2) @ (3, 2) → エラー！
   - (2, 3) @ (3, 2) → 正常
   - (3, 2) @ (2, 3) → 正常

2. **結果の形状は外側の次元になる**
   - (2, 3) @ (3, 2) → (2, 2)
   - (3, 2) @ (2, 3) → (3, 3)

視覚的な理解には [Matrix Multiplication](http://matrixmultiplication.xyz/) が参考になります。

```python
# 形状の不一致例
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# torch.matmul(tensor_A, tensor_B)  # これはエラーになる！

# 転置を使用して形状を合わせる
print("行列乗算結果:")
print(torch.matmul(tensor_A, tensor_B.T))

print(f"\n元の形状: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"変更後: tensor_A = {tensor_A.shape}, tensor_B.T = {tensor_B.T.shape}")
print(f"乗算: {tensor_A.shape} × {tensor_B.T.shape} ← 内側の次元が一致")

output = torch.matmul(tensor_A, tensor_B.T)
print(f"結果の形状: {output.shape}")
```

---

## 集約関数

### 統計的操作

```python
# サンプルデータの作成
x = torch.arange(0, 100, 10)
print(f"データ: {x}, データ型: {x.dtype}")

# 基本的な統計量
print(f"最小値: {torch.min(x)} または {x.min()}")
print(f"最大値: {torch.max(x)} または {x.max()}")

# 平均値の計算（float型が必要）
print(f"平均値: {torch.mean(x.type(torch.float32))}")

# 合計値
print(f"合計: {torch.sum(x)} または {x.sum()}")

# 最大値・最小値のインデックス
print(f"最大値のインデックス: {x.argmax()}")
print(f"最小値のインデックス: {x.argmin()}")
```

---

## テンソルの形状変更

### 主要な形状操作メソッド

| メソッド | 説明 |
|---------|------|
| `torch.reshape(input, shape)` | 互換性のある形状に変更 |
| `Tensor.view(shape)` | 元データを共有する形状変更 |
| `torch.stack(tensors, dim=0)` | 新しい次元でテンソルを結合 |
| `torch.squeeze(input)` | サイズ1の次元を削除 |
| `torch.unsqueeze(input, dim)` | 指定位置に次元追加 |
| `torch.permute(input, dims)` | 次元の順序を変更 |

### 実践例

```python
# 基本テンソルの作成
x = torch.arange(1., 10.)
print(f"元のテンソル: {x}, 形状: {x.shape}")

# reshape: 新しい形状に変更
x_reshaped = x.reshape(3, 3)
print(f"リシェイプ後: {x_reshaped}, 形状: {x_reshaped.shape}")

# view: メモリを共有する形状変更
z = x.view(3, 3)
print(f"ビュー: {z}, 形状: {z.shape}")

# 重要：viewは元データを共有するため、変更が反映される
z[:, 0] = 5
print(f"zを変更後:")
print(f"z: {z}")
print(f"x: {x}")  # xも変更されている！
```

### スタック操作

```python
# テンソルを積み重ねる
x_stacked = torch.stack([x, x], dim=1)
print(f"元のテンソル: {x}")
print(f"スタック後: {x_stacked}, 形状: {x_stacked.shape}")
```

### 次元の追加・削除

```python
# squeeze: サイズ1の次元を削除
x = torch.zeros(2, 1, 2, 1, 2)
x_squeezed = torch.squeeze(x)
print(f"squeeze後: {x_squeezed}, 形状: {x_squeezed.shape}")

# unsqueeze: 次元を追加
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(f"unsqueeze後: {x_unsqueezed}, 形状: {x_unsqueezed.shape}")
```

### 次元の並び替え

```python
# permute: 次元の順序を変更（画像処理でよく使用）
# 画像形式: (高さ, 幅, チャンネル) → (チャンネル, 高さ, 幅)
x_original = torch.rand(size=(224, 224, 3))  # HWC形式
x_permuted = torch.permute(x_original, (2, 0, 1))  # CHW形式に変更

print(f"元の形状: {x_original.shape}")
print(f"変更後: {x_permuted.shape}")

# permuteもメモリを共有
x_permuted[0, 0, 0] = 999
print(f"共有確認: original={x_original[0, 0, 0]}, permuted={x_permuted[0, 0, 0]}")
```

---

## インデックス操作

### データの選択と抽出

```python
# 3次元テンソルの作成
x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"テンソル: {x}, 形状: {x.shape}")

# 各次元へのアクセス
print(f"x[0]: {x[0]}")          # 最初の次元
print(f"x[0][0]: {x[0][0]}")     # 2次元目まで
print(f"x[0][0][0]: {x[0][0][0]}") # 特定の要素

# コロン記法を使用した選択
print(f"すべての行の最初の列: {x[:, 0, :]}")
print(f"すべての列の最初の行: {x[:, :, 0]}")
print(f"特定位置の要素: {x[:, 0, 0]}")
```

---

## NumPyとの相互変換

### なぜNumPyとの互換性が重要か？

NumPyは科学計算分野で最も広く使用されているライブラリです。PyTorchはNumPyとの間でシームレスなデータ変換機能を提供しています。

### 変換方法

- **NumPy → PyTorch**: `torch.from_numpy(ndarray)`
- **PyTorch → NumPy**: `torch.tensor.numpy()`

```python
import numpy as np

# NumPy配列からPyTorchテンソルへ
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array).type(torch.float32)

print(f"NumPy配列のデータ型: {array.dtype}")
print(f"変換: array={array}, tensor={tensor}")

# メモリ共有の確認
array = array * 10
print(f"NumPy変更後: array={array}, tensor={tensor}")  # tensorは変わらない

# PyTorchテンソルからNumPy配列へ
tensor = torch.ones(8)
numpy_tensor = tensor.numpy()

print(f"PyTorchからNumPy: tensor={tensor}, numpy={numpy_tensor}")

# メモリ共有の確認
tensor = tensor * 10
print(f"PyTorch変更後: tensor={tensor}, numpy={numpy_tensor}")  # numpyは変わらない
```

---

## 再現性の確保

### なぜ再現性が重要なのか？

機械学習研究において、実験の再現性は極めて重要です。同じコードを実行して同じ結果を得られることで、以下が可能になります：

- **研究の検証**: 他の研究者が結果を確認できる
- **デバッグの効率化**: 一貫した条件での問題特定
- **公平な比較**: 異なるアルゴリズムを同じ条件で評価

### ランダムシードの設定

```python
# 通常のランダムテンソル（毎回異なる値）
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print("ランダムテンソルA:")
print(random_tensor_A)
print("ランダムテンソルB:")
print(random_tensor_B)
print(f"同じか？: {torch.equal(random_tensor_A, random_tensor_B)}")

# シードを設定した再現可能なランダムテンソル
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)  # 同じシードを再設定
random_tensor_D = torch.rand(3, 4)

print("\n再現可能なテンソルC:")
print(random_tensor_C)
print("再現可能なテンソルD:")
print(random_tensor_D)
print(f"同じか？: {torch.equal(random_tensor_C, random_tensor_D)}")
```

---

## GPU計算

### GPU計算の利点

- **並列処理**: 数千のコアで同時計算
- **高速化**: 特に行列演算で大幅な速度向上
- **スケーラビリティ**: 大規模なモデルやデータセットに対応

### デバイスの設定

#### CUDA（NVIDIA GPU）の場合
```python
# NVIDIA GPU使用時
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")
```

#### Apple Silicon（MPS）の場合
```python
# Apple Silicon Mac使用時
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用デバイス: {device}")
```

### テンソルのデバイス移動

```python
# CPUでテンソルを作成
tensor = torch.tensor([1, 2, 3])
print(f"CPUテンソル: {tensor}, デバイス: {tensor.device}")

# GPUに移動
tensor_on_gpu = tensor.to(device)
print(f"GPUテンソル: {tensor_on_gpu}")
```

### CPUへの復帰

```python
# 重要：NumPyはGPUテンソルを直接扱えない
# tensor_on_gpu.numpy()  # これはエラーになる！

# CPUに戻してからNumPy配列に変換
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(f"CPU復帰後: {tensor_back_on_cpu}")
```

---

## まとめ

このノートでは、PyTorchの基礎となるテンソル操作について学習しました：

1. **テンソルの基本概念**: スカラー、ベクトル、行列、多次元配列
2. **テンソルの作成方法**: ランダム、ゼロ、1、範囲指定
3. **データ型の理解**: 計算精度とパフォーマンスのトレードオフ
4. **形状操作**: reshape, view, stack, squeeze, unsqueeze, permute
5. **演算処理**: 基本算術から行列乗算まで
6. **実用的なツール**: NumPy変換、再現性、GPU活用

これらの知識は、より高度な機械学習モデルの構築における基盤となります。次のステップでは、これらのテンソル操作を使用してニューラルネットワークを構築していきます。

---

## 参考資料

- [PyTorch公式ドキュメント](https://docs.pytorch.org/)
- [PyTorchチュートリアル](https://pytorch.org/tutorials/)
- [行列乗算の視覚化](http://matrixmultiplication.xyz/)

---
---
title: "PyTorch環境構築ガイド（Apple Silicon Mac）"
description: "Apple Silicon MacでPyTorchとチュートリアル実行に必要な環境をセットアップする詳細ガイド。MPS対応とよくあるエラーの解決方法。"
date: "2025-07-02"
tags: ["環境構築", "PyTorch", "Python", "セットアップ", "初心者向け", "Apple Silicon", "MPS"]
---

# PyTorch環境構築ガイド（Apple Silicon Mac）

このガイドでは、Apple Silicon MacでPyTorchチュートリアルを実行するための環境構築方法を詳しく説明します。

## 必要なソフトウェア

### 基本要件
- **Python 3.9以上** （推奨: 3.11）
- **Conda または pip**
- **Git**（オプション：ソースコード管理用）

### GPU使用時の追加要件
- **Apple Silicon Mac**（M1、M2、M3チップ）
- **MPS（Metal Performance Shaders）**対応

## セットアップ手順

### 1. Condaのインストール
```bash
# Minicondaのダウンロード・インストール
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### 2. 仮想環境の作成
```bash
# 新しい環境を作成
conda create -n pytorch-tutorial python=3.11
conda activate pytorch-tutorial
```

### 3. PyTorch（MPS対応版）のインストール
```bash
# MPS（Metal Performance Shaders）対応PyTorch
conda install pytorch torchvision torchaudio -c pytorch
```

### 4. 追加ライブラリのインストール
```bash
# 機械学習・可視化ライブラリ
conda install matplotlib pandas numpy scikit-learn jupyter
conda install requests tqdm -c conda-forge
```

## Conda vs pip の使い分け

### Conda推奨ケース
- **GPU対応PyTorch**: MPSとの互換性が重要
- **科学計算ライブラリ**: NumPy、SciPy、matplotlibなど
- **バイナリ依存関係**: コンパイル済みパッケージで高速

### pip推奨ケース
- **純粋なPythonパッケージ**: requestsやtqdmなど
- **最新版が必要**: conda-forgeに無い場合
- **軽量インストール**: 最小限の依存関係

## インストール確認

### GPU使用可能性の確認
```python
import torch

# PyTorchのバージョン確認
print(f"PyTorch version: {torch.__version__}")

# MPS使用可能性の確認
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) available")
else:
    print("CPU only")

# 推奨デバイスの確認
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Recommended device: {device}")
```

### 基本ライブラリの確認
```python
# 必要なライブラリのインポートテスト
import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

print("All libraries imported successfully!")
```

## よくあるエラーと解決方法

### MPS関連エラー

#### エラー: "MPS backend doesn't support..."
```python
# 解決方法: CPUにフォールバック
device = "mps" if torch.backends.mps.is_available() else "cpu"
# 特定の操作でエラーが出る場合
tensor = tensor.to("cpu")  # 一時的にCPUで実行
```

#### エラー: "RuntimeError: MPS backend out of memory"
```python
# 解決方法1: バッチサイズを小さくする
BATCH_SIZE = 16  # 32から16に変更

# 解決方法2: MPSメモリをクリア
torch.mps.empty_cache()
```

### 共通エラー

#### ModuleNotFoundError
```bash
# 解決方法: 仮想環境がアクティブか確認
conda activate pytorch-tutorial

# パッケージの再インストール
conda install [package-name]
```

#### SSL証明書エラー
```bash
# 解決方法: conda設定を更新
conda config --set ssl_verify false
# または
pip install --trusted-host pypi.org --trusted-host pypi.python.org [package-name]
```

## パフォーマンス最適化のヒント

### MPS使用時
- **バッチサイズ**: メモリに応じて調整（16-64推奨）
- **データローダー**: `num_workers=2-4`で並列化（Apple Siliconでは控えめに）
- **メモリ管理**: `torch.mps.empty_cache()`でメモリクリア

### CPU使用時
- **スレッド数**: `torch.set_num_threads(4)`で調整
- **小さなモデル**: パラメータ数を抑制
- **データサイズ**: 画像サイズを小さく（32x32 → 64x64）

## 次のステップ

環境構築が完了したら：

1. **[PyTorch fundamentals](01_pytorch_fundamentals.md)** でテンソル操作を学習
2. **[PyTorch workflow](02_pytorch_workflow.md)** で機械学習の流れを理解
3. プロジェクトルートの`requirements.txt`で追加依存関係を確認

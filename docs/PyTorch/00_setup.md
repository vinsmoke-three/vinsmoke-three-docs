---
title: "PyTorch環境構築ガイド"
description: "PyTorchとチュートリアル実行に必要な環境をセットアップする詳細ガイド。Mac（MPS）、Windows（CUDA）、Linux別の手順とよくあるエラーの解決方法。"
date: "2025-07-02"
tags: ["環境構築", "PyTorch", "Python", "セットアップ", "初心者向け", "Mac", "Windows", "Linux"]
---

# PyTorch環境構築ガイド

このガイドでは、PyTorchチュートリアルを実行するための環境構築方法を詳しく説明します。

## 📋 必要なソフトウェア

### 基本要件
- **Python 3.9以上** （推奨: 3.11）
- **Conda または pip**
- **Git**（オプション：ソースコード管理用）

### GPU使用時の追加要件
- **NVIDIA GPU + CUDA**（Windows/Linux）
- **Apple Silicon Mac**（MPS対応）

## 🛠️ OS別セットアップ手順

### 🍎 macOS（Apple Silicon推奨）

#### 1. Condaのインストール
```bash
# Minicondaのダウンロード・インストール
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

#### 2. 仮想環境の作成
```bash
# 新しい環境を作成
conda create -n pytorch-tutorial python=3.11
conda activate pytorch-tutorial
```

#### 3. PyTorch（MPS対応版）のインストール
```bash
# MPS（Metal Performance Shaders）対応PyTorch
conda install pytorch torchvision torchaudio -c pytorch
```

#### 4. 追加ライブラリのインストール
```bash
# 機械学習・可視化ライブラリ
conda install matplotlib pandas numpy scikit-learn jupyter
conda install requests tqdm -c conda-forge
```

### 🪟 Windows（CUDA対応）

#### 1. CUDA Toolkitのインストール
1. [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)からCUDA Toolkit（11.8推奨）をダウンロード
2. インストーラーを実行

#### 2. Condaのインストール
```cmd
# Minicondaのダウンロード・インストール（公式サイトから）
# https://docs.conda.io/en/latest/miniconda.html
```

#### 3. 仮想環境の作成
```cmd
# 新しい環境を作成
conda create -n pytorch-tutorial python=3.11
conda activate pytorch-tutorial
```

#### 4. PyTorch（CUDA対応版）のインストール
```cmd
# CUDA 11.8対応PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 5. 追加ライブラリのインストール
```cmd
# 機械学習・可視化ライブラリ
conda install matplotlib pandas numpy scikit-learn jupyter
conda install requests tqdm -c conda-forge
```

### 🐧 Linux（CUDA対応）

#### 1. CUDA Toolkitのインストール
```bash
# Ubuntu/Debian系の場合
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

#### 2. Condaのインストール
```bash
# Minicondaのダウンロード・インストール
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### 3. 仮想環境の作成
```bash
# 新しい環境を作成
conda create -n pytorch-tutorial python=3.11
conda activate pytorch-tutorial
```

#### 4. PyTorch（CUDA対応版）のインストール
```bash
# CUDA 11.8対応PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 5. 追加ライブラリのインストール
```bash
# 機械学習・可視化ライブラリ
conda install matplotlib pandas numpy scikit-learn jupyter
conda install requests tqdm -c conda-forge
```

## 📦 Conda vs pip の使い分け

### Conda推奨ケース
- **GPU対応PyTorch**: CUDAやMPSとの互換性が重要
- **科学計算ライブラリ**: NumPy、SciPy、matplotlibなど
- **バイナリ依存関係**: コンパイル済みパッケージで高速

### pip推奨ケース
- **純粋なPythonパッケージ**: requestsやtqdmなど
- **最新版が必要**: conda-forgeに無い場合
- **軽量インストール**: 最小限の依存関係

## ✅ インストール確認

### GPU使用可能性の確認
```python
import torch

# PyTorchのバージョン確認
print(f"PyTorch version: {torch.__version__}")

# GPU使用可能性の確認
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) available")
else:
    print("CPU only")

# 推奨デバイスの確認
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
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

## 🚨 よくあるエラーと解決方法

### Mac（MPS関連）

#### エラー: "MPS backend doesn't support..."
```python
# 解決方法: CPUにフォールバック
device = "mps" if torch.backends.mps.is_available() else "cpu"
# 特定の操作でエラーが出る場合
tensor = tensor.to("cpu")  # 一時的にCPUで実行
```

### Windows（CUDA関連）

#### エラー: "CUDA out of memory"
```python
# 解決方法1: バッチサイズを小さくする
BATCH_SIZE = 16  # 32から16に変更

# 解決方法2: GPUメモリをクリア
torch.cuda.empty_cache()
```

#### エラー: "No CUDA capable device"
1. NVIDIAドライバーの最新版をインストール
2. CUDA Toolkitのバージョン確認
3. PyTorchのCUDA対応版を再インストール

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

## 💡 パフォーマンス最適化のヒント

### GPU使用時
- **バッチサイズ**: GPUメモリに応じて調整（16-128推奨）
- **データローダー**: `num_workers=4-8`で並列化
- **混合精度**: `torch.cuda.amp`で高速化

### CPU使用時
- **スレッド数**: `torch.set_num_threads(4)`で調整
- **小さなモデル**: パラメータ数を抑制
- **データサイズ**: 画像サイズを小さく（32x32 → 64x64）

# 制限ボルツマンマシンのPytorchによる実装

## Overview
Pytorchを使用して制限ボルツマンマシンを実装した

## Contents
- 高速化のためGPU対応にした
- 学習はPersistent Contrastive Divergence(PCD)法を使用した
- 学習の効率化のためMini-batch学習可能
- 擬似対数尤度を確認しながら学習可能

## Requirements
- PytorchのDataLoader作成の都合上、Mini-Batchサイズは割り切れるように設定

## Modules
- numpy
- pytorch
- time

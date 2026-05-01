#!/usr/bin/env bash
set -e

# H-Softmax on VoxCeleb1: c=5, s=30, m=0, lambda_1=0, lambda_2=1
CUDA_VISIBLE_DEVICES=7 python main.py \
  --dataset v1 \
  --lambda_1 0 \
  --lambda_2 1 \
  --h_C 5 \
  --h_s 30 \
  --h_m 0 \
  --batch_size 256 \
  --augment

# HAM-Softmax on VoxCeleb1: c=3, s=30, m=0.2, lambda_1=0, lambda_2=1
CUDA_VISIBLE_DEVICES=7 python main.py \
  --dataset v1 \
  --lambda_1 0 \
  --lambda_2 1 \
  --h_C 3 \
  --h_s 30 \
  --h_m 0.2 \
  --batch_size 256 \
  --augment

# E.H. AM-Softmax on VoxCeleb1: RAM loss, c=3, s=30, m=0.2, lambda_1=0.3, lambda_2=0.7
CUDA_VISIBLE_DEVICES=7 python main.py \
  --dataset v1 \
  --loss_type ram \
  --lambda_1 0.3 \
  --lambda_2 0.7 \
  --h_C 3 \
  --h_s 30 \
  --h_m 0.2 \
  --batch_size 256 \
  --augment

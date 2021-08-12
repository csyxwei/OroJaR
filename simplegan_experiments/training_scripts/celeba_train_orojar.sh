#!/usr/bin/env bash

python train.py \
      --dataset_mode celeba \
      --model gan128 \
      --nz 30 \
      --reg_lambda 1e-6 \
      --dataroot dataset/CelebA/img_align_celeba \
      --name celeba_orojar  \
      --save_latest_freq 10000 \
      --display_freq 20000 \
      --display_sample_freq 10000 \
      --print_freq 10000
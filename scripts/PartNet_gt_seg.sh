#!/usr/bin/env bash

suffix=gt_seg

python evaluation/eval_PartNet.py \
    -o /data5/wan/PartNet_${suffix} \
    --log /data5/wan/PartNet_${suffix}/log_$(date +%m_%d_%H:%M:%S).txt \
    --num-shapes 20 --gt-2d \
    --loss-weights=recon=1

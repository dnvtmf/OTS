#!/usr/bin/env bash

python evaluation/eval_PartNet.py \
    -o /data5/wan/PartNet_grid \
    --log /data5/wan/PartNet_grid/log_$(date +%m_%d_%H:%M:%S).txt \
    -ssl --stability_score_thresh=0.92 --points_per_batch=128 \
    --max-steps 0

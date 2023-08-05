#!/usr/bin/env bash

args=(--max-steps=0 --print-interval=10 -n=100)
gpus=(0 1 3 4 8 9)
in_thresholds=(0.7 0.75 0.8 0.85 0.9 0.95)

mkdir -p ./results/exps_in_threshold

for (( i = 0; i < ${#gpus[@]}; i++ ))
do
    echo $i, gpu: ${gpus[i]}, in_threshold: ${in_thresholds[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -ssl \
    --log ./results/exps_in_threshold/log_in=${in_thresholds[i]}_$(date +%m_%d_%H:%M:%S).txt \
    --stability_score_thresh=0.92 --points_per_batch=200 \
    ${args[*]} --in-threshold ${in_thresholds[i]} \
    ^M"

    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -sam \
    --log ./results/exps_in_threshold/log_sam_in=${in_thresholds[i]}_$(date +%m_%d_%H:%M:%S).txt \
    ${args[*]} --in-threshold ${in_thresholds[i]} \
    ^M"
done

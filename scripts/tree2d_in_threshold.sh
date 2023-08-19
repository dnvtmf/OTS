#!/usr/bin/env bash

args=(--max-steps=0 --print-interval=10 -n=100 -o ./results/SA_1B)
gpus=(1 2 3 4 5 6)
in_thresholds=(0.7 0.75 0.8 0.85 0.9 0.95)

for (( i = 0; i < ${#gpus[@]}; i++ ))
do
    echo gpu: ${gpus[i]}, in_threshold: ${in_thresholds[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -ssl \
    --log log_in=${in_thresholds[i]} \
    --stability_score_thresh=0.92 --points_per_batch=200 \
    ${args[*]} --in-threshold ${in_thresholds[i]} \
    ^M"
    
    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -sam \
    --log log_in=${in_thresholds[i]} \
    ${args[*]} --in-threshold ${in_thresholds[i]} \
    ^M"
done

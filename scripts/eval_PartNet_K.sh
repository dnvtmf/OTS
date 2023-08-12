#!/usr/bin/env bash

K=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
gpus=(0 1 2 3 4 5 6 7)
args=(-ns=100 --max-steps=10)

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} for K-ratio=${K[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet/ --log log_K_${K[i]}_$(date +%m_%d_%H:%M:%S).txt \
        -ssl --stability_score_thresh=0.92 --points_per_batch=64 \
        -s ./results/eval.txt \
        --K-ratio=${K[i]} --image-size=512 --filename=K=${K[i]} \
        ${args[*]} \
        ^M"
done
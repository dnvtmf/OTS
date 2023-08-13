#!/usr/bin/env bash

K=(0.5 1.0 5.0 10.0)
gpus=(0 0 0 0)
args=(  )

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} for K-ratio=${K[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet/ --log log_K_${K[i]}.txt \
        -ssl --stability_score_thresh=0.92 \
        --points_per_batch=256 --max-steps=10 --image-size=512 \
        -s ./results/eval.txt -ns=100 \
        --K-ratio=${K[i]} --filename=K=${K[i]} \
        ${args[*]} \
        ^M"
done
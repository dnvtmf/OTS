#!/usr/bin/env bash

loss=(tree view match t2d)
gpus=(0 1 2 3 4)
args=(--force-3d)

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} use loss ${loss[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet_final/ --log log_loss_${loss[i]} \
        -ssl --stability_score_thresh=0.92 \
        --points_per_batch=256 --max-steps=10 --image-size=512 \
        -s ./results/eval.txt -ns=100 \
        --loss-weights=${loss[i]}=0 --filename=loss=${loss[i]} \
        ${args[*]} \
        ^M"
done 
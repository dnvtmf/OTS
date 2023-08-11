#!/usr/bin/env bash

loss=(recon tree mv vm match)
gpus=(0 1 2 3 5)
args=(-ns=100 --max-steps=10)

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} use loss ${loss[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet/ --log log_loss_${loss[i]}_$(date +%m_%d_%H:%M:%S).txt \
        -ssl --stability_score_thresh=0.92 --points_per_batch=64 \
        -s ./results/eval.txt \
        --loss-weights=${loss[i]}=0 --image-size=512 --filename=loss=${loss[i]} \
        ${args[*]} \
        ^M"
done 
#!/usr/bin/env bash

ablations=("loss-weights=tree=0" "loss-weights=t2d=0" "loss-weights=match=0" "loss-weights=view=0,mv=0" "loss-weights=recon=0" "loss-weights=vm=0" "gnn=None" "gnn=GCN")
gpus=(0 1 2 3 4 5 6 7)
args=()

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} use ablations ${ablations[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet_final/ --log log_ablations${ablations[i]} \
        -ssl --stability_score_thresh=0.92 \
        --points_per_batch=256 --max-steps=10 --image-size=512 \
        -s ./results/eval.txt -ns=100 \
        --${ablations[i]} --filename=${ablations[i]} \
        ${args[*]} \
        ^M"
done 
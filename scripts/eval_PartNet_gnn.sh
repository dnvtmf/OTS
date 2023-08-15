#!/usr/bin/env bash

gnn_type=(None GCN GAT)
gpus=(0 1 2)
args=(-ns=100 --max-steps=10)

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} use GNN ${gnn_type[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python3 evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet_final/ --log log_gnn_${gnn_type[i]} \
        -ssl --stability_score_thresh=0.92 --points_per_batch=64 \
        -s ./results/eval.txt \
        --gnn=${gnn_type[i]} --image-size=512 --filename=gnn=${gnn_type[i]} \
        ${args[*]} \
        ^M"
done
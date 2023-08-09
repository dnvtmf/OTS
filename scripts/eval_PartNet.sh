#!/usr/bin/env bash

categories=(Bed Chair Clock Dishwasher Earphone Faucet Lamp Table)
gpus=(2 3 4 5 6 7 8 9)

for (( i=0; i < 8; ++i ))
do
    echo use gpu${gpus[i]} on category ${categories[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python evaluation/eval_PartNet.py  \
        -o /data5/wan/PartNet/${categories[i]} --log log_$(date +%m_%d_%H:%M:%S).txt \
        -ssl --stability_score_thresh=0.92 --points_per_batch=64 \
        -s ~/data/PartNet/tree_seg/${categories[i]} \
        --max-steps=0 \
        ^M"
done
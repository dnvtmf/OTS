#!/usr/bin/env bash

start_index=(0 125 250 375 500 625 750 875)
gpus=(0 1 2 3 4 5 6 7 )
args=(-ns=125 --max-steps=10 --gt-2d)

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} start at ${start_index[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet_final/ --log log_start_${start_index[i]} \
        -ssl --stability_score_thresh=0.92 --points_per_batch=64 \
        --split ./results/PartNet_test_split.txt \
        --start-index=${start_index[i]} --image-size=512 \
        ${args[*]} \
        ^M"
done
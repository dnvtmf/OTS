#!/usr/bin/env bash

num_views=(10 20 50 200)
gpus=(0 1 2 3)
args=( )

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} use num_views=${num_views[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet/num_views_${num_views[i]} --log log_num_views=${num_views[i]} \
        -ssl --stability_score_thresh=0.92 \
        --points_per_batch=256 --max-steps=10 --image-size=512 \
        -s ./results/eval.txt -ns=100 \
        --num-views=${num_views[i]} --filename=nv=${num_views[i]} \
        ${args[*]} \
        ^M"
done 
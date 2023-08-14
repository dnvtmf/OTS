#!/usr/bin/env bash


loss=( tree=0.0  tree=1.0  tree=0.3  tree=3.0 \
       view=0.0  view=1.0  view=0.3  view=3.0 \
      match=0.0 match=1.0 match=0.3 match=3.0 \
        t2d=0.0   t2d=1.0   t2d=0.3   t2d=3.0 )
gpus=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7)
args=(-ns=20 --force-3d)

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    echo use gpu${gpus[i]} use loss ${loss[i]}
    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet/ --log log_loss_${loss[i]} \
        -ssl --stability_score_thresh=0.92 \
        --points_per_batch=256 --max-steps=10 --image-size=512 \
        -s ./results/eval.txt  \
        --loss-weights=${loss[i]} --filename=loss=${loss[i]} \
        ${args[*]} \
        ^M"
done 
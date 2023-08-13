#!/usr/bin/env bash

loss=(recon=0 recon=0.3 recon=1.0 recon=3 \
       tree=0  tree=0.3  tree=1.0  tree=3 \
        mv=0     mv=0.3    mv=1      mv=3 \
        vm=0     vm=0.3    vm=1      vm=3 \
     match=0  match=0.3 match=1   match=3 \
     edge=0    edge=0.3  edge=1    edge=3 \
      t2d=0     t2d=0.3   t2d=1     t2d=3 \
       es=0      es=0.3    es=1      es=3 \
     tree=0,tree2=0 tree=0,tree2=0.3 tree=0,tree2=1 tree=0,tree2=3)
gpus=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3)
args=(-ns=20 )

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
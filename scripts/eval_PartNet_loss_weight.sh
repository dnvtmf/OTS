#!/usr/bin/env bash

loss=(recon=0.0 recon=0.3 recon=0.6 recon=0.9 \
       tree=1.0  tree=3.0  tree=6.0  tree=10. \
         mv=0.3    mv=0.6    mv=1.0    mv=1.5 \
         vm=1.0    vm=3.0    vm=6.0    vm=10. \
      match=0.1 match=0.3 match=0.6 match=1.0 \
       edge=0.0  edge=0.1  edge=0.2  edge=0.3 \
        t2d=1.0   t2d=3.0   t2d=6.0   t2d=10. \
     tree=0,tree2=0 tree=0,tree2=0.3 tree=0,tree2=1 tree=0,tree2=3)
gpus=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7)
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
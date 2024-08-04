#!/usr/bin/env bash

num_gpus=$(gpustat --no-header | wc -l)
echo "There are ${num_gpus} gpus"

for (( i = 0;  i < ${num_gpus}; ++i ))
do 
    if ! screen -ls gpu${i}
    then
        echo "create gpu${i}"
        screen -dmS gpu${i}
    fi
    screen -S gpu${i} -p 0 -X stuff "^M"
    screen -S gpu${i} -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=${i}^M"
    screen -S gpu${i} -p 0 -X stuff "cd ~/Projects/Segmentation/TreeSeg^M"
done 
screen -ls
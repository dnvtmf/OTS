#!/usr/bin/env bash

scenes=(office_0  office_1  office_2  office_3  office_4  room_0  room_1  room_2)
gpus=(0 1 2 3 0 1 2 3)
args=(-ssl --stability_score_thresh=0.92 --points_per_batch=256 )
output=./results/Replica

num_gpus=${#gpus[@]}
if [[ ${num_gpus} -ne ${#scenes[@]} ]]
then 
    echo "gpus ${gpus} not match experments: ${scenes}" 
    exit 1
fi

for (( i=0; i < ${num_gpus}; ++i ))
do
    echo "Run scene ${scenes[i]} on gpu ${gpus[i]}"
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python evaluation/eval_Replica.py -o ${output} --scene ${scenes[i]} ${args[*]} ^M"
done 
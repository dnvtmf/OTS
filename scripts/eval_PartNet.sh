#!/usr/bin/env bash
cd $(cd $(dirname $0); pwd)/..
pwd

gpus=(0 1 2 3 4 5 6 7 8 9)
args=(--max-steps=10 --filename=final)

num_gpus=${#gpus[@]}
echo "There are ${num_gpus} gpus and ${num_scenes} scenes"

for (( i = 0;  i < ${num_gpus}; ++i ))
do
    gpu_id="gpu${gpus[$i]}"
    if ! screen -ls ${gpu_id}
    then
        echo "create ${gpu_id}"
        screen -dmS ${gpu_id}
    fi
    screen -S ${gpu_id} -p 0 -X stuff "^M"
    screen -S ${gpu_id} -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=${gpus[$i]}^M"
    screen -S ${gpu_id} -p 0 -X stuff "cd $PWD^M"
done
screen -ls%

for (( i=0; i < ${#gpus[@]}; ++i ))
do
    start_index=$(( 1200 * i / ${num_gpus} ))
    num_run=$(( 1200 / ${num_gpus} ))
    echo "use gpu${gpus[i]} run [${start_index} $((${start_index} + ${num_run})))"
    screen -S gpu${gpus[i]} -p 0 -X stuff "^M"
    screen -S gpu${gpus[i]} -p 0 -X stuff \
        "python3 evaluation/eval_PartNet.py  \
        -o ./results/PartNet/  --log log_start_${start_index} \
        -sam --points_per_batch=64 \
        --split ./results/PartNet_list.txt \
        --start-index=${start_index} --num-shapes ${num_run} --image-size=512 \
        ${args[*]}  \
        ^M"
done
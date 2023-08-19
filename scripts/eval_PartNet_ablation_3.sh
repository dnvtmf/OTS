#!/usr/bin/env bash
set -e

cases=(10 20 50 200)
gpus=(0 1 2 3)
args=(--gt-2d)
prefix=3
num_cases=${#cases[@]}
num_gpus=${#gpus[@]}
echo "There are ${num_gpus} gpus and ${num_cases} cases, prefix=${prefix}"

for (( i=0; i < ${num_cases}; ++i ))
do
    gpu_id=${gpus[$(( $i % ${num_gpus} ))]}
    echo use gpu${gpu_id} run case: ${cases[i]}
    screen -S gpu${gpu_id} -p 0 -X stuff "^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
    "python3 evaluation/eval_PartNet.py  \
        -o ./results/cache/PartNet/num_views_${cases[i]} --log log_${prefix}_nv_${cases[i]} \
        -ssl --stability_score_thresh=0.92 \
        --points_per_batch=256 --max-steps=10 --image-size=512 \
        --split ./results/eval.txt -ns=100 \
        --num-views=${cases[i]} --filename=${prefix}_nv_${cases[i]} \
        ${args[*]} \
    ^M"
done
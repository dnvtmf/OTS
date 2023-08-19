#!/usr/bin/env bash
set -e

cases=("K=0.0" "K=0.5" "K=1.0" "K=1.5" "K=2.0" "K=2.5" "K=3.0" "K=5.0" "K=7.0" "K=10.0")
gpus=(0 1 2 3 4 5 6 7 8 9)
args=()
prefix=2
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
        -o ./results/cache/PartNet/ --log log_${prefix}_ablation_${cases[i]} \
        -ssl --stability_score_thresh=0.92 \
        --points_per_batch=256 --max-steps=10 --image-size=512 \
        --split ./results/eval.txt -ns=100 \
        --${cases[i]} --filename=${prefix}_${cases[i]} \
        ${args[*]} \
    ^M"
done
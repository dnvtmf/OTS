#!/usr/bin/env bash
set -e

cases=("loss-weights=tree=0" "loss-weights=t2d=0" "loss-weights=match=0" "loss-weights=view=0,mv=0" "loss-weights=recon=0" "loss-weights=vm=0" "gnn=None" "gnn=GCN" "gnn=GAT" "no-X")
gpus=(1 2 3 4 5)
args=()
prefix=1
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
        --${cases[i]} --filename=${cases[i]} \
        ${args[*]} \
    ^M"
done
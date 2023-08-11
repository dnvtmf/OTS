#!/usr/bin/env zsh

iou_thresholds=(0.9 0.85 0.8 0.75 0.7)
gpus=(7 6 5 4 3)
args=(--max-steps=100 --print-interval=10 -n=100 --data-root=~/data/SA_1B_test)
num_gpus=${#gpus[@]}
output='./results/SA_1B_SemanticSAM_L_pred_iou_thresh'

if [[ ${num_gpus} -ne ${#iou_thresholds[@]} ]]
then 
    echo "gpus ${gpus} not match experments: ${iou_thresholds}" 
    exit 1
fi

for (( i=0; i < ${num_gpus}; ++i))
do
    echo "screen gpu${gpus[i]} run experiments pred_iou_thresh=${iou_thresholds[i]}"
    screen -S gpu${gpus[i]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o ${output} --log log_$(date +%m_%d_%H:%M:%S).txt \
    --suffix=pred_iou=${iou_thresholds[i]} \
    -ssl --stability_score_thresh=0.92 --points_per_batch=256 ${args[*]} \
    --pred_iou_thresh=${iou_thresholds[i]} \
    ^M"
done 
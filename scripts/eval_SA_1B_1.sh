#!/usr/bin/env zsh

suffix="_1.0"
output='./results/SA_1B'
args=(--max-steps=100 --print-interval=10 -n=1000 --explore-ratio=1 --data-root=~/data/SA_1B_test)

screen -S gpu3 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o ${output} --log log_$(date +%m_%d_%H:%M:%S).txt \
    -ssl --suffix=SemanticSAM_L${suffix} \
    --stability_score_thresh=0.92 --points_per_batch=128 ${args[*]} \
    ^M"

screen -S gpu4 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py \
    -o ${output} --log log_$(date +%m_%d_%H:%M:%S).txt \
    -sst --suffix=SemanticSAM_T${suffix} \
    --stability_score_thresh=0.92 --points_per_batch=128 ${args[*]} \
    ^M"

screen -S gpu5 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o ${output}  --log log_$(date +%m_%d_%H:%M:%S).txt \
    -sam --suffix=SAM${suffix} \
    ${args[*]} \
    ^M"

screen -S gpu6 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o ${output}  --log log_$(date +%m_%d_%H:%M:%S).txt \
    -samL --suffix=SAM_L${suffix} \
    ${args[*]} \
    ^M"

screen -S gpu7 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o ${output}  --log log_$(date +%m_%d_%H:%M:%S).txt \
    -samB --suffix=SAM_B${suffix} \
    ${args[*]} \
    ^M"
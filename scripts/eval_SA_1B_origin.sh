#!/usr/bin/env zsh

suffix="_origin"
args=(--max-steps=0 --print-interval=10 -n=1000 --uncompress)

screen -S gpu1 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -ssl \
    --log /data5/wan/SA_1B_SemanticSAM_L${suffix}/log_$(date +%m_%d_%H:%M:%S).txt \
    --stability_score_thresh=0.92 --points_per_batch=128 \
    -o /data5/wan/SA_1B_SemanticSAM_L${suffix} ${args[*]} \
    ^M"

screen -S gpu2 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -sst \
    --log /data5/wan/SA_1B_SemanticSAM_T${suffix}/log_$(date +%m_%d_%H:%M:%S).txt \
    --stability_score_thresh=0.92 --points_per_batch=128 \
    -o /data5/wan/SA_1B_SemanticSAM_T${suffix}  ${args[*]} \
    ^M"

screen -S gpu3 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -sam \
    --log /data5/wan/SA_1B_SAM${suffix}/log_$(date +%m_%d_%H:%M:%S).txt \
    -o /data5/wan/SA_1B_SAM${suffix} ${args[*]} \
    ^M"

screen -S gpu4 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -samL \
    --log /data5/wan/SA_1B_SAM_L${suffix}/log_$(date +%m_%d_%H:%M:%S).txt \
    -o /data5/wan/SA_1B_SAM_L${suffix} ${args[*]} \
    ^M"

screen -S gpu5 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -samB \
    --log /data5/wan/SA_1B_SAM_B${suffix}/log_$(date +%m_%d_%H:%M:%S).txt \
    -o /data5/wan/SA_1B_SAM_B${suffix} ${args[*]} \
    ^M"
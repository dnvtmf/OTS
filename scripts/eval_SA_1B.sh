#!/usr/bin/env zsh

args=(--max-steps=100 --print-interval=10 -n=1000 --force)

screen -S gpu0 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -ssl \
    --log /data5/wan/SA_1B_SemanticSAM_L/log_$(date +%m_%d_%H:%M:%S).txt \
    --stability_score_thresh=0.92 --points_per_batch=200 \
    -o /data5/wan/SA_1B_SemanticSAM_L ${args[*]} \
    ^M"


screen -S gpu1 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -sst \
    --log /data5/wan/SA_1B_SemanticSAM_T/log_$(date +%m_%d_%H:%M:%S).txt \
    --stability_score_thresh=0.92 --points_per_batch=200 \
    -o /data5/wan/SA_1B_SemanticSAM_T  ${args[*]} \
    ^M"

screen -S gpu3 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py -sam \
    --log /data5/wan/SA_1B_SAM/log_$(date +%m_%d_%H:%M:%S).txt \
    -o /data5/wan/SA_1B_SAM ${args[*]} \
    ^M"
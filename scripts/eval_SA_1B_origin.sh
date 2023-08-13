#!/usr/bin/env zsh

suffix="_origin"
args=(--max-steps=0 --print-interval=10 -n=1000 --uncompress)

screen -S gpu1 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -ssl --suffix=SemanticSAM_L${suffix} \
    --stability_score_thresh=0.92 --points_per_batch=128 ${args[*]} \
    ^M"

screen -S gpu2 -p 0 -X stuff \
     "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -sst --suffix=SemanticSAM_T${suffix} \
    --stability_score_thresh=0.92 --points_per_batch=128 ${args[*]} \
    ^M"

screen -S gpu3 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -sam --suffix=SAM${suffix} \
    ${args[*]} \
    ^M"

screen -S gpu4 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -samL --suffix=SAM_L${suffix} \
    ${args[*]} \
    ^M"

screen -S gpu5 -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -samB --suffix=SAM_B${suffix} \
    ${args[*]} \
    ^M"
#!/usr/bin/env zsh

suffix="_0.5"
args=(--max-steps=100 --print-interval=10 -n=1000)
gpus=(0 1 2 3 4)

screen -S gpu${gpus[0]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -ssl --suffix=SemanticSAM_L${suffix} \
    --stability_score_thresh=0.92 --points_per_batch=128 ${args[*]} \
    ^M"

screen -S gpu${gpus[1]} -p 0 -X stuff \
     "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -sst --suffix=SemanticSAM_T${suffix} \
    --stability_score_thresh=0.92 --points_per_batch=128 ${args[*]} \
    ^M"

screen -S gpu${gpus[2]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -sam --suffix=SAM${suffix} \
    ${args[*]} \
    ^M"

screen -S gpu${gpus[3]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -samL --suffix=SAM_L${suffix} \
    ${args[*]} \
    ^M"

screen -S gpu${gpus[4]} -p 0 -X stuff \
    "python evaluation/eval_SA_1B.py  \
    -o /data5/wan/SA_1B --log log \
    -samB --suffix=SAM_B${suffix} \
    ${args[*]} \
    ^M"
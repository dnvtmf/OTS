#!/usr/bin/env bash
python evaluation/eval_SA_1B.py -n 1000 \
    --log /data5/wan/SA_1B_SemanticSAM_L/log.txt \
    -o /data5/wan/SA_1B_SemanticSAM_L --print-interval 10


python evaluation/eval_SA_1B.py -n 1000 \
    --log /data5/wan/SA_1B_SemanticSAM_T/log.txt \
    -o /data5/wan/SA_1B_SemanticSAM_T --print-interval 10 --sst

python evaluation/eval_SA_1B.py -n 1000 \
    --log /data5/wan/SA_1B_SAM/log.txt \
    -o /data5/wan/SA_1B_SAM --print-interval 10 -sam
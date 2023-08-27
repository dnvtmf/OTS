#!/usr/bin/env zsh
while [ 1 ]; do
#    vglrun -ld $(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib
    python ./debug_gui.py
done
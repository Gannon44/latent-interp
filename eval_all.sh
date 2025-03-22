#!/bin/bash
# eval_all.sh

# Model list
models=("transformer" "unet" "nn" "linear")

# GPU assignment: assign half of the models to GPU 0 and half to GPU 1 (adjust indices as desired)
for i in "${!models[@]}"; do
    model="${models[$i]}"
    if (( $i % 2 == 0 )); then
        gpu=3
    else
        gpu=2
    fi
    # Create a new tmux session for this model evalutaion
    tmux new-session -d -s "eval_${model}" "python train.py --model ${model} --mode eval --epochs 200 --learning_rate 1e-3 --batch_size 32 --gpu_index ${gpu} --data_dir /data/ggonsior/atd12k"
    echo "Started evalutaion for model ${model} on GPU ${gpu} in tmux session eval_${model}"
done

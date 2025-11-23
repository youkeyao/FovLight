#!/bin/bash
while true; do
    accelerate launch --num_processes=2 --gpu_ids=2,3 train_video.py
    if [ $? -ne 0 ]; then
        echo "restart..."
        sleep 3
    else
        break
    fi
done
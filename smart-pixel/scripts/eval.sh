#!/bin/bash

MODEL_INDEX=$1


if [ $MODEL_INDEX -eq 0 ]; then
    CONFIG=./configs/baseline_qkeras.yml
    PRETRAINED_MODEL=./dense_baseline_qkeras/qkeras_dense_model_58.h5
elif [ $MODEL_INDEX -eq 1 ]; then
    CONFIG=./configs/small_qkeras.yml
    PRETRAINED_MODEL=./dense_small_qkeras/qkeras_dense_model_16.h5
elif [ $MODEL_INDEX -eq 2 ]; then
    CONFIG=./configs/large_qkeras.yml
    PRETRAINED_MODEL=./dense_large_qkeras/qkeras_dense_model_512.h5
elif [ $MODEL_INDEX -eq 3 ]; then
    CONFIG=./configs/large2_qkeras.yml
    PRETRAINED_MODEL=./dense_large2_qkeras/qkeras_dense_model_58_58.h5
else
    echo "Error"
fi

CUDA_VISIBLE_DEVICES="" python eval.py -c $CONFIG --pretrained-model $PRETRAINED_MODEL

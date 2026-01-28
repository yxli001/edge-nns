#!/bin/bash

PATH_PREFIX=./configs

if [ $1 -eq 0 ]; then
    CONFIG=$PATH_PREFIX/baseline_qkeras.yml
elif [ $1 -eq 1 ]; then
    CONFIG=$PATH_PREFIX/small_qkeras.yml
elif [ $1 -eq 2 ]; then
    CONFIG=$PATH_PREFIX/large_qkeras.yml
else
    echo "Error"
fi


python3 train.py -c $CONFIG

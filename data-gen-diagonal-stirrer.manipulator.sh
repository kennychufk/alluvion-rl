#!/usr/bin/env bash

curve="val-bidir-circles2"
array=(4 8 22 66 100)

for i in "${array[@]}"
do
    python data-gen-diagonal-stirrer.py --output-dir=/media/kennychufk/mldata/alluvion-data/$curve --shape-dir=/home/kennychufk/workspace/pythonWs/shape-al --num-buoys $i
done

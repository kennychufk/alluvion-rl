#!/usr/bin/env bash

curve="parametric-star-epicycloid"
array=( 4 5 6 7 8 10 12 16 22 30 40 52 66 82 100)
for i in "${array[@]}"
do
    python data-gen-diagonal-stirrer.py --output-dir=/media/kennychufk/mldata/alluvion-data/$curve --shape-dir=/home/kennychufk/workspace/pythonWs/shape-al --num-buoys $i
done

#!/bin/bash

#list of layer 0 - 15
layer=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
object_of_study="mlp_act"
use_wandb=True

for i in ${layer[@]}
do
    echo "##############################"
    python3 clf.py --layer $i --object_of_study $object_of_study --use_wandb $use_wandb
    echo "##############################"
done
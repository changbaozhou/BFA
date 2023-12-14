#!/bin/bash

# 定义参数数组

model=$1


checkpoints=(
    pyramidnet110_cifar10_SGD
    pyramidnet110_cifar10_SAM_rho0.05
    pyramidnet110_cifar10_SAM_rho0.1
    pyramidnet110_cifar10_SAM_rho0.25
    pyramidnet110_cifar10_SAM_rho0.8
    pyramidnet110_cifar10_SAM_rho1.0
    )


for checkpoint in "${checkpoints[@]}"
do
    seeds=(42 0 1 2023 88)
    # 循环执行 a.sh 脚本
    for seed in "${seeds[@]}"
    do
        echo "Executing BFA_cifar10.sh $model on $checkpoint with parameter: $seed"
        bash BFA_cifar10.sh "$model" "$seed"  "$checkpoint"
    done
done

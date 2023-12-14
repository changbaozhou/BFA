#!/bin/bash

# 定义参数数组

model=$1
checkpoints=(
    vgg19_cifar10_SGD
    vgg19_cifar10_SAM_rho0.01
    vgg19_cifar10_SAM_rho0.05
    vgg19_cifar10_SAM_rho0.08
    vgg19_cifar10_SAM_rho0.1
    vgg19_cifar10_SAM_rho0.15
    vgg19_cifar10_SAM_rho0.2
    vgg19_cifar10_SAM_rho0.25
    vgg19_cifar10_SAM_rho0.5
    vgg19_cifar10_SAM_rho0.8
    vgg19_cifar10_SAM_rho1.0
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

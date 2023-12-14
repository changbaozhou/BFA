#!/usr/bin/env sh
bash train_cifar10_bat.sh SGD 0 >>  resnet20_cifar10_bat_SGD.log 2>&1 &
bash train_cifar10_bat.sh SAM 0.15 >>  resnet20_cifar10_bat_SAM_rho0.15.log 2>&1 &
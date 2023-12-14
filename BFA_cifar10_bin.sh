#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"aaa2")
    PYTHON='/home/bobzhou/miniconda3/envs/bobzhou/bin/python' # python environment path
    TENSORBOARD='/home/bobzhou/miniconda3/envs/bobzhou/bin/tensorboard' # tensorboard environment path
    data_path='/home/bobzhou/dataset' # dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
# model=resnet32_quan
# model=resnet20_quan
model=resnet20_bin
# model=resnet18_quan
# model=preactresnet18_quan
# model=wideresnet28_quan
# model=vgg19_bn_quan
# model=pyramidnet110_quan
dataset=cifar10
# test_batch_size=256
test_batch_size=128

attack_sample_size=128 # number of data used for BFA
# n_iter=20 # number of iteration to perform BFA
n_iter=1 # number of iteration to perform BFA
k_top=100 # only check k_top weights with top gradient ranking in each layer

# checkpoint=/home/bobzhou/SAM/checkpoint/PreActResNet18_cifar10_SGD.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/PreActResNet18_cifar10_SAM.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/ResNet20_cifar10_SGD.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/ResNet20_cifar10_SAM_rho0.05.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/ResNet20_cifar10_SAM_rho0.1.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/ResNet20_cifar10_SAM_rho0.25.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/ResNet20_cifar10_SAM_rho0.8.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/ResNet32_cifar10_SGD.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/ResNet32_cifar10_SAM.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/WideResNet28_cifar10_SAM.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/WideResNet28_cifar10_SGD.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/PyramidNet_cifar10_SAM.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/PyramidNet_cifar10_SGD.pth

# checkpoint_name=ResNet18_cifar10_SGD
# checkpoint_name=ResNet18_cifar10_SAM_rho0.05
# checkpoint_name=ResNet18_cifar10_SAM_rho0.1
# checkpoint_name=ResNet18_cifar10_SAM_rho0.25
# checkpoint_name=ResNet18_cifar10_SAM_rho0.8
# checkpoint_name=ResNet18_cifar10_SAM_rho1.0

# checkpoint_name=resNet18_cifar10_SGD
# checkpoint_name=resnet18_cifar10_SAM_rho0.05
# checkpoint_name=resnet18_cifar10_SAM_rho0.1
# checkpoint_name=resnet18_cifar10_SAM_rho0.25
# checkpoint_name=resnet18_cifar10_SAM_rho0.8
# checkpoint_name=resnet18_cifar10_SAM_rho1.0

# checkpoint_name=ResNet20_cifar10_SAM_rho0.8
# checkpoint_name=ResNet20_cifar10_SAM_rho0.05
# checkpoint_name=ResNet20_cifar10_SAM_rho0.25
# checkpoint_name=ResNet20_cifar10_SGD

# checkpoint_name=resnet20_cifar10_SAM_rho1.0
# checkpoint_name=resnet20_cifar10_SAM_rho0.05
# checkpoint_name=resnet20_cifar10_SAM_rho0.01 
# checkpoint_name=resnet20_cifar10_SAM_rho0.08
# checkpoint_name=resnet20_cifar10_SAM_rho0.1
# checkpoint_name=resnet20_cifar10_SAM_rho0.15
# checkpoint_name=resnet20_cifar10_SGD


# checkpoint_name=wideresnet28_cifar10_SGD
# checkpoint_name=wideresnet28_cifar10_SAM_rho0.05
# checkpoint_name=wideresnet28_cifar10_SAM_rho0.1
# checkpoint_name=wideresnet28_cifar10_SAM_rho0.25
# checkpoint_name=wideresnet28_cifar10_SAM_rho0.8
# checkpoint_name=wideresnet28_cifar10_SAM_rho1.0

# checkpoint_name=pyramidnet110_cifar10_SGD
# checkpoint_name=pyramidnet110_cifar10_SAM_rho0.05
# checkpoint_name=pyramidnet110_cifar10_SAM_rho0.1
# checkpoint_name=pyramidnet110_cifar10_SAM_rho0.25
# checkpoint_name=pyramidnet110_cifar10_SAM_rho0.8
# checkpoint_name=pyramidnet110_cifar10_SAM_rho1.0

# checkpoint_name=vgg19_cifar10_SGD
# checkpoint_name=vgg19_cifar10_SAM_rho0.05
checkpoint_name=resnet20_cifar10_bin

# checkpoint=/home/bobzhou/SAR/output/${checkpoint_name}.pth
# checkpoint=/home/bobzhou/SAM/checkpoint/${checkpoint_name}.pth


checkpoint=/home/bobzhou/BFA/checkpoint/resnet20_bin_cifar10_SAM_rho0.15.pth
# checkpoint=/home/bobzhou/BFA_new/save/2023-08-21/cifar10_resnet20_bin_160_SGD_binarized/model_best.pth.tar

save_path=./save/${DATE}/${dataset}_${model}
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path

############### Neural network ############################
{
$PYTHON main_BFA.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ${save_path}  \
    --test_batch_size ${test_batch_size} \
    --workers 8 \
    --ngpu 1 \
    --epochs 100 \
    --print_freq 50 \
    --reset_weight \
    --n_iter ${n_iter} \
    --k_top ${k_top} \
    --attack_sample_size ${attack_sample_size} \
    --resume ${checkpoint} \
    --bfa \
    --distributed \
    --wandb_name ${checkpoint_name} \
    --manualSeed 42 \
    --fine_tune \
    --bin
    # --evaluate \   
} 

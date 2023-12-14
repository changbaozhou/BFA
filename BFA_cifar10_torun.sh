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
model=resnet20_quan
# model=resnet20_bin
# model=resnet18_quan
# model=preactresnet18_quan
# model=wideresnet28_quan
# model=vgg19_bn_quan
# model=pyramidnet110_quan
dataset=cifar10

# test_batch_size=256
test_batch_size=128

attack_sample_size=128 # number of data used for BFA
n_iter=100 # number of iteration to perform BFA
k_top=10 # only check k_top weights with top gradient ranking in each layer

# seed=$2

# checkpoint_name=resnet20_cifar10_SAM_rho0.08
checkpoint_name=resnet20_cifar10_SGD


checkpoint=/home/bobzhou/SAR/output/${checkpoint_name}.pth


save_path=./save/${DATE}/${dataset}_${model}
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path

############### Neural network ############################
{
$PYTHON main_BFA.py --dataset ${dataset} \
    --data_path ${data_path} \
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
    --wandb_name ${checkpoint_name} \
    --manualSeed 42  \
    --distributed \
    --fine_tune \
    # --bin \
    # --evaluate \
    # --fine_tune \
} 

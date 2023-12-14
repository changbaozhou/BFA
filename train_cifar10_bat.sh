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
model=resnet20_bin
dataset=cifar10
test_batch_size=128
attack_sample_size=128 # number of data used for BFA
optimizer=$1
rho=$2
label_info=binarized
gpu_id=$3


save_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info}
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path
wandb_name=${model}_${dataset}_${label_info}

############### Neural network ############################
{
$PYTHON main_BAT.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ${save_path}  \
    --epochs 200 \
    --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120 \
    --gammas 0.1 0.1 \
    --attack_sample_size ${attack_sample_size} \
    --test_batch_size ${test_batch_size} \
    --workers 8 \
    --ngpu 1 \
    --gpu_id ${gpu_id} \
    --print_freq 50 \
    --distributed \
    --decay 0.0003 \
    --momentum 0.9 \
    --wandb_name ${wandb_name} \
    --manualSeed 42 \
    --model_only \
    --rho ${rho} \
    # --evaluate \   
} 

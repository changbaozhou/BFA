from __future__ import division
from __future__ import absolute_import

import os, sys, shutil, time, random
import torch

import torch.nn.functional as F

from models.quan_resnet_cifar import resnet20_quan

import torch.nn.functional as F
# from apex import amp
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from models.quan_resnet_cifar import resnet20_quan
from models.quantization import quan_Conv2d, quan_Linear, quantize


def load_model(checkpoint):
    model = resnet20_quan(10)
    checkpoint = torch.load(checkpoint)
    new_checkpoint = {} ## 新建一个字典来访问模型的权值
    for k,value in checkpoint.items():
        key = k.split('module.')[-1]
        new_checkpoint[key] = value
    checkpoint = new_checkpoint

    state_tmp = model.state_dict()
    state_tmp.update(checkpoint)
    model.load_state_dict(state_tmp)
    return model

def reset_step(model):
    for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                # simple step size update based on the pretrained model or weight init
                m.__reset_stepsize__()

def reset_weight(model):
    # block for weight reset
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            m.__reset_weight__()
            # print(m.weight)

def draw_weight(plt,layer,hist_color, line_color, label):
    
    # 获取层的权重
  
    weights = layer.weight.data.cpu()
    weights_mean = torch.mean(weights.abs())
    weights_std = torch.std(weights)

    # 将权重展平为一维数组
    weights_flattened = weights.numpy().flatten()

    print(f'weights_mean:{weights_mean}')
    print(f'weights_std:{weights_std}')


    # 绘制权重分布直方图（模型1）
    plt.hist(weights_flattened, bins=50, density=True, alpha=0.5, color=hist_color, label=label)

    # 计算拟合高斯分布的参数（模型1）
    mu, std = norm.fit(weights_flattened)
    x_range = np.linspace(min(weights_flattened), max(weights_flattened), 100)
    pdf = norm.pdf(x_range, mu, std)
    plt.plot(x_range, pdf, line_color,label=label)


def draw_grad(plt, layer, hist_color):
    plt.figure(figsize=(10, 6))
    grads = layer.weight.grad.cpu()
    grads_flattened = grads.numpy().flatten()
    plt.hist(grads_flattened, bins=50, density=True, alpha=0.5, color=hist_color)




def main():

    # Init model, criterion, and optimizer
    dir = '/home/bobzhou/SAR/output/'
    checkpoint_name_1 = 'resnet20_cifar10_SGD'
    checkpoint_name_2 = 'resnet20_cifar10_SAM_rho0.08'
    checkpoint_name_3 = 'resnet20_cifar10_SAM_rho0.01'
    # checkpoint_name_2 = 'resnet20_cifar10_SAM_rho0.15'
    # checkpoint_name_3 = 'resnet20_cifar10_SAM_rho0.25'
    checkpoint1 = dir + checkpoint_name_1 +'.pth'
    checkpoint2 = dir + checkpoint_name_2 +'.pth'
    checkpoint3 = dir + checkpoint_name_3 +'.pth'

    
    
    
    model1 = load_model(checkpoint1)
    reset_step(model1)
    reset_weight(model1)

    model2 = load_model(checkpoint2)
    reset_step(model2)
    reset_weight(model2)

    model3 = load_model(checkpoint3)
    reset_step(model3)
    reset_weight(model3)

    # 绘制权重分布和拟合的高斯分布曲线
    plt.figure(figsize=(10, 6))
    # draw_weight(plt, model1.conv_1_3x3,'r','b', checkpoint_name_1)
    # draw_weight(plt, model2.conv_1_3x3,'g','y', checkpoint_name_2)
    # draw_weight(plt, model3.conv_1_3x3,'m','p', checkpoint_name_3)


    plt.title('Distribution of First Convolutional Layer Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("weight distribution")



if __name__ == '__main__':
    main()




# import argparse
# import torch


# import os 



# import torch.nn.functional as F
# import logging
# from datetime import timedelta
# import datetime
# # from apex import amp
# from torch.nn.parallel import DistributedDataParallel as DDP
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import numpy as np
# from models.quan_resnet_cifar import resnet20_quan
# from models.quantization import quan_Conv2d, quan_Linear, quantize


# def load_model(checkpoint):
#     model = resnet20_quan(10)
#     print(model)
#     checkpoint = torch.load(checkpoint)
#     new_checkpoint = {} ## 新建一个字典来访问模型的权值
#     for k,value in checkpoint.items():
#         key = k.split('module.')[-1]
#         new_checkpoint[key] = value
#     checkpoint = new_checkpoint
#     model.load_state_dict(checkpoint)
#     return model

# def reset_step(model):
#     for m in model.modules():
#             if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
#                 # simple step size update based on the pretrained model or weight init
#                 m.__reset_stepsize__()

# def reset_weight(model):
#     # block for weight reset
#     for m in model.modules():
#         if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
#             m.__reset_weight__()
#             # print(m.weight)

# def main():

#     checkpoint1 = '/home/bobzhou/SAR/output/resnet20_cifar10_SGD.pth'
#     checkpoint2 = '/home/bobzhou/SAR/output/resnet20_cifar10_SAM_rho0.05.pth'



#     # Model & Tokenizer Setup
#     model1 = load_model(checkpoint1)
#     reset_step(model1)
#     reset_weight(model1)
#     # 获取层的权重
#     layer_weights1 = model1.conv_1_3x3.weight.data.cpu()

#     weights_mean1 = torch.mean(layer_weights1)
#     weights_std1 = torch.std(layer_weights1)

#     # 将权重展平为一维数组
#     weights_flattened1 = layer_weights1.numpy().flatten()

#     print(f'weights_mean1:{weights_mean1}')
#     print(f'weights_std1:{weights_std1}')

#     model2 = load_model(checkpoint2)
#     reset_step(model2)
#     reset_weight(model2)

#     layer_weights2 = model2.conv_1_3x3.weight.data.cpu()

#     weights_mean2 = torch.mean(layer_weights2)
#     weights_std2 = torch.std(layer_weights2)

#     # 将权重展平为一维数组
#     weights_flattened2 = layer_weights2.numpy().flatten()

#     print(f'weights_mean2:{weights_mean2}')
#     print(f'weights_std2:{weights_std2}')

#     # 绘制权重分布和拟合的高斯分布曲线
#     plt.figure(figsize=(10, 6))

#     # 绘制权重分布直方图（模型1）
#     plt.hist(weights_flattened1, bins=50, density=True, alpha=0.5, color='b', label='model1')

#     # 计算拟合高斯分布的参数（模型1）
#     mu1, std1 = norm.fit(weights_flattened1)
#     x_range1 = np.linspace(min(weights_flattened1), max(weights_flattened1), 100)
#     pdf1 = norm.pdf(x_range1, mu1, std1)
#     plt.plot(x_range1, pdf1, 'r')

#     # 绘制权重分布直方图（模型2）
#     plt.hist(weights_flattened2, bins=50, density=True, alpha=0.5, color='g', label='model2')

#     # 计算拟合高斯分布的参数（模型2）
#     mu2, std2 = norm.fit(weights_flattened2)
#     x_range2 = np.linspace(min(weights_flattened2), max(weights_flattened2), 100)
#     pdf2 = norm.pdf(x_range2, mu2, std2)
#     plt.plot(x_range2, pdf2, 'm')

#     plt.title('Distribution of First Convolutional Layer Weights')
#     plt.xlabel('Weight Value')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     plt.savefig("weight distribution")



#     # # 绘制权重分布的直方图
#     # plt.figure(figsize=(10, 6))
#     # plt.hist(weights_flattened, bins=50, color='blue', alpha=0.7)
#     # # 计算拟合高斯分布的参数
#     # mu, std = norm.fit(weights_flattened)
#     # # 生成一系列x值来绘制拟合的高斯分布曲线
#     # x_range = np.linspace(min(weights_flattened), max(weights_flattened), 100)
#     # pdf = norm.pdf(x_range, mu, std)
#     # plt.plot(x_range, pdf, 'r', label='Fitted Gaussian')
#     # plt.title('Distribution of First Convolutional Layer Weights')
#     # plt.xlabel('Weight Value')
#     # plt.ylabel('Frequency')
#     # plt.grid(True)
#     # plt.savefig("weight distribution")
#     # plt.show()

#     # Training
#     # train(args, model)


# if __name__ == "__main__":
#     main()

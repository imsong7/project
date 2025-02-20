# -*- coding: utf-8 -*-

"""
Created on Thu Oct 19 15:05:31 2023

@author: PC
"""
#%%
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchsummary import summary
from PIL import Image
import time
import copy
#
from timestamp import timestamp
#%%
# To Normalize the dataset, calculate the == and std
def get_mean_std(dataset):
    meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset] # 이렇게도 반복문을 쓸 수 있다니 가히 존경스럽다
    stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.std([m[0] for m in stdRGB])
    stdG = np.std([m[1] for m in stdRGB])
    stdB = np.std([m[2] for m in stdRGB])

    return [(meanR, meanG, meanB), (stdR, stdG, stdB)]
#%%
'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.5):
        super().__init__()

        # BatchNorm has bias included, so set conv2d bias=False
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Adding dropout
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # Identity mapping, input and output feature map size, filter number is the same.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # Projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.5):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Adding dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Adding dropout
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=2, init_weights=True, dropout_prob=0.5):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, dropout_prob)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, dropout_prob)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, dropout_prob)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, dropout_prob)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights initialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout_prob))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18(dropout_prob=0.5):
    return ResNet(BasicBlock, [2,2,2,2], dropout_prob=dropout_prob)

def resnet34(dropout_prob=0.5):
    return ResNet(BasicBlock, [3, 4, 6, 3], dropout_prob=dropout_prob)

def resnet50(dropout_prob=0.5):
    return ResNet(BottleNeck, [3,4,6,3], dropout_prob=dropout_prob)

def resnet101(dropout_prob=0.5):
    return ResNet(BottleNeck, [3, 4, 23, 3], dropout_prob=dropout_prob)

def resnet152(dropout_prob=0.5):
    return ResNet(BottleNeck, [3, 8, 36, 3], dropout_prob=dropout_prob)
'''
#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(r"C:\Users\PC\Desktop\ResNet\Python Code\models\231024_add_dropout_model_120_32.pt",map_location = device) # 불러올 모델 입력
print(model)
model.eval() # 평가 모드로 전환
#%%
# 1. 새로운 이미지를 로드하고 전처리
image_folder = r""  # 새로운 이미지 파일 경로

preprocess = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(get_mean_std()[0], get_mean_std()[1])])

# 폴더 내의 모든 이미지 예측
for filename in os.listdir(image_folder):
    if  filename.endswith(".png"):  # 이미지 파일 확장자에 따라 수정
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)

        # 이미지 전처리
        image = preprocess(image)
        image = image.unsqueeze(0)  # 배치 차원 추가

        # 모델 예측
        with torch.no_grad():
            output = model(image.to(device))

        # 모델의 예측 클래스 출력
        _, predicted_class = output.max(1)
        predicted_class = predicted_class.item()


        # 이미지 파일 이름과 예측한 클래스 출력
        print(f"Image: {filename}, Predicted class: {predicted_class}")
        
        #timestamp#
        if predicted_class == 1:
            start_time, end_time = timestamp(filename)
            print(f"Found at {start_time}sec - {end_time}")
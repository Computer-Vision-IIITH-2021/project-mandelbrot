#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 10:55:00 2021

@author: Charan
"""

from model import model_selector
from dataloader import create_dataloaders
from train import train
import torch
import torch.nn as nn

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

# def main(pathname, model_dict_path = "model.pth",clf = False):

train_pathname = "../train2017/"
val_pathname = "../val2017/"
weight_path = "./Model_Weights"
device = get_device()
loaders = create_dataloaders(train_pathname, val_pathname)
model = model_selector(weight_path, layer = 5)
print ("---Loaded data and model---")
# if(torch.cuda.device_count() > 1):
#     model = nn.DataParallel(model)

model = model.to(device)

train(loaders, model)

# main(data_dir, clf = True)





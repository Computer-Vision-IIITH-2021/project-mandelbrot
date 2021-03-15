import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models

import numpy as np

import time
import os
import copy

class vgg_1_decoder(nn.Module):
    
    def __init__(self):
        decoder_1 = nn.Sequential( # Sequential,
	                            nn.ReflectionPad2d((1, 1, 1, 1)),
	                            nn.Conv2d(64,3,(3, 3)),
                                )
    
    def forward(x):
        x = decoder_1(x)
        return x

class vgg_2_decoder(nn.Module):

    def __init__(self):
        decoder_2 = nn.Sequential( # Sequential,
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(128,64,(3, 3)),
                                nn.ReLU(),
                                nn.UpsamplingNearest2d(scale_factor=2),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(64,64,(3, 3)),
                                nn.ReLU(),
                                nn.ReflectionPad2d((1, 1, 1, 1)),
                                nn.Conv2d(64,3,(3, 3)),
                            )
    
    def forward(x):
        x = decoder_2(x)
        return x

class vgg_3_decoder(nn.Module):

    def __init__(self):
        decoder_3 = nn.Sequential( # Sequential,
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,128,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,64,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,64,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,3,(3, 3)),
                )
    
    def forward(x):
        x = decoder_3(x)
        return x

class vgg_4_decoder(nn.Module):

    def __init__(self):
        decoder_4 = nn.Sequential( # Sequential,
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,256,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,128,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,64,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,64,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,3,(3, 3)),
                )
    
    def forward(x):
        x = decoder_4(x)
        return x

class vgg_5_decoder(nn.Module):

    def __init__(self):
        decoder_5 = nn.Sequential( # Sequential,
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,256,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,128,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,64,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,64,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,3,(3, 3)),
                )
    
    def forward(x):
        x = decoder_5(x)
        return x

class model_selector(nn.Module):
    def __init__(self, weights_path,layer = 5, pretrained = True, train_decoder = False):
        super(model_selector, self).__init__()
        self.num_layer = layer
        vgg19 = models.vgg19(pretrained=pretrained)

        features = list(vgg19.features)
        
        if(self.num_layer == 1):
            self.encoder = nn.Sequential(*features[:4])
            self.decoder = vgg_1_decoder()
        elif(self.num_layer == 2):
            self.encoder = nn.ModuleList(features[:9])
            self.decoder = vgg_2_decoder()
        elif(self.num_layer == 3):
            self.encoder = nn.ModuleList(features[:18])
            self.decoder = vgg_3_decoder()
        elif(self.num_layer == 4):
            self.encoder = nn.ModuleList(features[:27])
            self.decoder = vgg_4_decoder()
        elif(self.num_layer == 5):
            self.encoder = nn.Sequential(*features[:36])
            self.decoder = nn.Sequential( # Sequential,
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,512,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(512,256,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,256,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256,128,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128,64,(3, 3)),
                    nn.ReLU(),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,64,(3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64,3,(3, 3)),
                )
    
        if(train_decoder):
            for param in encoder.parameters():
                param.requires_grad = False
        
        if(not train_decoder):
            model_dict_path = os.path.join(weights_path,'feature_invertor_conv' + str(self.num_layer) + '_1.pth')
            self.decoder.load_state_dict(torch.load(model_dict_path))

    def forward(self,x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out




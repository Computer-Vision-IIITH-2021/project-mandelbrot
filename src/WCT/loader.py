import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import os
import torch.nn as nn
import math
from PIL import Image
from torch.autograd import Variable


def is_image_file(filename):
    ans=False
    for extension in [".png", ".jpg", ".jpeg"]:
        if filename.endswith(extension):
            ans=True
            break
    return ans

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize,do_patches = False, patch_kernel_size = None):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in os.listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.fineSize = fineSize
        self.do_patches = do_patches
        self.kernel_size = patch_kernel_size
        if(self.do_patches and self.kernel_size != None):
            self.fineSize = self.kernel_size

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = Image.open(contentImgPath).convert('RGB')
        styleImg = Image.open(styleImgPath).convert('RGB')
        trans=transforms.Compose([transforms.ToTensor()])
        # resize
        if(self.fineSize != 0 and not self.do_patches):
            w,h = contentImg.size
            neww=None
            newh=None
            if(h>w):
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = math.floor(w*newh/h)  
            else:
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = math.floor(h*neww/w)
            contentImg = contentImg.resize((neww,newh))
            styleImg = styleImg.resize((neww,newh))
        elif(self.fineSize != 0  and self.do_patches):
            w,h = styleImg.size
            neww=None
            newh=None
            if(h>w):
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = math.floor(w*newh/h)  
            else:
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = math.floor(h*neww/w)
            styleImg = styleImg.resize((neww,newh))

        # Preprocess Images
        contentImg = trans(contentImg)
        styleImg = trans(styleImg)
        return contentImg.squeeze(0),styleImg.squeeze(0),self.image_list[index]

    def __len__(self):
        return len(self.image_list)

import torch.nn as nn
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
img1 = scio.loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\Hermiston\hermiston2004.mat')['HypeRvieW']
img2 = scio.loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\Hermiston\hermiston2007.mat')['HypeRvieW']
# img1 = scio.loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\Hermiston\hermiston2004.mat')['HypeRvieW']
# img2 = scio.loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\Hermiston\hermiston2007.mat')['HypeRvieW']
img1 = torch.from_numpy(img1).to(torch.float32)
img2 = torch.from_numpy(img2).to(torch.float32)
conv3d = nn.Conv3d(in_channels=1,out_channels=10,kernel_size=(5,1,1),padding=(2,0,0))
img1 = img1.permute(2,0,1)
img2 = img2.permute(2,0,1)
# aaa = img1.view(1,1,242,390,200)
# bbb =conv3d(aaa)
mean1 = torch.mean(img1,dim=0)
mean2 = torch.mean(img2,dim=0)
std1 = torch.std(img1,dim=0)
std2= torch.std(img2,dim=0)
max1 = torch.max(img1,dim=0)
max2 = torch.max(img2,dim=0)
mim1 = torch.min(img1,dim=0)
mim2 = torch.min(img2,dim=0)
print(111)
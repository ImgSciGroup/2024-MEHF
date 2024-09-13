import os
import time

import cv2
import scipy.io as scio
import numpy as np
import torch.nn as nn
import torch
import scipy.io as io
import torch.optim as optim
import config as cfg
import get_train_dataset
import torchvision
# from My_module import mymodule as module
from My_module import Mutilscale as module
from HSI_data import HSI_data
from train import train, test, accuracy_assessment
import scipy.io as io
import matplotlib.pyplot as plt


def Predict_Label2Img(predict_label, img_gt):
    predict_img = torch.zeros_like(img_gt)
    predict_img[:, :] = 3
    num = predict_label.shape[0]  # 111583

    for i in range(num):
        x = int(predict_label[i][1])
        y = int(predict_label[i][2])
        l = predict_label[i][3]
        predict_img[x][y] = l
    # predict_img[predict_img==1]=255
    # predict_img[predict_img==3]=125
    return predict_img


def find_high_confidence(CMI_label, sort=0.005):
    return CMI_label[np.argsort(-CMI_label[:, 3])[0:int(len(CMI_label) * sort)]], CMI_label[
        np.argsort(CMI_label[:, 3])[0:int(len(CMI_label) * sort)]],


def union(old_indice, presu_indice):
    u = np.intersect1d(old_indice[:, 0], presu_indice[:, 0])
    new_indice = old_indice.copy()
    for i in range(len(presu_indice)):
        if (presu_indice[i, 0] not in old_indice[:, 0]):
            new_indice = np.append(new_indice, [presu_indice[i, :]], 0)
    return torch.tensor(new_indice)


current_dataset = cfg.current_dataset
current_model = cfg.current_model
model_name = current_dataset + current_model

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
pad = nn.ReplicationPad2d(1)
img2 = torch.rand(1, 1, 3, 3)  # np.random.rand(3,3)

# print(img2)
# #
# con1 = nn.ConvTranspose2d(1,1,kernel_size=3)
# img3 = con1(img2)
# print(img3)
# print(img2,img2.shape)
# print(pad(img2))
# print(pad(img2).squeeze(0))
fake = 0.005
print(fake)
data_sets = get_train_dataset.get_train_test_set(cfg.data)
data_sets1 = get_train_dataset.get_train_test_set(cfg.data)
img_gt = data_sets['img_gt']
train_data = HSI_data(data_sets, 'train')
test_data = HSI_data(data_sets, 'test')

model = module(cfg.model['in_fea_num']).to(device)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-3)

for i in range(1):
    print('stage:', i)

    train(train_data, model, loss_fun, optimizer, device, cfg.train['train_model'])
    pred_train_label, pred_train_acc, CMI1_train, CMI2_train = test(train_data, data_sets['img_gt'], model, device,
                                                                    cfg.test)
    pred_test_label, pred_test_acc, CMI1_test, CMI2_test = test(test_data, data_sets['img_gt'], model, device, cfg.test)
    predict_label = torch.cat([pred_train_label, pred_test_label], dim=0)
    # CMI1_label = torch.cat([CMI1_train, CMI1_test], dim=0)  # ch
    # CMI2_label = torch.cat([CMI2_train, CMI2_test], dim=0)  # un
    # CMI_final = CMI2_label
    #
    # # scio.savemat(r'./oa/canshu/wetsave_cmi'+'{},sample={}'.format(i,0.0033)+'.mat',{"img":np.array(Predict_Label2Img(CMI_final,img_gt))})
    #
    # CMI_final[:, 3] = CMI2_label[:, 3] - CMI1_label[:, 3]
    # change_add_sample, unchange_add_sample = find_high_confidence(CMI_fi nal, fake)
    # change_add_sample[:, 3] = 1
    # # change_add_sample[:,3]=1
    # # unchange_add_sample = find_high_confidence(CMI2_label)
    # unchange_add_sample[:, 3] = 0
    # # unchange_add_sample[:,3]=0
    # new_trainsample_1 = union(np.array(train_data.train_label), np.array(change_add_sample))
    # new_trainsample_2 = union(np.array(new_trainsample_1), np.array(unchange_add_sample))
    # random_index = torch.randperm(len(new_trainsample_2))
    # data_sets['init_sample_center'] = new_trainsample_2[random_index,]
    # train_data = HSI_data(data_sets, 'train')
    # test_data = HSI_data(data_sets, 'test')0
    predict_img = Predict_Label2Img(predict_label, img_gt)
    conf_mat, oa, kappa_co, P, R, F1, acc = accuracy_assessment(img_gt, predict_img)
    predict_img = predict_img.numpy()
    cv2.imwrite("Farmland_No_DF_AF.bmp", predict_img * 255)

    print(oa, kappa_co, P, R, F1, acc)
    # file = open(r'oa\canshu' + 'sample{},{}'.format(0.0033,i) + '.txt', 'w')
    # file.write('{},{},{},{},{},{}'.format(oa, kappa_co, P, R, F1, acc))
    # file.close()
    print("time:", time.localtime())

assessment_result = [round(oa, 4) * 100, round(kappa_co, 4), round(F1, 4) * 100, round(P, 4) * 100,
                     round(R, 4) * 100, model_name]

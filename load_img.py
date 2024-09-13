from scipy.io import loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 用来获取数据集和真值图
def get_dataset():
    # data_set_before = loadmat(r'D:\img\ChangeImage\River\mat_file\river_before.mat')['river_before'] #hermiston
    # data_set_after = loadmat(r'D:\img\ChangeImage\River\mat_file\river_after.mat')['river_after']
    # ground_truth = cv2.imread(r'D:\img\ChangeImage\River\River_gt.bmp', 0)
    # data_set_before = loadmat(r'D:\img\ChangeImage\Wetland\mat_file\farm06.mat')['imgh'] #hermiston
    # data_set_after = loadmat(r'D:\img\ChangeImage\Wetland\mat_file\farm07.mat')['imghl']
    # ground_truth =loadmat (r'D:\img\ChangeImage\Wetland\mat_file\label.mat')['label']
    # data_set_before = loadmat(r'D:\img\ChangeImage\HermistonCity\mat_file\hermiston2004.mat')['HypeRvieW']  # hermiston
    # data_set_after = loadmat(r'D:\img\ChangeImage\HermistonCity\mat_file\hermiston2007.mat')['HypeRvieW']
    # ground_truth = cv2.imread(r'D:\img\ChangeImage\HermistonCity\Hermiston_GT.bmp', 0)
    # ground_truth = loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\USA\SaGT.mat')['GT']
    # data_set_before = loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\USA\Sa1.mat')['T1'] #hermiston
    # data_set_after = loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\USA\Sa2.mat')['T2']
    # ground_truth = loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\USA\SaGT.mat')['GT']
    # ground_truth[ground_truth==1]=0
    # data_set_before = loadmat(r'D:\img\ChangeImage\USA\mat\USA_T1.mat')['T1']  # hermiston
    # data_set_after = loadmat(r'D:\img\ChangeImage\USA\mat\USA_T2.mat')['T2']
    # ground_truth = loadmat(r'D:\img\ChangeImage\USA\mat\USA_Binary.mat')['Binary']
    # data_set_before = loadmat(r'D:\RS_dataset\The River Data Set\river_before.mat')['river_before']
    # data_set_after = loadmat(r'D:\RS_dataset\The River Data Set\river_after.mat')['river_after']
    # ground_truth = cv2.imread(r'D:\text_all\GroundTrueth\gt_river.bmp', 0)

    data_set_before = loadmat(r'D:\img\ChangeImage\BayArea\mat\Bay_Area_2013.mat')['HypeRvieW'] #Bay
    data_set_after = loadmat(r'D:\img\ChangeImage\BayArea\mat\Bay_Area_2015.mat')['HypeRvieW']
    ground_truth =loadmat(r'D:\img\ChangeImage\BayArea\mat\bayArea_gtChangesolf.mat')['HypeRvieW']

    # data_set_before = loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\santaBarbara\mat\barbara_2013.mat')['HypeRvieW'] #Bay
    # data_set_after = loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\santaBarbara\mat\barbara_2014.mat')['HypeRvieW']
    # ground_truth =loadmat(r'D:\RS_dataset\ChangeDetectionDataset-master\santaBarbara\mat\barbara_gtChanges.mat')['HypeRvieW']
    # bay and santa dataset unique
    ground_truth[ground_truth==0]=3
    ground_truth[ground_truth == 1] = 1
    ground_truth[ground_truth == 2] = 0

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # for i in range(data_set_before.shape[0]):
    #     data_set_before[i, :, :] = scaler.fit_transform(data_set_before[i, :, :])
    # #     data_set_after[i, :, :] = scaler.fit_transform(data_set_after[i, :, :])
    # ground_truth = np.array(ground_truth).flatten()
    # for i in range(len(ground_truth)):
    #     if ground_truth[i] == 0:
    #         ground_truth[i] = 0
    #     elif ground_truth[i] != 0:
    #         ground_truth[i] = 1
    # ground_truth = ground_truth.reshape((data_set_before.shape[0], data_set_before.shape[1]))
    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt

import torch
from load_img import get_dataset
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt


def get_train_test_set(cfg):
    # current_dataset = cfg['current_dataset']
    train_set_num = cfg['train_set_num']
    patch_size = cfg['patch_size']
    img1, img2, gt = get_dataset()
    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    gt = torch.from_numpy(gt)
    img1 = img1.permute(2, 0, 1)  # channel放到第一维CxHxW
    img2 = img2.permute(2, 0, 1)
    img_gt = gt

    img1_pad, img2_pad, patch_coordinates = construct_sample(img1, img2, patch_size)
    data_sample = select_sample(img_gt, train_set_num)

    data_sample['img1_pad'] = img1_pad
    data_sample['img2_pad'] = img2_pad

    data_sample['patch_coordinates'] = patch_coordinates
    data_sample['img_gt'] = img_gt  #
    data_sample['ori_gt'] = gt

    return data_sample


def construct_sample(img1, img2, windowsize=5):
    _, height, width = img1.shape
    #padding
    pad = nn.ReplicationPad2d(int(windowsize // 2))
    pad_img1 = pad(img1.unsqueeze(0)).squeeze(0)
    pad_img2 = pad(img2.unsqueeze(0)).squeeze(0)

    # 坐标
    coordinates = torch.zeros((height * width, 4), dtype=torch.long)
    t = 0
    for h in range(height):
        for w in range(width):
            coordinates[t, :] = torch.tensor([h, h + windowsize, w, w + windowsize])
            t += 1
    #print("coordinates")
    #print(coordinates,coordinates.shape)
    return pad_img1, pad_img2, coordinates


def get_random_index(gt, ntr, class_num, indices_vector):
    if ntr < 1:
        ntr0 = int(ntr * class_num)

    else:
        ntr0 = ntr

    if ntr0 < 10:
        select_num = 10

    elif ntr0 > class_num // 2:
        select_num = class_num // 2

    else:
        select_num = ntr0
    select_num = torch.tensor(select_num)
    rand_indices0 = torch.randperm(class_num)
    rand_indices = indices_vector[rand_indices0]
    return select_num, rand_indices0


def index_generate(ntr, all_num):
    train_num = ntr * all_num
    test_num = all_num - train_num


def my_selectsample(text_index, train_index):
    return 1


def select_sample(gt, ntr):  # input tensor NxCxHxW, tensor gt HxW
    gt_vector = gt.reshape(-1, 1).squeeze(1)
    label = torch.unique(gt)
    first_time = True
    for each in range(len(label) - 1):  #bay
        indices_vector = torch.where(gt_vector == label[each])
        indices = torch.where(gt == label[each])
        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]
        class_num = torch.tensor(len(indices_vector))
        if ntr < 1:
            ntr0 = int(ntr * class_num)

        else:
            ntr0 = ntr

        if ntr0 < 10:
            select_num = 10

        elif ntr0 > class_num // 2:
            select_num = class_num // 2

        else:
            select_num = ntr0
        select_num = torch.tensor(select_num)
        rand_indices0 = torch.randperm(class_num)
        rand_indices = indices_vector[rand_indices0]

        tr_ind0 = rand_indices0[0:select_num]  #train_index_class
        te_ind0 = rand_indices0[select_num:]  #test_index_class
        tr_ind = rand_indices[0:select_num]
        te_ind = rand_indices[select_num:]
        select_train_gt_ind = torch.cat([tr_ind.unsqueeze(1),
                                         indices_row[tr_ind0].unsqueeze(1),
                                         indices_column[tr_ind0].unsqueeze(1),
                                         gt_vector[tr_ind].unsqueeze(1)],
                                        dim=1)
        select_tr_ind = torch.cat([tr_ind.unsqueeze(1),
                                   indices_row[tr_ind0].unsqueeze(1),
                                   indices_column[tr_ind0].unsqueeze(1)],
                                  dim=1
                                  )
        select_te_ind = torch.cat([te_ind.unsqueeze(1),
                                   indices_row[te_ind0].unsqueeze(1),
                                   indices_column[te_ind0].unsqueeze(1)],
                                  dim=1
                                  )
        if first_time:
            first_time = False
            init_sample_gt = select_train_gt_ind
            train_sample_center = select_tr_ind
            train_sample_num = select_num.unsqueeze(0)
            test_sample_center = select_te_ind
            test_sample_num = (class_num - select_num).unsqueeze(0)

        else:
            init_sample_gt = torch.cat([init_sample_gt, select_train_gt_ind], dim=0)
            train_sample_center = torch.cat([train_sample_center, select_tr_ind], dim=0)
            train_sample_num = torch.cat([train_sample_num, select_num.unsqueeze(0)])

            test_sample_center = torch.cat([test_sample_center, select_te_ind], dim=0)
            test_sample_num = torch.cat([test_sample_num, (class_num - select_num).unsqueeze(0)])
    rand_tr_ind = torch.randperm(train_sample_num.sum())
    init_sample_gt = init_sample_gt[rand_tr_ind]
    train_sample_center = train_sample_center[rand_tr_ind,]
    rand_te_ind = torch.randperm(test_sample_num.sum())
    test_sample_center = test_sample_center[rand_te_ind,]
    data_sample = {'train_sample_center': train_sample_center, 'train_sample_num': train_sample_num,
                   'test_sample_center': test_sample_center, 'test_sample_num': test_sample_num,
                   'init_sample_center': init_sample_gt
                   }
    return data_sample

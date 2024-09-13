import torch
import time
import datetime
import math
import os

from torch.utils.data import DataLoader


def adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, step_index):
    # Adjust the learning rate in stages
    if epoch < 1:
        lr = 0.0001 * lr_init
    elif epoch <= step_index[0]:
        lr = lr_init
    elif epoch <= step_index[1]:
        lr = lr_init * lr_gamma
    elif epoch > step_index[1]:
        lr = lr_init * lr_gamma ** 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(train_data, model, loss_fun, optimizer, device, cfg):
    torch.autograd.set_detect_anomaly(True)
    num_workers = 12
    gpu_num = 1

    save_folder = cfg['save_folder']
    save_name = cfg['save_name']

    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']

    epoch_size = cfg['epoch']
    batch_size = cfg['batch_size']

    # gpu_num
    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    '''# Load the model and start training'''
    model.train()

    if cfg['reuse_model']:
        print('load model...')
        checkpoint = torch.load(cfg['reuse_file'], map_location=device)
        start_epoch = checkpoint['epoch']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        start_epoch = 0

    batch_num = math.ceil(len(train_data) / batch_size)
    train_loss_save = []
    train_acc_save = []

    print('training...')

    for epoch in range(start_epoch + 1, epoch_size + 1):

        epoch_time0 = time.time()
        epoch_loss = 0
        predict_correct = 0
        label_num = 0

        batch_data = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True,drop_last=True)
        if lr_adjust:
            lr = adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, lr_step)
        else:
            lr = lr_init

        for batch_idx, batch_sample in enumerate(batch_data):
            iteration = (epoch - 1) * batch_num + batch_idx + 1
            batch_time0 = time.time()
            img1, img2, target, indices = batch_sample
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = target.to(device)

            prediction = model(img1, img2)
            loss = loss_fun(prediction, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time1 = time.time()
            batch_time = batch_time1 - batch_time0
            batch_eta = batch_time * (batch_num - batch_idx)
            epoch_eta = int(batch_time * (epoch_size - epoch) * batch_num + batch_eta)

            epoch_loss += loss.item()
            predict_label = prediction.detach().argmax(dim=1, keepdim=True)

            predict_correct += predict_label.eq(target.view_as(predict_label)).sum().item()
            label_num += len(target)

        train_acc = 100 * predict_correct/label_num
        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        print('Epoch: {}/{} || lr: {} || loss: {} || Train acc: {:.2f}% || '
              'Epoch time: {:.4f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss/batch_num, train_acc,
                      epoch_time, str(datetime.timedelta(seconds=epoch_eta))
                      )
              )

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        train_loss_save.append(epoch_loss / batch_num)
        train_acc_save.append(train_acc)

    # Store the final model
    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(save_folder, save_name + 'Final.pth'))
import torch
from torch.utils.data import DataLoader


def check_keys(model, pretrained_state_dict):

    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix前缀 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
    else:
        device = torch.cuda.current_device()    # gpu
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def test(test_data, origin_gt, model, device, cfg):

    #num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']
    batch_size = cfg['batch_size']

    model = load_model(model, cfg['model_weights'], device)
    model.eval()
    model = model.to(device)

    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Data load
    batch_data = DataLoader(test_data, batch_size, shuffle=True, pin_memory=True)

    predict_correct = 0
    label_num = 0
    predict_label = []
    change_gailv=[]
    unchange_gailv=[]
    for batch_idx, batch_sample in enumerate(batch_data):

        img1, img2, target, indices = batch_sample
        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.no_grad():
            prediction = model(img1, img2)

        label = prediction.cpu().argmax(dim=1, keepdim=True)

        if target.sum() > 0:
            predict_correct += label.eq(target.view_as(label)).sum().item()
            label_num += len(target)
        predict_label.append(torch.cat([indices, label], dim=1))
        change_gailv.append(torch.cat([indices,prediction.cpu()[:,0].unsqueeze(1)],dim=1))
        unchange_gailv.append(torch.cat([indices, prediction.cpu()[:, 1].unsqueeze(1)], dim=1))


    predict_label = torch.cat(predict_label, dim=0)   # torch.Size([22316, 4])
    change_gailv = torch.cat(change_gailv, dim=0)
    unchange_gailv = torch.cat(unchange_gailv, dim=0)
    test_acc = 100 * predict_correct / label_num
    if label_num > 0:
        print('OA {:.2f}%'.format(test_acc))

    return predict_label, test_acc,change_gailv,unchange_gailv
import  numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
def accuracy_assessment(img_gt, changed_map):
    '''
        assess accuracy of changed map based on ground truth
    '''
    esp = 1e-6

    height, width = changed_map.shape
    changed_map_ = np.reshape(changed_map, (-1,))
    img_gt_ = np.reshape(img_gt, (-1,))

    cm = np.ones((height * width,))
    cm[changed_map_ == 1] = 2
    cm[changed_map_ == 0] = 1

    gt = np.zeros((height * width,))
    gt[img_gt_ == 1] = 2
    gt[img_gt_ == 0] = 1

    # scikit-learn 混淆矩阵函数 sklearn.metrics.confusion_matrix API 接口
    conf_mat = confusion_matrix(y_true=gt, y_pred=cm, labels=[1, 2])
    kappa_co = cohen_kappa_score(y1=gt, y2=cm, labels=[1, 2])

    # TN, FP, FN, TP
    TN, FP, FN, TP = conf_mat.ravel()
    P = TP / (TP + FP + esp)
    R = TP / (TP + FN + esp)
    F1 = 2 * P * R / (P + R + esp)
    acc = (TP + TN) / (TP + TN + FP + FN + esp)

    oa = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    return conf_mat, oa, kappa_co, P, R, F1, acc




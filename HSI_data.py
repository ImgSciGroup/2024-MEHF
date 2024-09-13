import torch.utils.data as data


class HSI_data(data.Dataset):
    def __init__(self, data_sample, cfg):

        self.phase = cfg
        self.img1 = data_sample['img1_pad']
        self.img2 = data_sample['img2_pad']
        self.patch_coordinates = data_sample['patch_coordinates']
        self.gt = data_sample['img_gt']
        self.train_label = data_sample['init_sample_center'].long()
        if self.phase == 'train':
            self.data_indices = data_sample['train_sample_center']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_sample_center']

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_label)
        elif self.phase == 'test':
            return len(self.data_indices)

    def __getitem__(self, idx):
        if self.phase == 'train':
            index = self.train_label[idx]
            img_index = self.patch_coordinates[index[0]]
            img1 = self.img1[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
            img2 = self.img2[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
            label = index[3]
            return img1,img2,label,index[0:3]
        if self.phase =='test':
            index = self.data_indices[idx]
            img_index = self.patch_coordinates[index[0]]

            # From pad_img intercept samples based on coordinates
            img1 = self.img1[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
            img2 = self.img2[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
            label = self.gt[index[1], index[2]]

            return img1, img2, label, index



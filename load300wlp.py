import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import os
import cv2
import numpy as np

class Dataset300WLP(Dataset):
    def __init__(self, train_data_file, transform=None):
        super(Dataset300WLP, self).__init__()
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.transform = transform

    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip().split()
                if(len(item)==4):
                    item[0] = item[0] + ' ' + item[1]
                    item[1] = item[2] + ' ' + item[3]
                    del item[2:]
                self.train_data_list.append(item)

    def __len__(self):
        return len(self.train_data_list)

    def __getitem__(self, idx):
        img_path, label_path = self.train_data_list[idx][0], self.train_data_list[idx][1]
        img = cv2.imread(img_path)
        label = np.load(label_path)
        sample = {'origin_img':img, 'gt_label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    #Convert ndarrays in sample to Tensors.
    def __call__(self, sample):
        origin_img, gt_label = sample['origin_img'], sample['gt_label']
        
        # numpy img (H x W x C) ---> torch img (C x H x W)
        origin_img = origin_img.transpose((2, 0, 1))
        gt_label = gt_label.transpose((2, 0, 1))
        
        origin_img = origin_img.astype("float32")/255.0
        gt_label = gt_label.astype("float32")/255.0
        gt_label = np.clip(gt_label, 0, 1)

        return {'origin_img':torch.from_numpy(origin_img), 'gt_label':torch.from_numpy(gt_label)}

class ToNormalize(object):
    #Tensor normalize
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        origin_img, gt_label = sample['origin_img'], sample['gt_label']
        origin_img = transforms.functional.normalize(origin_img, self.mean, self.std, self.inplace)
        return {'origin_img':origin_img, 'gt_label':gt_label}


#wlp300 = Dataset300WLP('data/300wlp_all.txt')
#print(len(wlp300.train_data_list))
#print(wlp300.train_data_list)
#print(wlp300.train_data_list[2][0])
#print(wlp300.__len__())
#print(wlp300.train_data_list[2])
#print(wlp300.__getitem__(2))
#print(wlp300.train_data_list[5][0])
#print(wlp300.__getitem__(5)['gt_label'].shape)
"""
count = 0
for i in range(122450):
    if '.npy' in wlp300.train_data_list[i][1]:
        count += 1
    else:
        print(wlp300.train_data_list[i][1])
print(count)
"""
#print(wlp300.train_data_list[88800])
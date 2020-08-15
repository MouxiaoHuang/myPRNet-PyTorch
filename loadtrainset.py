import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import scipy.io as sio

class Trainset300WLP(Dataset):
    def __init__(self, trainset_path, trans=None):
        super(Trainset300WLP, self).__init__()
        self.trainset_path = trainset_path
        self.trans = trans
        self.trainset_path_list = []
        self.readData()
    
    def readData(self):
        with open(self.trainset_path) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip().split()
                if(len(item)==4):
                    item[0] = item[0] + ' ' + item[1]
                    item[1] = item[2] + ' ' + item[3]
                    del item[2:]
                self.trainset_path_list.append(item)
        
    def __len__(self):
        return len(self.trainset_path_list)
    
    def __getitem__(self, idx):
        img_path, label_path = self.trainset_path_list[idx][0], self.trainset_path_list[idx][1]
        img = cv2.imread(img_path) # 256 x 256 x 3
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)# BGR

        info = sio.loadmat(label_path)
        pose_para = info['Pose_Para'].T.astype(np.float32)# 7 x 1
        shape_para = info['Shape_Para'].astype(np.float32)# 199 x 1
        exp_para = info['Exp_Para'].astype(np.float32)# 29 x 1
        color_para = info['Color_Para'].T.astype(np.float32)# 7 x 1
        illum_para = info['Illum_Para'].T.astype(np.float32)# 10 x 1
        tex_para = info['Tex_Para'].astype(np.float32)# 199 x 1
        # label: 451 x 1
        label = np.vstack((pose_para, shape_para, exp_para, color_para, illum_para, tex_para))

        sample = {'origin_img':img, 'gt_label':label}
        if self.trans:
            sample = self.trans(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        origin_img, gt_label = sample['origin_img'],  sample['gt_label']
        
        origin_img = origin_img.transpose((2,0,1)) # 3 x 224 x 224
        gt_label = gt_label.transpose((1,0)) # 1 x 451
        origin_img = origin_img.astype("float32")/255.0
        gt_label = gt_label.astype("float32")

        return {'origin_img':torch.from_numpy(origin_img), 'gt_label':torch.from_numpy(gt_label)}

class ToNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self, sample):
        origin_img, gt_label = sample['origin_img'], sample['gt_label']
        origin_img = transforms.functional.normalize(origin_img, self.mean, self.std, self.inplace)
        return {'origin_img':origin_img, 'gt_label':gt_label}


#data = Trainset300WLP('Data/Net2traindata/trainsetFile.txt')
#print(data.trainset_path_list[88800])
#print(data.__len__())
#print(data.__getitem__(88800)['origin_img'].dtype)
#print(data.__getitem__(0)['gt_label'])
#print(data.__getitem__(0)['gt_label'].shape)

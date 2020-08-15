import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import random
import cv2
import argparse
from PIL import Image

from ResFCN256 import ResFCN256
from load300wlp import Dataset300WLP, ToTensor, ToNormalize
'''''
class LoadTrainData(object):
    def __init__(self, train_data_file):
        super(LoadTrainData, self).__init__()
        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self.train_data_list)

    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip().split()
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        for item in batch_list:
            if len(item)==2:
                img_name = item[0]
                label_name = item[1]
            else:
                img_name = item[0] + ' ' + item[1]
                label_name = item[2] + ' ' + item[3]
            img = cv2.imread(img_name)
            label = np.load(label_name)

            img_array = np.array(img, dtype=np.float32)
            imgs.append(img_array/255.0)

            label_array = np.array(label, dtype=np.float32)
            lables_array_norm = (label_array)/(255.0*1.1)
            labels.append(lables_array_norm)

        batch.append(imgs)
        batch.append(labels)
        return batch
    
    def __call__(self, batch_num):
        if(self.index+batch_num) <= self.num_data:
            batch_list = self.train_data_list[self.index:(self.index+batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num
            return batch_data
        elif self.index < self.num_data:
            batch_list = self.train_data_list[self.index:self.num_data]
            batch_data = self.getBatch(batch_list)
            self.index = 0
            return batch_data
        else:
            self.index = 0
            batch_list = self.train_data_list[self.index:(self.index+batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num
            return batch_data
'''''

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epoch = args.epoch
    train_data_file = args.train_data_file
    learning_rate = args.learning_rate
    model_path = args.model_path

    #save_dir = args.model_path
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    
    # mask
    weight_mask_path = 'mask/uv_weight_mask.png'
    face_mask_path = 'mask/uv_face_mask.png'
    weight_mask = cv2.imread(weight_mask_path, cv2.IMREAD_GRAYSCALE).astype('float32')
    weight_mask = weight_mask/255.0
    face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE).astype('float32')
    face_mask = face_mask/255.0
    final_mask = np.multiply(face_mask, weight_mask)
    final_weight_mask = np.zeros(shape=(1,256,256,3)).astype('float32')
    final_weight_mask[0,:,:,0]=final_mask
    final_weight_mask[0,:,:,1]=final_mask
    final_weight_mask[0,:,:,2]=final_mask
    #print(final_weight_mask.shape)
    final_weight_mask = final_weight_mask.transpose((0,3,1,2))
    #print(final_weight_mask.shape)
    final_weight_mask = torch.from_numpy(final_weight_mask)
    final_weight_mask = final_weight_mask.to(device=torch.device('cuda'))
    #print(final_weight_mask.shape)
    # mask
    #mask = cv2.imread('mask/weight_mask_final.jpg')
    #mask = mask.astype("float32")/255.0
    #mask = mask.transpose((2,0,1))
    #mask = torch.from_numpy(mask)#256x256x3 tensor
    #mask = mask.cuda()
    #print(mask.shape)

    # load training data 300wlp
    data300wlp = Dataset300WLP(train_data_file, 
                                transform=transforms.Compose([
    ToTensor(),
    ToNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    dataloader300wlp = DataLoader(data300wlp, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    print(data300wlp.__len__())
    
    # network
    net = ResFCN256()
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    net.to(device) 
    #print("net is:\n",net)

    # loss and optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5) # decays half after each 5 epoches
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.L1Loss(reduction="mean")
    print(optimizer.defaults['lr'])
    # train
    for ep in range(epoch):
        running_loss = 0.0
        for i,data in enumerate(dataloader300wlp, 0):
            img, label = data['origin_img'].cuda(), data['gt_label'].cuda()
            #img = img.cuda()
            #label = label.cuda()
            
            optimizer.zero_grad()

            #print(img.shape)
            #print(label.shape)
            #print(mask.shape)
            outputs = net(img)
            #print(outputs.shape)
            #outputs = torch.mul(outputs, mask)
            #label = torch.mul(label, mask)
            outputs = torch.mul(outputs, final_weight_mask)
            label = torch.mul(label, final_weight_mask)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            running_loss += loss.item()
            if i%20 == 19:
                #print('[%d, %5d] loss: %.3f' %
                #      (ep + 1, i + 1, running_loss / 20))
                print('[epoch: %d, %d, lr: %.10f] loss: ' % (ep+1, i+1, optimizer.param_groups[0]['lr']), running_loss/20)
                running_loss = 0.0
        scheduler.step()
    print("Final Loss: ", loss.item())
    print("Finished Training!")

    # save model
    torch.save(net.state_dict(), model_path)
    


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='my3DFaceRecon')
    par.add_argument('--train_data_file', default='data/300wlp_all.txt', type=str, help='Training data file (txt)')
    par.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    par.add_argument('--epoch', default=20, type=int, help='Epoches to train')
    par.add_argument('--batch_size', default=16, type=int, help='Batch size')
    par.add_argument('--model_path', default='model/model1.pth', help='Model path')
    par.add_argument('--gpu', default='0', type=str, help='GPU ID')

    main(par.parse_args())
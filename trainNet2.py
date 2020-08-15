import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import argparse
import os

#ResNet50 = torchvision.models.resnet50(pretrained=False)
#ResNet50.fc = torch.nn.Linear(2048, 235, bias=True) # [shape_para, exp_para, pose_para] = [199, 29, 7] --> 235
#回归完整参数
# x = (pose, shape, exp, color, illum, tex) [7,199,29,7,10,199] --> 451
ResNet50 = torchvision.models.resnet50(pretrained=False)
ResNet50.fc = torch.nn.Linear(2048, 451, bias=True)
#from ResNet50 import ResNet50
from loadtrainset import Trainset300WLP, ToTensor, ToNormalize

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epoch = args.epoch
    model_path = args.model_path
    learning_rate = args.learning_rate
    trainset_path = args.trainset_path

    print('batch size: ', batch_size)
    print('total epoch: ', epoch)

    trainset = Trainset300WLP(trainset_path, trans=transforms.Compose([
        ToTensor(),
        ToNormalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]))
    #trainsetLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    trainsetLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(trainset.__len__())

    #net2 = ResNet50()
    device = torch.device('cuda')
    #net2.to(device)
    ResNet50.to(device)

    optimizer = optim.Adam(ResNet50.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.L1Loss(reduction='mean')
    #print(optimizer.defaults['lr'])


    for ep in range(epoch):
        #print('ep: ', ep)
        running_loss = 0.0
        for i,data in enumerate(trainsetLoader,0):
            img, label = data['origin_img'].cuda(), data['gt_label'].cuda()

            #print(img.shape)#batch_size x 3 x 256 x 256
            #print(label.shape)#batch_size x 1 x 235

            optimizer.zero_grad()

            outputs = ResNet50(img)
            #print(outputs.shape)
            outputs = torch.unsqueeze(outputs, 1)
            #print('label_shape: ', label.shape)#batch_size x 1 x 451
            #print('outputs_shape: ', outputs.shape)#batch_size x 1 x 451
            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i%20 == 19:
                print('[epoch: %d, %d, lr: %.10f] loss: ' % (ep+1, i+1, optimizer.param_groups[0]['lr']), running_loss/20)
                running_loss = 0.0
        scheduler.step()
    print("Finished Loss: ", loss.item)
    print("Finished Training!")

    state = {'model':ResNet50.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':ep}
    torch.save(state, model_path)
    #torch.save(ResNet50.state_dict(), model_path)



if __name__ == '__main__':
    par = argparse.ArgumentParser(description='myFaceRecon_Net2')
    par.add_argument('--trainset_path', default='Data/Net2traindata/trainsetFile.txt')
    par.add_argument('--model_path', default='model/model_param.pth', type=str)
    par.add_argument('--gpu', default='0', type=str)
    par.add_argument('--learning_rate', default=0.0001, type=float)
    par.add_argument('--batch_size', default=16, type=int)
    par.add_argument('--epoch', default=20, type=int)

    main(par.parse_args())
import torch
import torchvision
import cv2
import dlib
import numpy as np
import scipy.io as scio

from torchvision import transforms
from skimage.transform import estimate_transform,warp,rescale,resize


trans_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ResNet50 = torchvision.models.resnet50(pretrained=False)
ResNet50.fc = torch.nn.Linear(2048, 235, bias=True) # [shape_para, exp_para, pose_para] = [199, 29, 7] --> 235
# x = (pose, shape, exp, color, illum, tex) [7,199,29,7,10,199] --> 451
#ResNet50.fc = torch.nn.Linear(2048, 451, bias=True)

#checkpoint = torch.load('model/model_param.pth')
#ResNet50.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#epoch = checkpoint(['epoch'])

ResNet50.load_state_dict(torch.load('model/model2.pth'))
ResNet50.eval()
if(torch.cuda.device_count()>0):
    ResNet50 = ResNet50.to("cuda")

net2 = ResNet50

image = cv2.imread('TestImages/04.jpg')
cv2.imshow('input image', image)
face_detector = dlib.cnn_face_detection_model_v1('Data/face_detector/mmod_human_face_detector.dat')
detected_faces = face_detector(image, 1)
d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
old_size = (right - left + bottom - top)/2
center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
size = int(old_size*1.58)
print('size: ', size)

# crop image
src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
DST_PTS = np.array([[0,0], [0,256 - 1], [256 - 1, 0]])
tform = estimate_transform('similarity', src_pts, DST_PTS)
image = image/255.
image = warp(image, tform.inverse, output_shape=(256, 256))
cv2.imshow('before resize', image)
image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
cv2.imshow('net2 in-image', image)
[h,w,c] = image.shape
imageTensor = trans_img(image)
imageTensor = imageTensor.unsqueeze(0)
imageTensor = imageTensor.type(torch.FloatTensor)
print(h, w, c)

outputs = net2.forward(imageTensor.cuda())
print(outputs.shape)
#print(outputs.data[0][234])
outputs_np = outputs.data.cpu().numpy()
print(outputs_np.shape)
#print(outputs_np[0][234])

cv2.waitKey(0)

scio.savemat('04.mat', {'Shape_Para':outputs_np[0][0:199], 'Exp_Para':outputs_np[0][199:228], 'Pose_Para':outputs_np[0][228:235]})
#scio.savemat('6_param.mat', {'Pose_Para':outputs_np[0][0:7], 'Shape_Para':outputs_np[0][7:206], 'Exp_Para':outputs_np[0][206:235], 'Color_Para':outputs_np[0][235:242], 'Illum_Para':outputs_np[0][242:252], 'Tex_Para':outputs_np[0][252:451]})
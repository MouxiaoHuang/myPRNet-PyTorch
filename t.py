import numpy as np
import scipy.io as sio

label_path = 'C:/Users/MSI/Desktop/300WLP/AFW/AFW_134212_1_0.mat'

info = sio.loadmat(label_path)
"""
pose_para = info['Pose_Para'].T.astype(np.float32)# 7 x 1，依次是R（pitch，yaw，roll）、t、s
shape_para = info['Shape_Para'].astype(np.float32)# 199 x 1
exp_para = info['Exp_Para'].astype(np.float32)# 29 x 1
print(pose_para)

print(pose_para.shape)
print(shape_para.shape)
print(exp_para.shape)

x = np.vstack((pose_para, exp_para))
x = np.vstack((x, shape_para))
print(x.shape)
print(x.dtype)

x = x.transpose((1,0))
print(x.shape)
for i in range(7):
    print(x[0][i])
"""
print(info.keys())
pose = info['Pose_Para'].T.astype(np.float32)
shape = info['Shape_Para'].astype(np.float32)
exp = info['Exp_Para'].astype(np.float32)
color = info['Color_Para'].T.astype(np.float32)
illum = info['Illum_Para'].T.astype(np.float32)
tex = info['Tex_Para'].astype(np.float32)
label = np.vstack((pose, shape, exp, color, illum, tex))
print(label.shape)
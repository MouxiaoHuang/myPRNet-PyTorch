import cv2
import numpy as np
import math
import dlib

face_detector = dlib.cnn_face_detection_model_v1('Data/face_detector/mmod_human_face_detector.dat')
lm_detector = dlib.shape_predictor('Data/face_detector/shape_predictor_68_face_landmarks.dat')

def calAngle(p8col, p8row, p33col, p33row):
    tanAnlge = (p8col-p33col)*1.0/((p8row-p33row)*1.0)
    Angle = math.atan(tanAnlge) * 180 / 3.14
    return Angle

def rotate_bound(image,angle):#angle > 0为顺时针
    #获取图像的尺寸
    #旋转中心
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    
    return cv2.warpAffine(image,M,(nW,nH))

def process_rotate_image(image):
    detected_faces = face_detector(image, 1)
    face = detected_faces[0].rect
    lm = lm_detector(image, face)
    rotated_img = rotate_bound(image, calAngle(lm.part(8).x, lm.part(8).y, lm.part(33).x, lm.part(33).y))
    return rotated_img
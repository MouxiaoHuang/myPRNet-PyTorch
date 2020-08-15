import torch
import numpy as np
from ResFCN256 import ResFCN256

import dlib
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp

"""
class Net1:
    def __init__(self, is_dlib=False, prefix='.'):
        self.resolution_inp = 256
        self.resolution_op = 256

        if is_dlib:
            detector_path = os.path.join(prefix, 'Data/face_detector/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(detector_path)
        
        self.pos_predictor = ResFCN256()
        self.pos_predictor.load_state_dict(torch.load('model/model1.pth'))
        self.pos_predictor.eval()
        if torch.cuda.device_count() > 0:
            self.pos_predictor = self.pos_predictor.to("cuda")

        self.uv_kpt_ind = np.loadtxt('Data/uv-data/uv_kpt_ind.txt').astype(np.int32)
        self.face_ind = np.loadtxt('Data/uv-data/face_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt('Data/uv-data/triangles.txt').astype(np.int32)
        self.uv_coords = self.generate_uv_coords()

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution), range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution ** 2,-1])
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0],1])))
        return uv_coords
    
    def dlib_detect(self, image):
        return self.face_detector(image,1)

    def net1_forward(self, image):
        return self.pos_predictor(image)

    def process(self, input, image_info=None):
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        if image_info is not None:
            if np.max(image_info.shape) > 4:
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0,:])
                right = np.max(kpt[0,:])
                top = np.min(kpt[1,0])
                bottom = np.max(kpt[1,:])
            else:
                bbox = image_info
                left = bbox[0]
                right = bbox[1]
                top = bbox[2]
                bottom = bbox[3]
            old_size = (right - left + bottom - top)/2
            center = np.array([right-(right-left)/2.0, bottom-(bottom-top)/2.0])
            size = int(old_size*1.6)
        else:
            detected_faces = self.dlib_detect(image)
            if len(detected_faces)==0:
                print('warning: no detected faces')
                return None
            d = detected_faces[0].rect
            left = d.left()
            right = d.right()
            top = d.top()
            bottom = d.bottom()
            old_size = (right-left+bottom-top)/2
            center = np.array([right-(right-left)/2.0, bottom-(bottom-top)/2.0 + old_size*0.14])
            size = int(old_size*1.58)
        
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/255.0
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        cropped_pos = self.net1_forward(cropped_image)

        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])

        return pos

    def get_landmarks(self, pos):
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt

    def get_vertices(self, pos):
        # 3d position map: 3 x 256 x256
        all_vertices = np.reshape(pos, [self.resolution_op ** 2, -1])
        vertices = all_vertices[self.face_ind, :]
        return vertices
"""

###############################


class Net1:
    def __init__(self, model_dir, **kwargs):
        self.resolution_inp = kwargs.get("resolution_inp") or 256
        self.resolution_op = kwargs.get("resolution_op") or 256
        self.channel = kwargs.get("channel") or 3
        self.size = kwargs.get("size") or 16

        self.uv_kpt_ind_path = kwargs.get("uv_kpt_path") or "mask/uv_kpt_ind.txt"
        self.face_ind_path = kwargs.get("face_ind_path") or "mask/face_ind.txt"
        self.triangles_path = kwargs.get("triangles_path") or "mask/triangles.txt"

        # load model
        #self.pos_predictor = ResFCN256()
        #state = torch.load(model_dir)
        #self.pos_predictor.load_state_dict(state['prnet'])
        #self.pos_predictor.eval()
        self.pos_predictor = ResFCN256()
        self.pos_predictor.load_state_dict(torch.load(model_dir))
        self.pos_predictor.eval()
        if torch.cuda.device_count() > 0:
            self.pos_predictor = self.pos_predictor.to("cuda")

        # load uv_file
        self.uv_kpt_ind = np.loadtxt(self.uv_kpt_ind_path).astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(self.face_ind_path).astype(np.int32)
        self.triangles = np.loadtxt(self.triangles_path).astype(np.int32)

        self.uv_coords = self.generate_uv_coords()

    def dlib_detect(self, image):
        return self.face_detector(image, 1)

    def net1_forward(self, img):
        return self.pos_predictor(img) # 3d position map: 3 x 256 x256 array

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution), range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution ** 2,-1])
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0],1])))
        return uv_coords

    def get_landmarks(self, pos):
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt

    def get_vertices(self, pos):
        # 3d position map: 3 x 256 x256
        all_vertices = np.reshape(pos, [self.resolution_op ** 2, -1])
        vertices = all_vertices[self.face_ind, :]
        return vertices

    def get_colors_from_texture(self, texture):
        '''
        Args:
            texture: the texture map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        all_colors = np.reshape(texture, [self.resolution_op**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors

    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3
        #colors = image[ind[:,0], ind[:,1], :]

        return colors
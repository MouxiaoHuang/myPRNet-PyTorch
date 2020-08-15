import cv2
import skimage
import numpy as np
import os
from glob import glob
import scipy.io as sio
import argparse
import ast
import dlib
from skimage.transform import estimate_transform,warp,rescale,resize

from api import Net1
from torchvision import transforms, utils, models
import torch

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.rotate_image import process_rotate_image
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture
def main(args):
    if args.isShow or args.isTexture:
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box, plot_obj

    trans_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    net1 = Net1(args.model)
    #net1 = Net1(is_dlib=args.isDlib)

    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '.png')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    print("#" * 20)
    print("[Net1] {} picture were under processing ~ ".format(total_num))
    print("#" * 20)
    print(image_path_list)

    for i, image_path in enumerate(image_path_list):
        #print(i)
        #print(image_path)
        name = image_path.strip().split('\\')[-1][:-4]

        #image = cv2.imread(image_path)
        image = skimage.io.imread(image_path)
        #if(i==0):
        #    image = cv2.imread('D:/project/face3d/examples/results/posmap_300WLP/0.jpg')
            #image = cv2.imread('D:/project/my3DFaceRecon/TestImages/0.jpg')

        #image = process_rotate_image(image)#矫正图片人脸，使其竖直

        face_detector = dlib.cnn_face_detection_model_v1('Data/face_detector/mmod_human_face_detector.dat')
        detected_faces = face_detector(image, 1)
        d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
        left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
        size = int(old_size*1.58)
        print(size)
        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,256 - 1], [256 - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        image = image/255.
        image = warp(image, tform.inverse, output_shape=(256, 256))
        #cv2.imshow('cropped image', image)
        [h,w,c] = image.shape
        imageTensor = trans_img(image)
        imageTensor = imageTensor.unsqueeze(0)
        imageTensor = imageTensor.type(torch.FloatTensor)
        pos = net1.net1_forward(imageTensor.cuda())
        
        if(i==0):
            t = np.load('C:/Users/MSI/Desktop/model_uv/01.npy')
            t = t.transpose(2,0,1)
            #print(t.shape)
            ttensor = torch.from_numpy(t)
            ttensor = ttensor.unsqueeze(0)
            #print(ttensor.shape)
            ttensor = ttensor.cuda()
            ttensor = ttensor/255.0
            #pos = ttensor
            
            print(pos.shape)

       
        #[h,w,c] = image.shape

        #image = cv2.resize(image, (256,256))
        #image_t = trans_img(image)
        #image_t = image_t.unsqueeze(0)
        #pos = net1.net1_forward(image_t.cuda())

        out = pos.cpu().detach().numpy()
        pos = np.squeeze(out)
        cropped_pos = pos * 255
        pos = cropped_pos.transpose(1,2,0)
        if pos is None:
            continue

        if args.is3d or args.isMat or args.isPose or args.isShow:
            vertices = net1.get_vertices(pos)
            if args.isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:,1] = h - 1 - save_vertices[:,1]

        if args.isImage:
            cv2.imwrite(os.path.join(save_folder, name + '.jpg'), image)

        if args.is3d:
            colors = net1.get_colors(image, vertices)
            cv2.imshow('colormap', image)
            #if(i==0):
                #img = cv2.imread('D:/project/face3d/examples/results/posmap_300WLP/0.jpg')
                #img = skimage.io.imread('D:/project/face3d/examples/results/posmap_300WLP/0.jpg')
                #img = skimage.io.imread('C:/Users/MSI/Desktop/model_uv/00.jpg')
                #colors = net1.get_colors(img, vertices)
            #print(colors.shape)
            #colors = colors[:, [1,0,2]]
            #print(np.max(colors), np.min(colors))

            if args.isTexture:
                if args.texture_size != 256:
                    pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
                else:
                    pos_interpolated = pos.copy()
                texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
                if args.isMask:
                    vertices_vis = get_visibility(vertices, net1.triangles, h, w)
                    uv_mask = get_uv_mask(vertices_vis, net1.triangles, net1.uv_coords, h, w, net1.resolution_op)
                    uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
                    texture = texture*uv_mask[:,:,np.newaxis]
                write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, net1.triangles, texture, net1.uv_coords/net1.resolution_op)#save 3d face with texture(can open with meshlab)
            else:
                write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, net1.triangles, colors) #save 3d face(can open with meshlab)

        if args.isKpt or args.isShow:
            kpt = net1.get_landmarks(pos)
            np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)

        if args.isPose or args.isShow:
            camera_matrix, pose = estimate_pose(vertices)
            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)
            np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix)
            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)
        
        if args.isMat:
            sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': net1.triangles})

        if args.isShow:
            image_pose = plot_pose_box(image, camera_matrix, kpt)
            #if(i==0):
            #    tmp_img = cv2.imread('D:/project/face3d/examples/results/posmap_300WLP/0.jpg')
            #    image = tmp_img
            
            cv2.imshow('sparse alignment', plot_kpt(image, kpt))
            cv2.imshow('dense alignment', plot_vertices(image, vertices))
            cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
            #cv2.imshow('obj', plot_obj(image, vertices, colors))
            cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Net1 (PRN)')

    parser.add_argument('-i','--inputDir', default='TestImages/', type=str)
    parser.add_argument('-o','--outputDir', default='TestImages/results', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--model', default='model/model1.pth', type=str)
    
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval)
    parser.add_argument('--is3d', default=True, type=ast.literal_eval)
    parser.add_argument('--isMat', default=True, type=ast.literal_eval)
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval)
    parser.add_argument('--isPose', default=False, type=ast.literal_eval)
    parser.add_argument('--isShow', default=True, type=ast.literal_eval)
    parser.add_argument('--isImage', default=True, type=ast.literal_eval)

    parser.add_argument('--isFront', default=False, type=ast.literal_eval)
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval)
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval)
    parser.add_argument('--isMask', default=False, type=ast.literal_eval)
    parser.add_argument('--texture_size', default=256, type=int)

    main(parser.parse_args())
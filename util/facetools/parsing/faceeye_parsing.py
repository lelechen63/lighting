#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import torch
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

import pickle



to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def vis_parsing_maps(im, parsing_anno, stride=1, show=False, save_parsing_path='imgs/gg.png', save_vis_path = None):

    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    # num_of_class = 19
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if show:
        cv2.imshow('parsing res', vis_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save result 
    cv2.imwrite(save_parsing_path, vis_parsing_anno)
    if save_vis_path is not None:
        cv2.imwrite(save_vis_path, vis_im)

    # return vis_im


def parsing(img, facenet, idet, save_face_path= None ):
    # img = Image.open(img_path)
    with torch.no_grad():
        shape = img.size
        image = img.resize((512, 512), Image.BILINEAR)
        image = to_tensor(image)
        image = torch.unsqueeze(image, 0)
        image = image.cuda()
        out = facenet(image)[0]
        parsing_maps = out.squeeze(0).cpu().numpy().argmax(0).astype('float32')
    parsing_maps = cv2.resize(parsing_maps, shape, interpolation=cv2.INTER_NEAREST)

    if save_face_path is not None:
        binary_mask= np.zeros((shape), np.uint8)
        binary_mask[parsing_maps==17] = 1
        binary_mask[parsing_maps>14] = 0
        binary_mask[parsing_maps>0] = 1

        # front_img = img * binary_mask[:,:, np.newaxis]
        cv2.imwrite(save_face_path, binary_mask)
    im = np.array(img)[..., ::-1]
    # im = cv2.imread(img_path)[..., ::-1]
    # try:
    blank_image1 = np.zeros((shape), np.uint8)
    blank_image2 = np.zeros((shape), np.uint8)
    eye_lms = idet.detect_iris(im)
    lms =   eye_lms[0][0,...].astype(np.int32)[:,::-1]

    cv2.fillConvexPoly(blank_image1, lms[:8], 8)
    cv2.fillConvexPoly(blank_image2, lms[8:16], 7)
    
    blank_image = blank_image1 + blank_image2
    blank_image[blank_image <15 ] = 0
    parsing_maps += blank_image
    parsing_maps[parsing_maps>19] = 19

    blank_image1 = np.zeros((shape), np.uint8)
    blank_image2 = np.zeros((shape), np.uint8)

    lms = eye_lms[0][1,...].astype(np.int32)[:,::-1]
    cv2.fillConvexPoly(blank_image1, lms[:8], 8)
    cv2.fillConvexPoly(blank_image2, lms[8:16], 8)

    blank_image = blank_image1 + blank_image2
    blank_image[blank_image < 16 ] = 0
    parsing_maps += blank_image 
    parsing_maps[parsing_maps>20] = 20  
    # except:
    #     print('no eyes ***********') 
        
    return parsing_maps


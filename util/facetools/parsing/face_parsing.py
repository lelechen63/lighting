#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import torch
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import BiSeNet


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


def parsing(imgs, net, mask_path = None):


    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        shape = imgs.size
        image = imgs.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing_maps = out.squeeze(0).cpu().numpy().argmax(0).astype('float32')
        parsing_maps = cv2.resize(parsing_maps, shape, interpolation=cv2.INTER_NEAREST)
        # return parsing_maps
        if mask_path is not None:
            binary_mask= np.zeros((shape), np.uint8)
            binary_mask[parsing_maps==17] = 1
            binary_mask[parsing_maps>14] = 0
            binary_mask[parsing_maps>0] = 1

            # front_img = img * binary_mask[:,:, np.newaxis]
            cv2.imwrite(mask_path, binary_mask)
        return parsing_maps

import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import random
from os import path as osp
import pickle
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import MNIST
from tqdm import tqdm
import pytorch_lightning as pl
from data.data import FacescapeDataModule
from options.step1_train_options import TrainOptions

from pytorch_lightning.callbacks import ModelCheckpoint

import util.util as util
import os
from util.visualizer import Visualizer
from util.render_class import meshrender

def pl2normal(checkpoint):
    state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'discriminator' in k:
            continue
        if 'vgg' in k :
            continue 
        if 'cls' in k :
            continue 
        name = k[10:]
        new_state_dict[name] = v
    return new_state_dict

opt = TrainOptions().parse()
opt.datasetname = "fs_texmesh"

def main():
    if opt.name == 'gmesh' :
        from model.meshnetwork import AE as module 
        opt.datasetname = "fs_mesh"

    totalmeanmesh = torch.FloatTensor( np.load( "./predef/meanmesh.npy" ) )
    totalstdmesh = torch.FloatTensor(np.load( "./predef/meshstd.npy" ))

    checkpoint_path = './checkpoints/gmesh/latest.ckpt'
    homepath = './predef'
    device = torch.device('cuda', 0)


    transform_fp = osp.join(homepath, 'transform.pkl')
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    edge_index_list = [util.to_edge_index(adj).to(device) for adj in tmp['adj']]

    down_transform_list = [
        util.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        util.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]
    module =  module(3,
                [16, 16, 16, 32],
                256,
                edge_index_list,
                down_transform_list,
                up_transform_list,
                K=6)

    checkpoint = torch.load(checkpoint_path)
    module.load_state_dict(pl2normal(checkpoint['state_dict']))

    dm = FacescapeDataModule(opt)
    dm.setup()
    testdata = dm.test_dataloader()
    opt.name = opt.name + '_test'
    visualizer = Visualizer(opt)
    l2loss = torch.nn.MSELoss()
    module = module.to(device)
    with torch.no_grad():
        for num,batch in enumerate(testdata):
            rec_mesh_A, code = module( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))
            # save_p = os.path.join('/data/home/uss00022/lelechen/data/Facescape/textured_meshes/',  batch['A_path'][0] + '_mesh.npz')
            save_p = os.path.join('/mnt/Backup/lele/Facescape/meshcode/',  batch['A_path'][0] + '_mesh.npz')
            tmp =  batch['A_path'][0].split('/')
            os.makedirs(  os.path.join('/mnt/Backup/lele/Facescape/meshcode/',  tmp[0], tmp[1]), exists_ok =True)
            print (save_p)
            np.savez( save_p, w=code.detach().cpu().numpy())
            
            # tmp = batch['A_path'][0].split('/')
            # gt_mesh = batch['Amesh'].data[0].cpu() * totalstdmesh + totalmeanmesh
            # rec_Amesh = rec_mesh_A.data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh 
            # gt_mesh = gt_mesh.float()
            # rec_Amesh = rec_Amesh.float()
            # gt_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),gt_mesh )
            # rec_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]), rec_Amesh )

            # gt_Amesh = np.ascontiguousarray(gt_Amesh, dtype=np.uint8)
            # gt_Amesh = util.writeText(gt_Amesh, batch['A_path'][0], 100)
            # visuals = OrderedDict([
            #     ('gt_Amesh', gt_Amesh),
            #     ('rec_Amesh', rec_Amesh),
            
            #     ])
            # visualizer.display_current_results(visuals, num, 1000000)
    
    opt.isTrain = True
    dm = FacescapeDataModule(opt)
    dm.setup()
    testdata = dm.test_dataloader()
    opt.name = opt.name + '_test'
    visualizer = Visualizer(opt)
    l2loss = torch.nn.MSELoss()
    module = module.to(device)
    with torch.no_grad():
        for num,batch in enumerate(testdata):
            rec_mesh_A, code = module( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))
            # save_p = os.path.join('/data/home/uss00022/lelechen/data/Facescape/textured_meshes/',  batch['A_path'][0] + '_mesh.npz')
            save_p = os.path.join('/mnt/Backup/lele/Facescape/meshcode/',  batch['A_path'][0] + '_mesh.npz')
            tmp =  batch['A_path'][0].split('/')
            os.makedirs(  os.path.join('/mnt/Backup/lele/Facescape/meshcode/',  tmp[0], tmp[1]), exists_ok =True)
            print (save_p)
            np.savez( save_p, w=code.detach().cpu().numpy())
            
            
            
            # tmp = batch['A_path'][0].split('/')
            # gt_mesh = batch['Amesh'].data[0].cpu() * totalstdmesh + totalmeanmesh
            # rec_Amesh = rec_mesh_A.data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh 
            # gt_mesh = gt_mesh.float()
            # rec_Amesh = rec_Amesh.float()
            # gt_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),gt_mesh )
            # rec_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]), rec_Amesh )

            # gt_Amesh = np.ascontiguousarray(gt_Amesh, dtype=np.uint8)
            # gt_Amesh = util.writeText(gt_Amesh, batch['A_path'][0], 100)
            # visuals = OrderedDict([
            #     ('gt_Amesh', gt_Amesh),
            #     ('rec_Amesh', rec_Amesh),
            
            #     ])
            # visualizer.display_current_results(visuals, num, 1000000)

main()
print ('++++++++++++ SUCCESS ++++++++++++++++!')
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import random
from os import path as osp
import pickle


from tqdm import tqdm
import pytorch_lightning as pl
from data.data import FacescapeDataModule
from options.img2code_train_options import TrainOptions

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
opt.datasetname = 'fs_mesh'
def main():
    device = torch.device('cuda', 0)
    Encoder = torch.load('./checkpoints/MeshEncoderDecoder/encoder.pth')
    Decoder = torch.load('./checkpoints/MeshEncoderDecoder/decoder.pth')


    totalmeanmesh = torch.FloatTensor( np.load( "./predef/meanmesh.npy" ) )
    totalstdmesh = torch.FloatTensor(np.load( "./predef/meshstd.npy" ))

    dm = FacescapeDataModule(opt)
    dm.setup()
    testdata = dm.test_dataloader()
    opt.name = opt.name + '_test'
    visualizer = Visualizer(opt)
    l2loss = torch.nn.MSELoss()
    Encoder = Encoder.to(device)
    with torch.no_grad():
        for num,batch in enumerate(testdata):
            print (num, '/', len(testdata))
            rec_mesh_A, code = module( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))

            code = Encoder( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))
            save_p = os.path.join(opt.dataroot +  '/meshcode/',  batch['A_path'][0] + '_mesh.npy')
            tmp =  batch['A_path'][0].split('/')
            os.makedirs(  os.path.join(opt.dataroot + '/meshcode/',  tmp[0], tmp[1]), exist_ok =True)
            np.save( save_p, code.detach().cpu().numpy())
            
    opt.isTrain = True
    dm.setup()
    testdata = dm.test_dataloader()
    with torch.no_grad():
        for num,batch in enumerate(testdata):
            print (num, '/', len(testdata))
            rec_mesh_A, code = module( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))

            code = Encoder( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))
            save_p = os.path.join(opt.dataroot + '/meshcode/',  batch['A_path'][0] + '_mesh.npy')
            tmp =  batch['A_path'][0].split('/')
            os.makedirs(  os.path.join(opt.dataroot + '/meshcode/',  tmp[0], tmp[1]), exist_ok =True)
            np.save( save_p, code.detach().cpu().numpy())
            
            
            

main()
print ('++++++++++++ SUCCESS ++++++++++++++++!')
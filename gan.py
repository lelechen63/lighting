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
if opt.debug:
    opt.nThreads = 1
# opt.datasetname = "fs_texmesh"

if  opt.name == 'meshtexgan':
    from model.model2 import MeshTexGANModule as module 
    opt.datasetname = "fs_meshtex"
elif opt.name == 'texgan':
    from model.model2 import TexGANModule as module 
    opt.datasetname = "fs_tex"
elif opt.name == 'tex':
    from model.model2 import TexModule as module 
    opt.datasetname = "fs_tex"
elif opt.name == 'gmesh' :
    from model.model2 import GraphConvMeshModule as module 
    opt.datasetname = "fs_mesh"
elif opt.name == 'disgmesh' :
    from model.model2 import DisGraphConvMeshModule as module 
    opt.datasetname = "fs_mesh"
elif opt.name == 'disgmesh2' :
    from model.model2 import DisGraphConvMeshModule2 as module 
    opt.datasetname = "fs_mesh"
totalmeanmesh = torch.FloatTensor( np.load( "./predef/meanmesh.npy" ) )#.view(-1,3) 
totalstdmesh = torch.FloatTensor(np.load( "./predef/meshstd.npy" ))#.view(-1,3)

meantex = torch.FloatTensor(np.load('./predef/meantex.npy')).permute(2, 0,1)
stdtex = torch.FloatTensor(np.load('./predef/stdtex.npy')).permute(2,0,1)

dm = FacescapeDataModule(opt)

if opt.isTrain:
    print ( opt.gpu_ids)
    if opt.no_vgg_loss:
        opt.name += '_novgg'
    if opt.no_cls_loss:
        opt.name += '_nocls'

    model = module(opt)
    print (model)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath= os.path.join(opt.checkpoints_dir, opt.name),
        filename='texmesh-{epoch:02d}-{train_loss:.2f}'
    )

    if len( opt.gpu_ids ) == 1:
        trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=1,  max_epochs= 100000, progress_bar_refresh_rate=20)

    else:
        trainer = pl.Trainer(callbacks=[checkpoint_callback], precision=16,gpus= len( opt.gpu_ids ), accelerator='ddp', max_epochs= 100000, progress_bar_refresh_rate=20)
    # trainer = pl.Trainer(gpus=4, accelerator='dp', max_epochs= 10000, progress_bar_refresh_rate=20)

    trainer.fit(model, dm)

else:
    print ('!!!!!!' + opt.name +'!!!!!!!!')
    if opt.name == 'texgan' :
        checkpoint_path = './checkpoints/texgan/latest.ckpt'
        
        from model.model2 import TexGenerator as module

        module = module(3,3)
    
        checkpoint = torch.load(checkpoint_path)
        module.load_state_dict(pl2normal(checkpoint['state_dict']))

        dm.setup()
        testdata = dm.test_dataloader()
        opt.name = opt.name + '_test'
        visualizer = Visualizer(opt)
        l2loss = torch.nn.MSELoss()
        print ('***********', len(testdata),'*************')
        for num,batch in enumerate(testdata):
            rec_tex_A= \
            module(  batch['Atex'])
            Atex = batch['Atex'].data[0].cpu()  #* stdtex + meantex 
            Atex = util.tensor2im(Atex  , normalize = False)
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = np.clip(Atex, 0, 255)


            rec_tex_A_vis =rec_tex_A.data[0].cpu()# * stdtex + meantex  
            rec_tex_A_vis = util.tensor2im(rec_tex_A_vis, normalize = False)
            rec_tex_A_vis = np.ascontiguousarray(rec_tex_A_vis, dtype=np.uint8)
            rec_tex_A_vis = np.clip(rec_tex_A_vis, 0, 255)

            tmp = batch['A_path'][0].split('/')
            visuals = OrderedDict([
                ('Atex', Atex),
                ('rec_tex_A',rec_tex_A_vis)
                ])
            visualizer.display_current_results(visuals, num, 1000000)
    elif opt.name == 'tex' :
        checkpoint_path = './checkpoints/tex/latest.ckpt'
        
        from model.model2 import TexGenerator as module

        module = module(opt.loadSize, not opt.no_linearity,
            3, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)
    
        checkpoint = torch.load(checkpoint_path)
        module.load_state_dict(pl2normal(checkpoint['state_dict']))

        dm.setup()
        testdata = dm.test_dataloader()
        opt.name = opt.name + '_test'
        visualizer = Visualizer(opt)
        l2loss = torch.nn.MSELoss()
        print ('***********', len(testdata),'*************')
        for num,batch in enumerate(testdata):
            rec_tex_A= \
            module(  batch['Atex'])
            Atex = batch['Atex'].data[0].cpu()  * stdtex + meantex 
            Atex = util.tensor2im(Atex  , normalize = False)
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = np.clip(Atex, 0, 255)


            rec_tex_A_vis =rec_tex_A.data[0].cpu() * stdtex + meantex  
            rec_tex_A_vis = util.tensor2im(rec_tex_A_vis, normalize = False)
            rec_tex_A_vis = np.ascontiguousarray(rec_tex_A_vis, dtype=np.uint8)
            rec_tex_A_vis = np.clip(rec_tex_A_vis, 0, 255)

            tmp = batch['A_path'][0].split('/')
            visuals = OrderedDict([
                ('Atex', Atex),
                ('rec_tex_A',rec_tex_A_vis)
                ])
            visualizer.display_current_results(visuals, num, 1000000)

    elif opt.name =='meshtexgan':
        checkpoint_path = './checkpoints/meshtexgan/latest.ckpt'
        device = torch.device('cuda', 0)
        from model.model2 import MeshTexGenerator as module 
        module = module(opt.loadSize, not opt.no_linearity, 
            3, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)
        checkpoint = torch.load(checkpoint_path)
        module.load_state_dict(pl2normal(checkpoint['state_dict']))

        dm.setup()
        testdata = dm.test_dataloader()
        # testdata = random.shuffle(testdata)
        opt.name = opt.name + '_test'
        visualizer = Visualizer(opt)
        l2loss = torch.nn.MSELoss()
        module = module.to(device)
        for num,batch in enumerate(testdata):
            rec_tex_A, rec_mesh_A, code = \
            module(batch['Atex'].to(device), batch['Amesh'].to(device) )
            tmp = batch['A_path'][0].split('/')
            gt_mesh = batch['Amesh'].data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh
            rec_Amesh = rec_mesh_A.data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh 
            gt_mesh = gt_mesh.float()
            rec_Amesh = rec_Amesh.float()

            gt_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),gt_mesh )
            rec_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]), rec_Amesh )

            gt_Amesh = np.ascontiguousarray(gt_Amesh, dtype=np.uint8)
            gt_Amesh = util.writeText(gt_Amesh, batch['A_path'][0], 100)
            
            Atex = batch['Atex'].data[0].cpu()  * stdtex + meantex 
            Atex = util.tensor2im(Atex  , normalize = False)
            
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])

            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = np.clip(Atex, 0, 255)

            rec_tex_A_vis =rec_tex_A.data[0].cpu() * stdtex + meantex  
            rec_tex_A_vis = util.tensor2im(rec_tex_A_vis, normalize = False)
            
            rec_tex_A_vis = np.ascontiguousarray(rec_tex_A_vis, dtype=np.uint8)
            rec_tex_A_vis = np.clip(rec_tex_A_vis, 0, 255)


            tmp = batch['A_path'][0].split('/')
            visuals = OrderedDict([
                ('Atex', Atex),
                ('rec_tex_A',rec_tex_A_vis),
                ('gt_Amesh', gt_Amesh),
                ('rec_Amesh', rec_Amesh),
                ])
            visualizer.display_current_results(visuals, num, 1000000)

    elif opt.name =='gmesh':
        from model.meshnetwork import AE as module 
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

        dm.setup()
        testdata = dm.test_dataloader()
        opt.name = opt.name + '_test'
        visualizer = Visualizer(opt)
        # l2loss = torch.nn.MSELoss()
        loss = []
        for num,batch in enumerate(testdata):
            if num == 100:
                break
            module = module.to(device)
            rec_mesh_A, code = module( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))


            tmp = batch['A_path'][0].split('/')
            gt_mesh = batch['Amesh'].data[0].cpu() * totalstdmesh + totalmeanmesh
            rec_Amesh = rec_mesh_A.data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh 
            
            print (rec_Amesh.shape, gt_mesh.shape)
            loss.append( ((rec_Amesh.view(-1) -  gt_mesh )** 2).mean())
            
            # gt_mesh = gt_mesh.float()
            # rec_Amesh = rec_Amesh.float()
            # gt_Amesh = meshrender( opt.dataroot, int(tmp[0]), int(tmp[-1].split('_')[0]),gt_mesh )
            # rec_Amesh = meshrender(opt.dataroot,int(tmp[0]), int(tmp[-1].split('_')[0]), rec_Amesh )


            # gt_Amesh = np.ascontiguousarray(gt_Amesh, dtype=np.uint8)
            # gt_Amesh = util.writeText(gt_Amesh, batch['A_path'][0], 100)

            # visuals = OrderedDict([
            #     ('gt_Amesh', gt_Amesh),
            #     ('rec_Amesh', rec_Amesh),
            
            #     ])
            # visualizer.display_current_results(visuals, num, 1000000)
    print (sum(loss)/len(loss))
print ('++++++++++++ SUCCESS ++++++++++++++++!')

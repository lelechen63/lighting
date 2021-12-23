import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import random
from os import path as osp
import pickle
import pytorch_lightning as pl
from data.data import FacescapeDataModule
from options.img2code_train_options import TrainOptions

from pytorch_lightning.callbacks import ModelCheckpoint

import util.util as util
import os
from util.visualizer import Visualizer

from util.render_class import meshrender
import numpy as np
from model.meshnetwork import *
from util import mesh_sampling
import  legacy
import dnnlib

opt = TrainOptions().parse()

if opt.debug:
    opt.nThreads = 1


if  opt.name == 'img2meshcode':
    from model.img2codeModel import Image2MeshcodeModule as module
    opt.datasetname = 'fs_code'
elif opt.name == 'MeshEncoderDecoder':
    from model.img2codeModel import MeshEncodeDecodeModule as module
    opt.datasetname = 'fs_mesh'
elif  opt.name == 'img2texcode':
    from model.img2codeModel import Image2TexcodeModule as module
    opt.datasetname = 'fs_code'

dm = FacescapeDataModule(opt)

totalmeanmesh = torch.FloatTensor( np.load( "./predef/meanmesh.npy" ) )#.view(-1,3) 
totalstdmesh = torch.FloatTensor(np.load( "./predef/meshstd.npy" ))#.view(-1,3)

meantex = torch.FloatTensor(np.load('./predef/meantex.npy')).permute(2, 0,1)
stdtex = torch.FloatTensor(np.load('./predef/stdtex.npy')).permute(2,0,1)

if opt.isTrain:
    print ( opt.name)
    model = module(opt)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath= os.path.join(opt.checkpoints_dir, opt.name),
        filename= opt.name +  '-{epoch:02d}-{train_loss:.2f}'
    )

    if len( opt.gpu_ids ) == 1:
        trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=1,  max_epochs= 100000, progress_bar_refresh_rate=20)
    else:
        # trainer = pl.Trainer(callbacks=[checkpoint_callback], precision=16,gpus= len( opt.gpu_ids ), accelerator='ddp', max_epochs= 100000, progress_bar_refresh_rate=20)
        trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=len( opt.gpu_ids ), accelerator='dp', max_epochs= 10000, progress_bar_refresh_rate=20)

    trainer.fit(model, dm)

else:
    print ('!!!!!!' + opt.name +'!!!!!!!!')
    if opt.name == 'MeshEncoderDecoder':
        
        device = torch.device('cuda', 0)
        Encoder = torch.load('./checkpoints/MeshEncoderDecoder/encoder.pth')
        Decoder = torch.load('./checkpoints/MeshEncoderDecoder/decoder.pth')

        dm.setup()
        testdata = dm.test_dataloader()
        opt.name = opt.name + '_test'
        visualizer = Visualizer(opt)
        l2loss = torch.nn.MSELoss()
        Encoder = Encoder.to(device)
        Decoder = Decoder.to(device)
        loss = []
        for num,batch in enumerate(testdata):
            if num == 100:
                break
            
            code = Encoder( batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).to(device))
            rec_mesh_A = Decoder(code)
            loss_mesh = l2loss(rec_mesh_A.cpu(), batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).detach() )
            print (loss_mesh)
            loss.append(loss_mesh)
            tmp = batch['A_path'][0].split('/')
            gt_mesh = batch['Amesh'].data[0].cpu() * totalstdmesh + totalmeanmesh
            rec_Amesh = rec_mesh_A.data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh
            gt_mesh = gt_mesh.float()
            rec_Amesh = rec_Amesh.float()
            gt_Amesh = meshrender( opt.dataroot, int(tmp[0]), int(tmp[-1].split('_')[0]),gt_mesh )
            rec_Amesh = meshrender(opt.dataroot,int(tmp[0]), int(tmp[-1].split('_')[0]), rec_Amesh )
            gt_Amesh = np.ascontiguousarray(gt_Amesh, dtype=np.uint8)
            gt_Amesh = util.writeText(gt_Amesh, batch['A_path'][0], 100)
            visuals = OrderedDict([
                ('gt_Amesh', gt_Amesh),
                ('rec_Amesh', rec_Amesh),
            
                ])
            visualizer.display_current_results(visuals, num, 1000000)
        print (sum(loss)/len(loss))

    elif opt.name == 'img2meshcode' :
        device = torch.device('cuda', 0)
        ImgEncoder = torch.load('./checkpoints/img2meshcode/ImageEncoder.pth')
        MeshCodeDecoder = torch.load('./checkpoints/img2meshcode/meshcode_dec.pth')
        Decoder = torch.load('./checkpoints/MeshEncoderDecoder/decoder.pth')

        dm.setup()
        testdata = dm.test_dataloader()
        opt.name = opt.name + '_test'
        visualizer = Visualizer(opt)
        l2loss = torch.nn.MSELoss()
        ImgEncoder = ImgEncoder.to(device)
        MeshCodeDecoder = MeshCodeDecoder.to(device)
        Decoder = Decoder.to(device)
        loss = []
        for num,batch in enumerate(testdata):
            if num == 100:
                break
            
            img_fea = ImgEncoder( batch['image'].to(device))
            img_fea = img_fea.view(img_fea.shape[0], -1)
        
            fakecode = MeshCodeDecoder(img_fea)
            loss_code = l2loss(fakecode.cpu(), batch['meshcode'].detach() )
            rec_mesh = Decoder(fakecode)
            rec_Amesh = rec_mesh.data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh

            rec_mesh_gt = Decoder(batch['meshcode'].to(device))
            rec_mesh_gt = rec_mesh_gt.data[0].cpu().view(-1) * totalstdmesh + totalmeanmesh

            loss_mesh = l2loss(rec_Amesh, batch['mesh'] )
            print ("loss_mesh: ", loss_mesh, "  loss_code", loss_code)
            loss.append(loss_mesh)
            tmp = batch['A_path'][0].split('/')
            gt_mesh = batch['mesh'].data[0].cpu() 
            
            gt_mesh = gt_mesh.float()
            rec_Amesh = rec_Amesh.float()
            rec_mesh_gt = rec_mesh_gt.float()
            gt_Amesh = meshrender( opt.dataroot, int(tmp[0]), int(tmp[-1].split('_')[0]),gt_mesh )
            rec_Amesh = meshrender(opt.dataroot,int(tmp[0]), int(tmp[-1].split('_')[0]), rec_Amesh )
            rec_mesh_gt = meshrender(opt.dataroot,int(tmp[0]), int(tmp[-1].split('_')[0]), rec_mesh_gt )
            gt_Amesh = np.ascontiguousarray(gt_Amesh, dtype=np.uint8)
            # gt_Amesh = util.writeText(gt_Amesh, batch['A_path'][0], 100)
            visuals = OrderedDict([
                ('rec_Amesh', rec_Amesh),
                ('rec_mesh_gt', rec_mesh_gt),
                ('gt_Amesh', gt_Amesh),
                ])
            visualizer.display_current_results(visuals, num, 1000000)
        print (sum(loss)/len(loss))
    
    elif opt.name == 'img2texcode' :
        device = torch.device('cuda', 0)
        ImgEncoder = torch.load('./checkpoints/img2texcode/ImageEncoder.pth')
        TexCodeDecoder = torch.load('./checkpoints/img2texcode/texturecode_dec.pth')
        with dnnlib.util.open_url('/home/uss00022/lelechen/github/stylegannerf/checkpoints/00012-target-face256/network-snapshot-002400.pkl') as fp:
            Decoder = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False) # type: ignore

        dm.setup()
        testdata = dm.test_dataloader()
        opt.name = opt.name + '_test'
        visualizer = Visualizer(opt)
        l2loss = torch.nn.MSELoss()
        ImgEncoder = ImgEncoder.to(device)
        TexCodeDecoder = TexCodeDecoder.to(device)
        Decoder = Decoder.to(device)
        loss = []
        for num,batch in enumerate(testdata):
            if num == 100:
                break
            
            img_fea = ImgEncoder( batch['image'].to(device))
            img_fea = img_fea.view(img_fea.shape[0], -1)
        
            fakecode = TexCodeDecoder(img_fea)
            loss_code = l2loss(fakecode.cpu(), batch['texcode'].detach() )
            fakecode = fakecode.repeat(14,1)

            fake_tex = Decoder.synthesis(fakecode.unsqueeze(0), noise_mode='const')
            fake_tex = (fake_tex + 1) * (255/2)
            fake_tex = fake_tex.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu()
            
            print (fake_tex.shape)
            print ( batch['tex'][0].shape)
            loss_tex = l2loss(fake_tex, batch['tex'][0] )
 
            fake_tex = fake_tex.numpy()
            
            rec_tex = Decoder.synthesis(batch['texcode'].repeat(14,1).unsqueeze(0).to(device), noise_mode='const')
            rec_tex = (rec_tex + 1) * (255/2)
            rec_tex = rec_tex.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            
            
            print ("loss_tex: ", loss_tex, "  loss_code", loss_code)
            loss.append(loss_mesh)
            tmp = batch['A_path'][0].split('/')
            gt_tex = batch['tex'].data[0].cpu().numpy()
            
          
            visuals = OrderedDict([
                ('fake_tex', fake_tex),
                ('rec_tex', rec_tex),
                ('gt_tex', gt_tex),
                ])
            visualizer.display_current_results(visuals, num, 1000000)
        print (sum(loss)/len(loss))
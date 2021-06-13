import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
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
opt = TrainOptions().parse()
if opt.name == 'tex' :
    from model.model2 import TexModule as module 
if opt.name == 'mesh' :
    from model.model2 import MeshModule as module 
elif opt.name == 'texmesh':
    from model.model2 import TexMeshModule as module 
elif opt.name == 'texmeshreal':
    from model.model import TexMeshModule as module 
elif opt.name == 'texgan':
    from model.model2 import TexGANModule as module 
opt.datasetname = "fs_texmesh"




dm = FacescapeDataModule(opt)

if opt.isTrain:
    model = module(opt)
    print ( opt.gpu_ids)
    if opt.no_vgg_loss:
        opt.name += '_novgg'
    if opt.no_cls_loss:
        opt.name += '_nocls'
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath= os.path.join(opt.checkpoints_dir, opt.name),
        filename='texmesh-{epoch:02d}-{train_loss:.2f}'
    )

    if len( opt.gpu_ids ) == 1:
        trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=1,  max_epochs= 10000, progress_bar_refresh_rate=20)

    else:
        trainer = pl.Trainer(callbacks=[checkpoint_callback], precision=16,gpus= len( opt.gpu_ids ), accelerator='ddp', max_epochs= 10000, progress_bar_refresh_rate=20)
    # trainer = pl.Trainer(gpus=4, accelerator='dp', max_epochs= 10000, progress_bar_refresh_rate=20)

    

    trainer.fit(model, dm)

else:
    print ('!!!!!!' + opt.name +'~!!!!!!!!')
    if opt.name == 'tex' :
        pass
    elif opt.name =='texmesh':
        checkpoint_path = '/data/home/us000042/lelechen/github/lighting/lightning_logs/version_30/checkpoints/epoch=720-step=152851.ckpt'
        
        from model.model2 import TexMeshGenerator as module 
    elif opt.name =='mesh':
        from model.model2 import MeshGenerator as module 
        checkpoint_path = '/data/home/us000042/lelechen/github/lighting/lightning_logs/version_58/checkpoints/epoch=2116-step=673205.ckpt'

    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    print (checkpoint.keys())

    module =  module(opt.loadSize, not opt.no_linearity, 3, opt.code_n,opt.encoder_fc_n, opt.ngf,opt.n_downsample_global, opt.n_blocks_global,opt.norm)

    def pl2normal(checkpoint):
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'discriminator' in k:
                continue
            
            name = k[10:]
            new_state_dict[name] = v
        return new_state_dict
    
    if opt.name =='mesh':
        module.load_state_dict(checkpoint['state_dict'])

    elif opt.name =='texmesh':
        module.load_state_dict(pl2normal(checkpoint['state_dict']))

    # model = model.load_from_checkpoint(checkpoint)
    # trainer = pl.Trainer()
    # results = trainer.test(model=model, datamodule = dm, verbose=True)

    dm.setup()
    testdata = dm.test_dataloader()
    
    opt.name = opt.name + '_test'


    visualizer = Visualizer(opt)


    for num,batch in enumerate(testdata):
        if opt.name == 'texmesh':
            rec_tex_A, rec_mesh_A, rec_tex_B, rec_mesh_B, \
            rec_tex_AB, rec_mesh_AB, rec_tex_BA, rec_mesh_BA = \
            module(  batch['Atex'], batch['Amesh'],batch['Btex'],batch['Bmesh'] )
            Atex = util.tensor2im(batch['Atex'][0])
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
            
            Btex = util.tensor2im(batch['Btex'][0])
            Btex = np.ascontiguousarray(Btex, dtype=np.uint8)
            Btex = util.writeText(Btex, batch['B_path'][0])

            tmp = batch['A_path'][0].split('/')
            gt_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),batch['Amesh'].data[0] )
            rec_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]), rec_mesh_A.data[0])

            tmp = batch['B_path'][0].split('/')
            gt_Bmesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),batch['Bmesh'].data[0] )
            rec_Bmesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]), rec_mesh_B.data[0])


            visuals = OrderedDict([
                ('Atex', Atex),
                ('Btex', Btex),
                ('rec_tex_A', util.tensor2im(rec_tex_A.data[0])),
                ('rec_tex_B', util.tensor2im(rec_tex_B.data[0])),
                ('rec_tex_AB', util.tensor2im(rec_tex_AB.data[0])),
                ('rec_tex_BA', util.tensor2im(rec_tex_BA.data[0])),
                ('gt_Amesh', gt_Amesh),
                ('rec_Amesh', rec_Amesh),
                ('gt_Bmesh', gt_Bmesh),
                ('rec_Bmesh', rec_Bmesh)
            
                ])
        
        elif opt.name =='mesh':
            rec_mesh_A = module(   batch['Amesh'] )

            tmp = batch['A_path'][0].split('/')
            gt_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),batch['Amesh'].data[0] )
            rec_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]), rec_mesh_A.data[0])

            visuals = OrderedDict([
                ('gt_Amesh', gt_Amesh),
                ('rec_Amesh', rec_Amesh),
            
                ])


        visualizer.display_current_results(visuals, num, 1000000)
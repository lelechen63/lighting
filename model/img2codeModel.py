import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
from .blocks import *
from model import lossNet
import torchvision
from collections import OrderedDict
import util.util as util
import os
from os import path as osp
from util.visualizer import Visualizer
from util.render_class import meshrender
import numpy as np
from model.meshnetwork import *
from util import mesh_sampling

from model.conv import ChebConv
from .inits import reset
from torch_scatter import scatter_add
from PIL import Image
import cv2
import pickle

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class MeshEncodeDecodeModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        homepath = './predef'
        device = torch.device('cuda', int(opt.gpu_ids[0]))

        template_fp = osp.join(homepath, 'meshmean.obj')
        transform_fp = osp.join(homepath, 'transform.pkl')
        if not osp.exists(transform_fp):
            print('Generating transform matrices...')
            mesh = Mesh(filename=template_fp)
            ds_factors = [4, 4, 4, 4]
            _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
            tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}

            with open(transform_fp, 'wb') as fp:
                pickle.dump(tmp, fp)
            print('Done!')
            print('Transform matrices are saved in \'{}\''.format(transform_fp))
        else:
            with open(transform_fp, 'rb') as f:
                tmp = pickle.load(f, encoding='latin1')

        edge_index_list = [util.to_edge_index(adj).to(device) for adj in tmp['adj']]
        # edge_index_list = [util.to_edge_index(adj) for adj in tmp['adj']]

        down_transform_list = [
            util.to_sparse(down_transform).to(device)
            # util.to_sparse(down_transform)
            for down_transform  in tmp['down_transform']
        ]
        up_transform_list = [
            util.to_sparse(up_transform).to(device)
            # util.to_sparse(up_transform)
            for up_transform in tmp['up_transform']
        ]

        self.l2loss = torch.nn.MSELoss()
        self.Encoder = MeshEncoder(3,
                [16, 16, 16, 32],
                256,
                edge_index_list,
                down_transform_list,
                up_transform_list,
                K=6)

        self.Decoder = MeshDecoder(3,
                [16, 16, 16, 32],
                256,
                edge_index_list,
                down_transform_list,
                up_transform_list,
                K=6)
        
        self.visualizer = Visualizer(opt)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    
    def forward(self, A_mesh):
        z = self.Encoder(A_mesh)
        x = self.Decoder(z)
        return x, z

    def training_step(self, batch, batch_idx):
        # generate images
        rec_mesh_A, code = self(batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3))
        # regularization
        loss_code = ( code ** 2 ).mean()
        # mesh loss
        loss_mesh = self.l2loss(rec_mesh_A, batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).detach() )
        loss = loss_mesh + loss_code * 0.1
        tqdm_dict = {'loss_mesh': loss_mesh, "loss_code" :loss_code }
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()}            
        self.visualizer.print_current_errors(self.current_epoch, batch_idx, errors, 0)
        self.visualizer.plot_current_errors(errors, batch_idx)
        return output
          
    def configure_optimizers(self):
        lr = self.opt.lr
        opt_g = torch.optim.Adam((list(self.Encoder.parameters()) + list(self.Decoder.parameters())), lr=lr, betas=(self.opt.beta1, 0.999))
        def lr_foo(epoch):
            lr_scale = 0.95 ** int(epoch/10)
            if lr_scale < 0.08:
                lr_scale = 0.08
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_foo)
        return [opt_g], [scheduler]
    
    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            # print ('!!!!!save model')
            # self.trainer.save_checkpoint( os.path.join( self.ckpt_path, '%05d.ckpt'%self.current_epoch) )
            torch.save(self.Encoder, os.path.join( self.ckpt_path, 'encoder.pth'))
            torch.save(self.Decoder, os.path.join( self.ckpt_path, 'decoder.pth'))


class Img2MeshCodeModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        ngf = 64
        code_n = 256

        self.ImageEncoder = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), 
            nn.ReLU(True),  

            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  # 2

            nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  #4

            nn.Conv2d(ngf*2 , ngf  * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 4),
            nn.ReLU(True), # 8

            nn.Conv2d(ngf*4 , ngf  * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 4),
            nn.ReLU(True), # 16

            nn.Conv2d(ngf*4 , ngf  * 8, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 8),
            nn.ReLU(True),  #32

            nn.Conv2d(ngf*8 , ngf  * 8, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 8),
            nn.ReLU(True),  #64

            nn.Conv2d(ngf*8 , ngf  * 16, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 16),
            nn.ReLU(True),  #128
        )

        self.enc_input_size = int(ngf * 16 * self.tex_shape/128 * self.tex_shape/128  + ngf * 4)

        self.meshcode_dec = nn.Sequential(
            nn.Linear( self.enc_input_size, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4,code_n),
            )

        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.resblocks = nn.Sequential(*model)
        self.visualizer = Visualizer(opt)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)


    def forward(self, image ):
        img_fea = self.ImageEncoder(image)
        x = self.resblocks(img_fea)
        x = x.view(x.shape[0], -1)
        
        code = self.meshcode_dec(x)
        return code
    
    def training_step(self, batch, batch_idx):

        # generate images
        code = self( batch['image'] )
        # regularization
        # loss_code = ( code ** 2 ).mean()
        # mesh loss
        loss = self.l2loss(code, batch['meshcode'].detach() )

        tqdm_dict = { "loss" :loss }

        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()}            
        self.visualizer.print_current_errors(self.current_epoch, batch_idx, errors, 0)
        self.visualizer.plot_current_errors(errors, batch_idx)
        return output
          
    def configure_optimizers(self):
        lr = self.opt.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(self.opt.beta1, 0.999))
        def lr_foo(epoch):
            lr_scale = 0.95 ** int(epoch/10)
            if lr_scale < 0.08:
                lr_scale = 0.08
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR( opt_g, lr_lambda=lr_foo )
        return [opt_g], [scheduler]
    
    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            # print ('!!!!!save model')
            # self.trainer.save_checkpoint( os.path.join( self.ckpt_path, '%05d.ckpt'%self.current_epoch) )
            torch.save(self.Decoder, os.path.join( self.ckpt_path, 'image2mesh.pth'))
            


import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
from .blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock, ResnetBlock
from model import lossNet
import torchvision
from collections import OrderedDict
import util.util as util
import os
from os import path as osp
from util.visualizer import Visualizer
from util.render_class import meshrender
from model.meshnetwork import AE
from util import  mesh_sampling
import pickle
# pickle.dump(some_object)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class TexMeshEncoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer= nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()
        self.tex_shape = tex_shape
        activation = nn.ReLU(True)

        # print (tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
        #         ngf, n_downsampling, n_blocks, norm_layer, \
        #         padding_type )
        # print('!!!!!!!!!!!!!!!')
        
        self.CNNencoder = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), 
            nn.ReLU(True),  
            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  # 2

            # nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
            # norm_layer(ngf  * 2),
            # nn.ReLU(True),  #4

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
            # nn.Conv2d(ngf*16 , ngf  * 16, kernel_size=3, stride=2, padding=1),
            # norm_layer(ngf  * 16),
            # nn.ReLU(True),  #4
        )
        self.meshencoder = nn.Sequential(
            nn.Linear( 78951, ngf*2),
            nn.ReLU(True),
            nn.Linear( ngf*2, ngf*2),
            nn.ReLU(True),
            nn.Linear( ngf*2, ngf*2),
            nn.ReLU(True),
            nn.Linear( ngf*2, ngf*2),
            nn.ReLU(True),
            nn.Linear( ngf*2, ngf*4),
            nn.ReLU(True)
            )

        self.enc_input_size = int(ngf * 16 * self.tex_shape/128 * self.tex_shape/128  + ngf * 4)
        # self.identity_enc = nn.Sequential(
        #     nn.Linear( self.enc_input_size, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4,code_n),
        #     nn.ReLU(True),
        #     )

        # self.expression_enc = nn.Sequential(
        #     nn.Linear( self.enc_input_size, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4,code_n),
        #     nn.ReLU(True),
        #     )
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.resblocks = nn.Sequential(*model)
    def forward(self, tex, mesh):
        tex_encoded = self.CNNencoder(tex)
        tex_encoded = self.resblocks(tex_encoded).view(tex_encoded.shape[0], -1)
        mesh_encoded = self.meshencoder(mesh)
        # encoded= torch.cat([mesh_encoded, tex_encoded], 1)
        
        # identity_code = self.identity_enc(encoded)
        # expression_code = self.expression_enc(encoded)
        # return identity_code, expression_code
        return tex_encoded, mesh_encoded



class TexEncoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer= nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()
        self.tex_shape = tex_shape
        activation = nn.ReLU(True)

        # print (tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
        #         ngf, n_downsampling, n_blocks, norm_layer, \
        #         padding_type )
        # print('!!!!!!!!!!!!!!!')
        
        self.CNNencoder = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), 
            nn.ReLU(True),  
            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  # 2

            # nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
            # norm_layer(ngf  * 2),
            # nn.ReLU(True),  #4

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
     
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.resblocks = nn.Sequential(*model)
    def forward(self, tex):
        tex_encoded = self.CNNencoder(tex)
        tex_encoded = self.resblocks(tex_encoded).view(tex_encoded.shape[0], -1)
        return tex_encoded


class TexDecoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer = nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()

        self.tex_shape = tex_shape
        activation = nn.ReLU(True)   
      
        self.tex_fc_dec = nn.Sequential(
            nn.Linear( ngf*4 * 2, ngf*16 * 4 * 4),
            nn.ReLU(True)
            )
     
        self.tex_decoder = nn.Sequential(
            # nn.ConvTranspose2d(ngf * 16, ngf * 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # norm_layer(ngf * 16), 
            # nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 8), 
            nn.ReLU(True), #2

            # nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            # norm_layer(ngf * 8), 
            # nn.ReLU(True), #4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 4), 
            nn.ReLU(True), #8

            nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 4), 
            nn.ReLU(True), #16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2), 
            nn.ReLU(True), #32

            nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2), 
            nn.ReLU(True), #64

            nn.ConvTranspose2d(ngf * 2, ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf), 
            nn.ReLU(True), #128
        )
        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)

    def forward(self, tex_code):
       
        tex_dec = tex_code.view(tex_code.shape[0], -1, 4,4) # not sure 

        decoded = self.tex_decoder(tex_dec)
        rec_tex = self.output_layer(decoded)
        return rec_tex   


class MeshEncoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer= nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()
        self.tex_shape = tex_shape
        activation = nn.LeakyReLU(0.2, True)  
        self.meshencoder = nn.Sequential(
            nn.Linear( 78951, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*2),
            activation
        )
        # self.idenc = nn.Sequential(
        #     nn.Linear( ngf*2, ngf*2),
        #     activation,
        #     nn.Linear( ngf*2, ngf*4),
        #     activation
        #     )
        self.expenc = nn.Sequential(
            nn.Linear( ngf*2, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*4),
            activation
            )

    def forward(self, mesh):
        mesh_encoded = self.meshencoder(mesh)
        # encoded_id = self.idenc(mesh_encoded)
        encoded_exp = self.expenc(mesh_encoded)
        return encoded_exp
        # return encoded_id, encoded_exp
class MeshDecoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer = nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()

        self.tex_shape = tex_shape
        activation = nn.LeakyReLU(0.2, True)     
        

        self.id_dex = nn.Sequential(
            nn.Linear( ngf*4, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*2),
            activation,
            nn.Linear( ngf*2, ngf*2),
            activation,
            nn.Linear( ngf*2, 78951)
        )
        # self.exp_dex = nn.Sequential(
        #     nn.Linear( ngf*8, ngf*4),
        #     activation,
        #     nn.Linear( ngf*4, ngf*4),
        #     activation,
        #     nn.Linear( ngf*4, ngf*4),
        #     activation,
        #     nn.Linear( ngf*4, 78951)
        # )
    def forward(self, exp_code):
    # def forward(self, id_code, exp_code):
        id_rec = self.id_dex(exp_code)
        return id_rec
        # exp_rec = self.exp_dex(torch.cat([id_code, exp_code],  1))
        # rec_mesh = id_rec + exp_rec
        # return id_rec, rec_mesh


class MeshGenerator(nn.Module):
    def __init__(self, tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer='batch', \
                padding_type='reflect'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer)  

        self.meshEnc = MeshEncoder(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)

        self.meshDec = MeshDecoder(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)
    def forward(self,  A_mesh ):
        
            # idcode, expcode = self.meshEnc( A_mesh)
            expcode = self.meshEnc( A_mesh)
            # reconstruction
            recmesh = self.meshDec( expcode)
            return recmesh

class MeshModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        input_nc = 3
        # self.totalmeanmesh = torch.FloatTensor( np.load( "./predef/meanmesh.npy" ) ) 

        # networks
        self.generator = MeshGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        if not opt.no_cls_loss:
            self.CLSloss = lossNet.CLSLoss(opt)

        self.visualizer = Visualizer(opt)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    

    def forward(self, A_mesh):
        return self.generator(A_mesh)
    
    def training_step(self, batch, batch_idx):
        # self.batch = batch
        # train generator
        # generate images
        rec_mesh_A = \
        self(batch['Amesh'])
        map_type = batch['map_type']

        loss_mesh = 0

        # id loss
        loss_id = 0 # self.l2loss(idmesh, batch['Aidmesh'] )
        # mesh loss
        loss_final = self.l2loss(rec_mesh_A, batch[ 'Amesh' ].detach() )
        loss = loss_final
        # loss = loss_id + loss_final
        tqdm_dict = {'loss_final': loss_final }

        # tqdm_dict = { 'loss_id': loss_id, 'loss_final': loss_final }
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
        # return [opt_g]
        def lr_foo(epoch):
            # if epoch < 10:
            #     lr_scale = 0.8 ** (10 - epoch)
            # else:
            lr_scale = 0.95 ** int(epoch/10)
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_foo )
                        

        return [opt_g], [scheduler]
    


    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            print ('!!!!!save model')
            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, '%05d.ckpt'%self.current_epoch) )
        self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )


class GraphConvMeshModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        input_nc = 3
        homepath = './predef'
        device = torch.device('cuda', 0)

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

        down_transform_list = [
            util.to_sparse(down_transform).to(device)
            for down_transform in tmp['down_transform']
        ]
        up_transform_list = [
            util.to_sparse(up_transform).to(device)
            for up_transform in tmp['up_transform']
        ]

        self.generator = AE(3,
                [16, 16, 16, 32],
                8,
                edge_index_list,
                down_transform_list,
                up_transform_list,
                K=6)
        # print(model)


        # networks
        # self.generator = MeshGenerator(opt.loadSize, not opt.no_linearity, 
        #     input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
        #     opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        if not opt.no_cls_loss:
            self.CLSloss = lossNet.CLSLoss(opt)

        self.visualizer = Visualizer(opt)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    

    def forward(self, A_mesh):
        
        return self.generator(A_mesh)
    
    def training_step(self, batch, batch_idx):
        # self.batch = batch
        # train generator
        # generate images
        print (batch['Amesh'].shape)
        rec_mesh_A = \
        self(batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3))
        map_type = batch['map_type']

        loss_mesh = 0

        # id loss
        loss_id = 0 # self.l2loss(idmesh, batch['Aidmesh'] )
        # mesh loss
        loss_final = self.l2loss(rec_mesh_A, batch[ 'Amesh' ].detach() )
        loss = loss_final
        # loss = loss_id + loss_final
        tqdm_dict = {'loss_final': loss_final }

        # tqdm_dict = { 'loss_id': loss_id, 'loss_final': loss_final }
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
        # return [opt_g]
        def lr_foo(epoch):
            # if epoch < 10:
            #     lr_scale = 0.8 ** (10 - epoch)
            # else:
            lr_scale = 0.95 ** int(epoch/10)
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_foo )
                        

        return [opt_g], [scheduler]
    


    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            print ('!!!!!save model')
            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, '%05d.ckpt'%self.current_epoch) )
        self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )


class  TexMeshDecoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer = nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()

        self.tex_shape = tex_shape
        activation = nn.ReLU(True)   
        # self.identity_dec = nn.Sequential(
        #     nn.Linear( code_n, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4, ngf*4),
        #     nn.ReLU(True),
        #     # nn.Linear( ngf*4, ngf*4),
        #     # nn.ReLU(True),
        #     nn.Linear( ngf*4,ngf*4),
        #     nn.ReLU(True),
        #     )
        # self.exp_dec = nn.Sequential(
        #     nn.Linear( code_n, ngf*4),
        #     nn.ReLU(True),
        #     nn.Linear( ngf*4, ngf*4),
        #     nn.ReLU(True),
        #     # nn.Linear( ngf*4, ngf*4),
        #     # nn.ReLU(True),
        #     nn.Linear( ngf*4,ngf*4),
        #     nn.ReLU(True),
        #     )
        self.tex_fc_dec = nn.Sequential(
            nn.Linear( ngf*4 * 2, ngf*16 * 4 * 4),
            nn.ReLU(True)
            )
        self.mesh_fc_dec = nn.Sequential(
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, 78951),
            )
        ### upsample

        self.tex_decoder = nn.Sequential(
            # nn.ConvTranspose2d(ngf * 16, ngf * 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # norm_layer(ngf * 16), 
            # nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 8), 
            nn.ReLU(True), #2

            # nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            # norm_layer(ngf * 8), 
            # nn.ReLU(True), #4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 4), 
            nn.ReLU(True), #8

            nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 4), 
            nn.ReLU(True), #16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2), 
            nn.ReLU(True), #32

            nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2), 
            nn.ReLU(True), #64

            nn.ConvTranspose2d(ngf * 2, ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf), 
            nn.ReLU(True), #128
        )
        model = []
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)

    # def forward(self, id_code, exp_code):
    #     exp_fea = self.exp_dec(exp_code)
    #     id_fea = self.identity_dec(id_code)
    #     feature = torch.cat([exp_fea, id_fea], axis = 1)
    #     rec_mesh = self.mesh_fc_dec(feature)

    #     tex_dec = self.tex_fc_dec(feature)
    #     # tex_dec = tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, int(self.tex_shape / 128),int(self.tex_shape / 128)) # not sure 
        
    #     tex_dec = tex_dec.view(tex_dec.shape[0], -1, 4,4) # not sure 

    #     decoded = self.tex_decoder(tex_dec)
    #     rec_tex = self.output_layer(decoded)
    #     return rec_tex, rec_mesh

    def forward(self, tex_code, mesh_code):
        # exp_fea = self.exp_dec(exp_code)
        # id_fea = self.identity_dec(id_code)
        # feature = torch.cat([exp_fea, id_fea], axis = 1)
        rec_mesh = self.mesh_fc_dec(mesh_code)

        # tex_dec = self.tex_fc_dec(tex_code)
        # tex_dec = tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, int(self.tex_shape / 128),int(self.tex_shape / 128)) # not sure 
        
        tex_dec = tex_code.view(tex_code.shape[0], -1, 4,4) # not sure 

        decoded = self.tex_decoder(tex_dec)
        rec_tex = self.output_layer(decoded)
        return rec_tex, rec_mesh   



class TexGenerator(nn.Module):
    def __init__(self, tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer='batch', \
                padding_type='reflect'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer)  

        self.texEnc = TexEncoder(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)

        self.texDec = TexDecoder(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)
    def forward(self, A_tex ):
        
        tex_code = self.texEnc(A_tex)

        # reconstruction
        rec_tex_A = self.texDec(tex_code)
        return rec_tex_A

class TexMeshGenerator(nn.Module):
    def __init__(self, tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer='batch', \
                padding_type='reflect'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer)  

        self.texmeshEnc = TexMeshEncoder(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)

        self.texmeshDec = TexMeshDecoder(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)
    def forward(self, A_tex, A_mesh, B_tex = None, B_mesh = None ):
        if B_tex is not None:
            # A_tex, A_mesh, B_tex, B_mesh = input_lists[0], input_lists[1], input_lists[2], input_lists[3]
            A_id_code, A_exp_code = self.texmeshEnc(A_tex, A_mesh)
            B_id_code, B_exp_code = self.texmeshEnc(B_tex, B_mesh)

            # reconstruction
            rec_tex_A, rec_mesh_A = self.texmeshDec(A_id_code, A_exp_code)
            rec_tex_B, rec_mesh_B = self.texmeshDec(B_id_code, B_exp_code)

            # mismatch
            rec_tex_AB, rec_mesh_AB = self.texmeshDec(A_id_code, B_exp_code)
            rec_tex_BA, rec_mesh_BA = self.texmeshDec(B_id_code, A_exp_code)

            return rec_tex_A, rec_mesh_A, rec_tex_B, rec_mesh_B, rec_tex_AB, rec_mesh_AB, rec_tex_BA, rec_mesh_BA
        else:
            # A_tex, A_mesh = input_lists[0], input_lists[1]
            tex_code, mesh_code = self.texmeshEnc(A_tex, A_mesh)

            # reconstruction
            rec_tex_A, rec_mesh_A = self.texmeshDec(tex_code, mesh_code)
            return rec_tex_A, rec_mesh_A

class TexMeshModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.opt = opt
        input_nc = 3
        # networks
        self.generator = TexMeshGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        if not opt.no_vgg_loss:             
            self.VGGloss = lossNet.VGGLoss()
        if not opt.no_cls_loss:
            self.CLSloss = lossNet.CLSLoss(opt)

        self.GANloss = lossNet.GANLoss()
        self.visualizer = Visualizer(opt)
        # self.meshrender = MeshRender()

        # if len(gpu_ids) and torch.cuda.is_available():
        #     network.cuda()


    def forward(self, A_tex, A_mesh, B_tex, B_mesh):
        return self.generator(A_tex, A_mesh)
    
    def training_step(self, batch, batch_idx):
        self.batch = batch
        # train generator
        # generate images
        rec_tex_A, rec_mesh_A = \
        self(batch['Atex'], batch['Amesh'],batch['Btex'],batch['Bmesh'])
        map_type = batch['map_type']

        # VGG loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG += self.VGGloss(rec_tex_A, batch['Atex']) * self.opt.lambda_feat
        
        # CLS loss
        loss_G_CLS = 0
        if not self.opt.no_cls_loss:
            loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aid'] , 'id') * self.opt.lambda_cls
            loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aexp'] , 'exp') * self.opt.lambda_cls

        # pix loss
        loss_G_pix = 0
        # reconstruction loss
        if not self.opt.no_pix_loss:
            loss_G_pix += self.l1loss(rec_tex_A, batch['Atex']) * self.opt.lambda_pix

        #mesh loss
        loss_mesh = 0
        if not self.opt.no_mesh_loss:
            loss_mesh += self.l1loss(rec_mesh_A, batch['Amesh'])* self.opt.lambda_mesh
            # mismatch loss
    

        loss = loss_G_pix + loss_G_VGG + loss_G_CLS + loss_mesh 
        tqdm_dict = {'loss_pix': loss_G_pix, 'loss_G_VGG': loss_G_VGG, 'loss_G_CLS': loss_G_CLS, 'loss_mesh': loss_mesh}
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
            if epoch < 10:
                lr_scale = 0.8 ** (10 - epoch)
            else:
                lr_scale = 0.95 ** int(epoch/10)
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt_g,
            lr_lambda=lr_foo
        )

        return [opt_g], [scheduler]
    
    # def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
    #     if self.trainer.global_step > 30:
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = 0.8 * self.opt.lr
    #     optimizer.step()
    #     optimizer.zero_grad()

    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            batch = self.batch
            rec_tex_A, rec_mesh_A = \
            self(batch['Atex'], batch['Amesh'],batch['Btex'],batch['Bmesh'])

            Atex = util.tensor2im(batch['Atex'][0])
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
            # tmp = batch['A_path'][0].split('/')
            # gg = batch['Amesh'].data[0].cpu()
            # gg = gg.numpy()
            # gg = torch.from_numpy(gg.astype(np.float32))
            # gt_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),gg )
            
            # gg =rec_mesh_A.data[0].cpu()
            # gg = gg.numpy()
            # gg = torch.from_numpy(gg.astype(np.float32))

            # rec_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),gg)
            visuals = OrderedDict([
            ('Atex', Atex),
            # ('Amesh', gt_Amesh),
            ('rec_tex_A', util.tensor2im(rec_tex_A.data[0])),
            # ('rec_Amesh', rec_Amesh)
        
            ])
       
            self.visualizer.display_current_results(visuals, self.current_epoch, 1000000) 


class TexModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.opt = opt
        input_nc = 3
        self.totalmeantex = np.load( "./predef/meantex.npy" )
        self.save_hyperparameters()

        self.generator = TexGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        
        if not opt.no_vgg_loss:             
            self.VGGloss = lossNet.VGGLoss()
        if not opt.no_cls_loss:
            self.CLSloss = lossNet.CLSLoss(opt)

        self.visualizer = Visualizer(opt)


    def forward(self, A_tex):
        return self.generator(A_tex)
    
    def training_step(self, batch, batch_idx):
        self.batch = batch
        # train generator
        # generate images
        rec_tex_A =  self(batch['Atex'] )
        map_type = batch['map_type']

        # VGG loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG += self.VGGloss(rec_tex_A, batch['Atex']) * self.opt.lambda_feat
        
        # CLS loss
        loss_G_CLS = 0
        if not self.opt.no_cls_loss:
            loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aid'] , 'id') * self.opt.lambda_cls
            loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aexp'] , 'exp') * self.opt.lambda_cls

        # pix loss
        loss_G_pix = 0
        # reconstruction loss
        if not self.opt.no_pix_loss:
            loss_G_pix += self.l1loss(rec_tex_A, batch['Atex']) * self.opt.lambda_pix

    
        loss_mesh = 0
        loss = loss_G_pix + loss_G_VGG + loss_G_CLS  
        tqdm_dict = {'loss_pix': loss_G_pix, 'loss_G_VGG': loss_G_VGG, 'loss_G_CLS': loss_G_CLS, 'loss_mesh': loss_mesh}
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
            if epoch < 10:
                lr_scale = 0.8 ** (10 - epoch)
            else:
                lr_scale = 0.95 ** int(epoch/10)
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt_g,
            lr_lambda=lr_foo
        )

        return [opt_g], [scheduler]
    
    # def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
    #     if self.trainer.global_step > 30:
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = 0.8 * self.opt.lr
    #     optimizer.step()
    #     optimizer.zero_grad()

    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            batch = self.batch
            rec_tex_A =  self(batch['Atex'])

            Atex = util.tensor2im(batch['Atex'][0])
            
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
          
            visuals = OrderedDict([
            ('Atex', Atex),
            ('rec_tex_A', util.tensor2im(rec_tex_A.data[0]))
            ])
       
            self.visualizer.display_current_results(visuals, self.current_epoch, 1000000) 



class TexGANModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.opt = opt
        input_nc = 3
        # networks
        self.generator = TexGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.discriminator = MultiscaleDiscriminator(input_nc = 6)   

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        if not opt.no_vgg_loss:             
            self.VGGloss = lossNet.VGGLoss()
        if not opt.no_cls_loss:
            self.CLSloss = lossNet.CLSLoss(opt)

        self.GANloss = lossNet.GANLoss()
        self.visualizer = Visualizer(opt)
        self.totalmeantex = np.load( "./predef/meantex.npy" )
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    def forward(self, A_tex):
        return self.generator(A_tex)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch = batch
        # train generator
        # generate images
        rec_tex_A = \
        self(batch['Atex'])
        map_type = batch['map_type']

        if optimizer_idx ==0:       
            # VGG loss
            loss_G_VGG = 0
            if not self.opt.no_vgg_loss:
                loss_G_VGG += self.VGGloss(rec_tex_A, batch['Atex']) * self.opt.lambda_feat
            
            # CLS loss
            loss_G_CLS = 0
            if not self.opt.no_cls_loss:
                loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aid'] , 'id') * self.opt.lambda_cls
                loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aexp'] , 'exp') * self.opt.lambda_cls

            # pix loss
            loss_G_pix = 0
            # reconstruction loss
            if not self.opt.no_pix_loss:
                loss_G_pix += self.l1loss(rec_tex_A, batch['Atex']) * self.opt.lambda_pix

            loss_mesh = 0 
            g_loss = self.GANloss(self.discriminator(  torch.cat((batch['Atex'], rec_tex_A), dim=1) ), True)

            loss = loss_G_pix + loss_G_VGG + loss_G_CLS + loss_mesh + g_loss
            tqdm_dict = {'loss_pix': loss_G_pix, 'loss_G_VGG': loss_G_VGG, 'loss_G_CLS': loss_G_CLS, 'loss_mesh': loss_mesh, 'loss_GAN': g_loss }
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()}            
            self.visualizer.print_current_errors(self.current_epoch, batch_idx, errors, 0)
            self.visualizer.plot_current_errors(errors, batch_idx)
            return output
        if optimizer_idx == 1:

            real_loss = self.GANloss(self.discriminator( torch.cat((batch['Atex'], batch['Atex']), dim=1)), True)
            fake_loss = self.GANloss( self.discriminator( torch.cat((batch['Atex'], rec_tex_A.detach() ), dim = 1) ), False)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
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
        
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.opt.beta1, 0.999))

        return [opt_g, opt_d], []

    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            batch = self.batch
            rec_tex_A = \
            self(batch['Atex'])

            Atex = util.tensor2im(batch['Atex'][0])
            
            # print (Atex.shape)
            # print (self.totalmeantex.shape)

            # Atex = Atex + self.totalmeantex
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)

            Atex = util.writeText(Atex, batch['A_path'][0])

            rec_tex_A_vis = util.tensor2im(rec_tex_A.data[0])
            # rec_tex_A_vis = rec_tex_A_vis + self.totalmeantex
            # rec_tex_A_vis = np.ascontiguousarray(rec_tex_A_vis, dtype=np.uint8)
            # rec_tex_A_vis = np.clip(rec_tex_A_vis, 0, 255)
            visuals = OrderedDict([
            ('Atex', Atex),
            ('rec_tex_A', rec_tex_A_vis ),
        
            ])
       
            self.visualizer.display_current_results(visuals, self.current_epoch, 1000000) 

            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=True, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class TexMeshGANModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.opt = opt
        input_nc = 3
        # networks
        self.generator = TexMeshGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.discriminator = MultiscaleDiscriminator(input_nc = 6)   

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        if not opt.no_vgg_loss:             
            self.VGGloss = lossNet.VGGLoss()
        if not opt.no_cls_loss:
            self.CLSloss = lossNet.CLSLoss(opt)

        self.GANloss = lossNet.GANLoss()
        self.visualizer = Visualizer(opt)

        # if len(gpu_ids) and torch.cuda.is_available():
        #     network.cuda()


    def forward(self, A_tex, A_mesh, B_tex, B_mesh):
        return self.generator(A_tex, A_mesh)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch = batch
        # train generator
        # generate images
        rec_tex_A, rec_mesh_A = \
        self(batch['Atex'], batch['Amesh'],batch['Btex'],batch['Bmesh'])
        map_type = batch['map_type']

        if optimizer_idx ==0:       
            # VGG loss
            loss_G_VGG = 0
            if not self.opt.no_vgg_loss:
                loss_G_VGG += self.VGGloss(rec_tex_A, batch['Atex']) * self.opt.lambda_feat
            
            # CLS loss
            loss_G_CLS = 0
            if not self.opt.no_cls_loss:
                loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aid'] , 'id') * self.opt.lambda_cls
                loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aexp'] , 'exp') * self.opt.lambda_cls

            # pix loss
            loss_G_pix = 0
            # reconstruction loss
            if not self.opt.no_pix_loss:
                loss_G_pix += self.l1loss(rec_tex_A, batch['Atex']) * self.opt.lambda_pix

            #mesh loss
            loss_mesh = 0
            if not self.opt.no_mesh_loss:
                loss_mesh += self.l1loss(rec_mesh_A, batch['Amesh'])* self.opt.lambda_mesh
                # mismatch loss
            
            
            g_loss = self.GANloss(self.discriminator(  torch.cat((batch['Atex'], rec_tex_A), dim=1) ), True)

            loss = loss_G_pix + loss_G_VGG + loss_G_CLS + loss_mesh + g_loss
            tqdm_dict = {'loss_pix': loss_G_pix, 'loss_G_VGG': loss_G_VGG, 'loss_G_CLS': loss_G_CLS, 'loss_mesh': loss_mesh, 'loss_GAN': g_loss }
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()}            
            self.visualizer.print_current_errors(self.current_epoch, batch_idx, errors, 0)
            self.visualizer.plot_current_errors(errors, batch_idx)
            return output
        if optimizer_idx == 1:

            real_loss = self.GANloss(self.discriminator( torch.cat((batch['Atex'], batch['Atex']), dim=1)), True)

            fake_loss = self.GANloss( self.discriminator( torch.cat((batch['Atex'], rec_tex_A.detach() ), dim = 1) ), False)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
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
        
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.opt.beta1, 0.999))

        return [opt_g, opt_d], []

    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            batch = self.batch
            rec_tex_A, rec_mesh_A = \
            self(batch['Atex'], batch['Amesh'],batch['Btex'],batch['Bmesh'])

            Atex = util.tensor2im(batch['Atex'][0])
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
            # tmp = batch['A_path'][0].split('/')
            # gg = batch['Amesh'].data[0].cpu()
            # gg = gg.numpy()
            # gg = torch.from_numpy(gg.astype(np.float32))
            # gt_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),gg )
            
            # gg =rec_mesh_A.data[0].cpu()
            # gg = gg.numpy()
            # gg = torch.from_numpy(gg.astype(np.float32))

            # rec_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),gg)
            visuals = OrderedDict([
            ('Atex', Atex),
            # ('Amesh', gt_Amesh),
            ('rec_tex_A', util.tensor2im(rec_tex_A.data[0])),
            # ('rec_Amesh', rec_Amesh)
        
            ])
       
            self.visualizer.display_current_results(visuals, self.current_epoch, 1000000) 



class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=True, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

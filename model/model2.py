
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
from util import  mesh_sampling
import pickle
from model.conv import ChebConv
from .inits import reset
from torch_scatter import scatter_add

# pickle.dump(some_object)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class TexEncoder2(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer= nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()
        self.tex_shape = tex_shape
        
        self.CNNencoder = nn.Sequential(
            Conv2dWNUB( 3, 32, 128, 128, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ), #2
            Conv2dWNUB( 32, 32, 64, 64, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),#4
            Conv2dWNUB( 32, 64, 32, 32, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),#8
            Conv2dWNUB( 64, 64, 16, 16, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),#16
            Conv2dWNUB( 64, 128, 8, 8, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),#32
            Conv2dWNUB( 128, 256, 4, 4, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),#64
        )

        self.codefc = nn.Sequential(
            LinearWN(256 * 4 * 4, 512 ), nn.LeakyReLU( 0.2, inplace = True )
            )
        self.codelast = LinearWN( 512, 256 )
        self.apply( lambda x: glorot( x, 0.2 ) )
        glorot( self.codelast, 1.0 )
    def forward(self, tex):
        tex_encoded = self.CNNencoder(tex).view(tex.shape[0], -1)
        tex_encoded  = self.codefc(tex_encoded)
        tex_encoded = self.codelast(tex_encoded)
        return tex_encoded


class TexDecoder2(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer = nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()

        self.tex_shape = tex_shape
        self.tex_fc_dec = nn.Sequential(
            LinearWN( 256 , 256 * 4 * 4 ), nn.LeakyReLU( 0.2, inplace = True ) )
        self.tex_decoder = nn.Sequential(
            ConvTranspose2dWNUB( 256, 256, 8, 8, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),
            ConvTranspose2dWNUB( 256, 128, 16, 16, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),
            ConvTranspose2dWNUB( 128, 64, 32, 32, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),
            ConvTranspose2dWNUB( 64, 32, 64, 64, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),
            ConvTranspose2dWNUB( 32, 8, 128, 128, 4, 2, 1 ), nn.LeakyReLU( 0.2, inplace = True ),
            ConvTranspose2dWNUB( 8, 3, 256, 256, 4, 2, 1 ) )
        self.apply( lambda x: glorot( x, 0.2 ) )
        glorot( self.tex_fc_dec[ -1 ], 1.0 )
    def forward(self, tex_code):
        tex_code = self.tex_fc_dec(tex_code)
        tex_dec = tex_code.view(tex_code.shape[0], 256, 4,4) # not sure 
        decoded = self.tex_decoder(tex_dec)
        return decoded   

class TexEncoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer= nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()
        self.tex_shape = tex_shape
        activation = nn.ReLU(True)
        
        self.CNNencoder = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), 
            nn.ReLU(True),  
            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  # 2

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

        self.codefc = nn.Sequential(
            nn.Linear( ngf * 16 * 4 * 4, ngf * 4),
            # nn.ReLU(True),
            # nn.Linear( ngf*16 , ngf * 4 ),
            # nn.ReLU(True),
            nn.Linear( ngf*4 , 256),
            nn.ReLU(True)
        )
    def forward(self, tex):
        tex_encoded = self.CNNencoder(tex)
        tex_encoded = self.resblocks(tex_encoded).view(tex_encoded.shape[0], -1)
        tex_encoded  = self.codefc(tex_encoded)
        return tex_encoded



class TexDecoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer = nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()

        self.tex_shape = tex_shape
        activation = nn.ReLU(True)   
      
        self.tex_fc_dec = nn.Sequential(
            nn.Linear( 256, ngf * 4),
            nn.ReLU(True),
            # nn.Linear( ngf*4 , ngf * 16),
            # nn.ReLU(True),
            nn.Linear( ngf*4, ngf*16 * 4 * 4),
            nn.ReLU(True)
            )
     
        self.tex_decoder = nn.Sequential(
        
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 8), 
            nn.ReLU(True), #2

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
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0)]   
        # model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]    
        self.output_layer = nn.Sequential(*model)

    def forward(self, tex_code):
        tex_code = self.tex_fc_dec(tex_code)
        tex_dec = tex_code.view(tex_code.shape[0], -1, 4,4) # not sure 

        decoded = self.tex_decoder(tex_dec)
        rec_tex = self.output_layer(decoded)
        return rec_tex   



class TexMeshEncoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer= nn.BatchNorm2d, \
                padding_type='reflect', edge_index_list = None, down_transform_list = None,\
                up_transform_list= None, K = 6 ):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer)  
        self.CNNencoder = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), 
            nn.ReLU(True),  
            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  # 2

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
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=nn.ReLU(True) , norm_layer=norm_layer)]
        self.resblocks = nn.Sequential(*model)

        self.codefc = nn.Sequential(
            nn.Linear( ngf * 16 * 4 * 4, ngf * 4),
            nn.Linear( ngf*4 , 256),
            nn.ReLU(True)
        )
        ## for mesh encoder        
        self.edge_index = edge_index_list
        self.down_transform = down_transform_list

        self.meshconv = nn.ModuleList([
            Enblock(3,16,K),
            Enblock(16,16,K),
            Enblock(16,16,K),
            Enblock(16,32,K)]
        )
        num_vert = down_transform_list[-1].size(0)
        self.meshfc = nn.Sequential(
            nn.Linear(num_vert * 32, 256),
            nn.ReLU(True)
        )
        self.fusefc =nn.Sequential(
            nn.Linear(512, 256)
        )
    def forward(self, tex, mesh):
        tex_encoded = self.CNNencoder(tex)
        tex_encoded = self.resblocks(tex_encoded).view(tex_encoded.shape[0], -1)
        tex_encoded  = self.codefc(tex_encoded)

        for i, layer in enumerate( self.meshconv):
            mesh = layer( mesh, self.edge_index[i], self.down_transform[i])
        mesh_encoded = mesh.view(mesh.shape[0], -1)
        mesh_encoded = self.meshfc(mesh_encoded)
        code = self.fusefc( torch.cat( [ tex_encoded, mesh_encoded], 1))
        
        return code


class TexMeshDecoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer= nn.BatchNorm2d, \
                padding_type='reflect', edge_index_list = None, down_transform_list = None,\
                up_transform_list= None, K = 6 ):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer)  
        self.tex_fc_dec = nn.Sequential(
            nn.Linear( 256, ngf * 4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*16 * 4 * 4),
            nn.ReLU(True)
            )
     
        self.tex_decoder = nn.Sequential(
        
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 8), 
            nn.ReLU(True), #2

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
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0)]   
        self.output_layer = nn.Sequential(*model)

        # mesh decoder 
        self.edge_index = edge_index_list
        self.up_transform = up_transform_list
        self.num_vert = down_transform_list[-1].size(0)
        self.meshfc = nn.Sequential(
            nn.Linear(256, self.num_vert * 32))

        self.meshconv = nn.ModuleList([
            Deblock(32,32,K),
            Deblock(32,16,K),
            Deblock(16,16,K),
            Deblock(16,16,K)]
        )
        self.meshlast =   ChebConv(16, 3, K)
        self.num_deblocks = len(self.meshconv) - 1
    def forward(self, code ):
        tex_code = self.tex_fc_dec(code)
        tex_dec = tex_code.view(tex_code.shape[0], -1, 4,4)
        decoded = self.tex_decoder(tex_dec)
        rec_tex = self.output_layer(decoded)

        mesh_code = self.meshfc(code).view(-1, self.num_vert, 32)

        for i,layer in enumerate(self.meshconv):
            mesh_code = layer(mesh_code, self.edge_index[self.num_deblocks - i],
                          self.up_transform[self.num_deblocks - i])
        rec_mesh = self.meshlast(mesh_code, self.edge_index[0])

        return rec_tex, rec_mesh

class GraphConvMeshModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
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
        self.l2loss = torch.nn.MSELoss()
        self.generator = AE(3,
                [16, 16, 16, 32],
                256,
                edge_index_list,
                down_transform_list,
                up_transform_list,
                K=6)
        self.visualizer = Visualizer(opt)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    def forward(self, A_mesh): 
        
        return self.generator(A_mesh)
    
    def training_step(self, batch, batch_idx):
        # self.batch = batch
        # train generator
        # generate images
        rec_mesh_A, code = \
        self(batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3)) 
        map_type = batch['map_type']

        loss_mesh = 0

        # regularization
        loss_code = ( code ** 2 ).mean()
        # id loss
        loss_id = 0 # self.l2loss(idmesh, batch['Aidmesh'] )
        # mesh loss
        loss_land = 0# self.l2loss(rec_mesh_A[:,self.land_inx], batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3)[:,self.land_inx].detach() ) 
        loss_mesh = self.l2loss(rec_mesh_A, batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).detach() )

        loss = loss_mesh + loss_code* 0.1
        # loss = loss_id + loss_final
        tqdm_dict = {'loss_mesh': loss_mesh, "loss_code" :loss_code }

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
            if lr_scale < 0.08:
                lr_scale = 0.08
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_foo )
                        

        return [opt_g], [scheduler]
    


    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            # print ('!!!!!save model')
            # self.trainer.save_checkpoint( os.path.join( self.ckpt_path, '%05d.ckpt'%self.current_epoch) )
            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )

class DisGraphConvMeshModule(pl.LightningModule):
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

        self.generator = DisAE(3,
                [16, 16, 16, 32],
                256,
                edge_index_list,
                down_transform_list,
                up_transform_list,
                K=6)
        
        land_tex = './predef/landmark_indices.txt'
        land_tex = open(land_tex, 'r')
        Lines = land_tex.readlines()
        self.land_inx = []
        for line in Lines:
            self.land_inx.append(int(line))

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        if not opt.no_cls_loss:
            self.CLSloss = lossNet.CLSLoss(opt)

        self.visualizer = Visualizer(opt)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    

    def forward(self, A_mesh, B_mesh):
        
        return self.generator(A_mesh, B_mesh)
    
    def training_step(self, batch, batch_idx):
        # self.batch = batch
        # train generator
        # generate images
        rec_mesh_A, rec_mesh_B, rec_mesh_AB, rec_mesh_BA, Aexp,Aid, Bexp, Bid = \
        self(batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3), batch['Bmesh'].view(batch['Bmesh'].shape[0], -1, 3)) 
        map_type = batch['map_type']

        loss_code = 0

        # regularization
        loss_code += ( Aexp ** 2  ).mean() +(Aid ** 2).mean()
        loss_code += ( Bexp ** 2  ).mean() +(Bid ** 2).mean()
        
        # mismatch loss
        loss_mis = 0
        print (rec_mesh_AB.shape, batch['Bmesh'].shape, )
        if not self.opt.no_mismatch_loss:
            for i in range(map_type.shape[0]):
                if map_type[i] == 0: # same id, diff exp, mismatch is decided by exp
                    loss_mis += self.l2loss(rec_mesh_AB[i].unsqueeze(0), batch['Bmesh'][i].unsqueeze(0).view(1, -1, 3).detach()) * self.opt.lambda_feat * self.opt.lambda_mismatch
                    loss_mis += self.l2loss(rec_mesh_BA[i].unsqueeze(0), batch['Amesh'][i].unsqueeze(0).view(1, -1, 3).detach()) * self.opt.lambda_feat* self.opt.lambda_mismatch
                else:
                    loss_mis += self.l2loss(rec_mesh_AB[i].unsqueeze(0), batch['Amesh'][i].unsqueeze(0).view(1, -1, 3).detach()) * self.opt.lambda_feat* self.opt.lambda_mismatch
                    loss_mis += self.l2loss(rec_mesh_BA[i].unsqueeze(0), batch['Bmesh'][i].unsqueeze(0).view(1, -1, 3).detach()) * self.opt.lambda_feat* self.opt.lambda_mismatch
            # reconstruction loss

        # mesh loss
        loss_mesh = 0
        loss_mesh += self.l2loss(rec_mesh_A, batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).detach() )
        loss_mesh += self.l2loss(rec_mesh_B, batch['Bmesh'].view(batch['Bmesh'].shape[0], -1, 3).detach() )

        loss = loss_mesh + loss_code* 0.1 + loss_mis 
        # loss = loss_id + loss_final
        tqdm_dict = {'loss_mesh': loss_mesh, "loss_code" :loss_code, "loss_miss" :loss_mis }

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
            lr_scale = 0.95 ** int(epoch/10)
            if lr_scale < 0.08:
                lr_scale = 0.08
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_foo )
        return [opt_g], [scheduler]

    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            # print ('!!!!!save model')
            # self.trainer.save_checkpoint( os.path.join( self.ckpt_path, '%05d.ckpt'%self.current_epoch) )
            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )

class DisGraphConvMeshModule2(pl.LightningModule):
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

        self.generator = DisAE2(3,
                [16, 16, 16, 32],
                256,
                edge_index_list,
                down_transform_list,
                up_transform_list,
                K=6)
        
        land_tex = './predef/landmark_indices.txt'
        land_tex = open(land_tex, 'r')
        Lines = land_tex.readlines()
        self.land_inx = []
        for line in Lines:
            self.land_inx.append(int(line))
        print(self.land_inx)
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
        rec_mesh_A, idA, Aexpcode, Aidcode = \
        self(batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3)) 
        map_type = batch['map_type']

        loss_code = 0

        # regularization
        loss_code += ( Aexpcode ** 2  ).mean() +(Aidcode ** 2).mean()
        
   
        # mesh loss
        loss_mesh = 0
        loss_mesh += self.l2loss(rec_mesh_A, batch['Amesh'].view(batch['Amesh'].shape[0], -1, 3).detach() )
        
        loss_mesh += self.l2loss(idA, batch['Aidmesh'].view(batch['Aidmesh'].shape[0], -1, 3).detach() )

        loss = loss_mesh + loss_code* 0.1 
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
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(self.opt.beta1, 0.999))
        # return [opt_g]
        def lr_foo(epoch):
            # if epoch < 10:
            #     lr_scale = 0.8 ** (10 - epoch)
            # else:
            lr_scale = 0.95 ** int(epoch/10)
            if lr_scale < 0.08:
                lr_scale = 0.08
            return lr_scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_foo )
        return [opt_g], [scheduler]
    


    def on_epoch_end(self):
        if self.current_epoch % 5 == 0:
            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )
  

class TexGenerator(nn.Module):
    def __init__(self, tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer='batch', \
                padding_type='reflect'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer)  

        self.texEnc = TexEncoder2(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)

        self.texDec = TexDecoder2(tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf, n_downsampling, n_blocks, norm_layer, padding_type)
    def forward(self, A_tex ):
        
        tex_code = self.texEnc(A_tex)

        # reconstruction
        rec_tex_A = self.texDec(tex_code)
        return rec_tex_A


class TexGANModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.opt = opt
        input_nc = 3
        # networks
        self.generator = TexGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.discriminator = MultiscaleDiscriminator(input_nc = 3)   

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
       
        self.GANloss = lossNet.GANLoss()
        self.visualizer = Visualizer(opt)
        self.meantex = np.load('./predef/meantex.npy')
        self.stdtex = np.load('./predef/stdtex.npy')
        self.meantex = torch.FloatTensor(self.meantex).permute(2, 0,1)
        self.stdtex = torch.FloatTensor(self.stdtex).permute(2,0,1)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    def forward(self, A_tex):
        return self.generator(A_tex)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch = batch
        # train generator
        # generate images
        rec_tex_A = \
        self(batch['Atex'])

        if optimizer_idx ==0:                  
          
            # pix loss
            loss_G_pix = 0
            # reconstruction loss
            if not self.opt.no_pix_loss:
                loss_G_pix += self.l1loss(rec_tex_A, batch['Atex']) * self.opt.lambda_pix

            loss_mesh = 0 
            g_loss = self.GANloss(self.discriminator(   rec_tex_A), True)

            loss = loss_G_pix    + loss_mesh + g_loss
            tqdm_dict = {'loss_pix': loss_G_pix, 'loss_mesh': loss_mesh, 'loss_GAN': g_loss }
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

            real_loss = self.GANloss(self.discriminator(  batch['Atex']), True)
            fake_loss = self.GANloss( self.discriminator( rec_tex_A.detach()) , False)

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

            
            # Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
           
            Atex = batch['Atex'].data[0].cpu()  * self.stdtex + self.meantex 
            Atex = util.tensor2im(Atex  , normalize = False)
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = np.clip(Atex, 0, 255)

            rec_tex_A_vis =rec_tex_A.data[0].cpu() * self.stdtex + self.meantex  
            rec_tex_A_vis = util.tensor2im(rec_tex_A_vis, normalize = False)            
            rec_tex_A_vis = np.ascontiguousarray(rec_tex_A_vis, dtype=np.uint8)
            rec_tex_A_vis = np.clip(rec_tex_A_vis, 0, 255)
            visuals = OrderedDict([
            ('Atex', Atex),
            ('rec_tex_A', rec_tex_A_vis ),
        
            ])
       
            self.visualizer.display_current_results(visuals, self.current_epoch, 1000000) 

            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )

class TexModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.opt = opt
        input_nc = 3
        # networks
        self.generator = TexGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
       
        self.visualizer = Visualizer(opt)
        self.meantex = np.load('./predef/meantex.npy')
        self.stdtex = np.load('./predef/stdtex.npy')
        self.meantex = torch.FloatTensor(self.meantex).permute(2, 0,1)
        self.stdtex = torch.FloatTensor(self.stdtex).permute(2,0,1)
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
    def forward(self, A_tex):
        return self.generator(A_tex)
    
    def training_step(self, batch, batch_idx):
        self.batch = batch
        # train generator
        # generate images
        rec_tex_A = \
        self(batch['Atex'])
          
        # pix loss
        loss_G_pix = 0
        # reconstruction loss
        print (rec_tex_A.max(), batch['Atex'].max(),rec_tex_A.min(), batch['Atex'].min())
        if not self.opt.no_pix_loss:
            loss_G_pix += self.l1loss(rec_tex_A, batch['Atex']) * self.opt.lambda_pix

      
        loss = loss_G_pix    
        tqdm_dict = {'loss_pix': loss_G_pix }
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
        

        return [opt_g], []

    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            batch = self.batch
            rec_tex_A = \
            self(batch['Atex'])

            
            # Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
           
            Atex = batch['Atex'].data[0].cpu()  * self.stdtex + self.meantex 
            Atex = util.tensor2im(Atex  , normalize = False)
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = np.clip(Atex, 0, 255)

            rec_tex_A_vis =rec_tex_A.data[0].cpu() * self.stdtex + self.meantex  
            rec_tex_A_vis = util.tensor2im(rec_tex_A_vis, normalize = False)            
            rec_tex_A_vis = np.ascontiguousarray(rec_tex_A_vis, dtype=np.uint8)
            rec_tex_A_vis = np.clip(rec_tex_A_vis, 0, 255)
            visuals = OrderedDict([
            ('Atex', Atex),
            ('rec_tex_A', rec_tex_A_vis ),
        
            ])
       
            self.visualizer.display_current_results(visuals, self.current_epoch, 1000000) 

            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )
class MeshTexGenerator(nn.Module):
    def __init__(self, tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer='batch', \
                padding_type='reflect'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer)  

        # define the mesh part
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

        self.texmeshEnc = TexMeshEncoder( tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer='batch', \
                padding_type='reflect', edge_index_list = edge_index_list, \
                 down_transform_list = down_transform_list,\
                up_transform_list= up_transform_list, K = 6)

        self.texmeshDec = TexMeshDecoder( tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer='batch', \
                padding_type='reflect', edge_index_list = edge_index_list, \
                 down_transform_list = down_transform_list,\
                up_transform_list= up_transform_list, K = 6)

        # self.reset_parameters()
    def reset_parameters(self):
        for name, param in self.named_parameters():
            # print (name)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

   
    def forward(self, A_tex, A_mesh ):
        # encode
        code = self.texmeshEnc(A_tex, A_mesh)

        # reconstruction
        rec_tex_A,rec_mesh_A  = self.texmeshDec(code)

        return rec_tex_A, rec_mesh_A, code

class MeshTexGANModule(pl.LightningModule):
    def __init__(self, opt ):
        super().__init__()
        self.opt = opt
        input_nc = 3
        # networks
        self.generator = MeshTexGenerator(opt.loadSize, not opt.no_linearity, 
            input_nc, opt.code_n,opt.encoder_fc_n, opt.ngf, 
            opt.n_downsample_global, opt.n_blocks_global,opt.norm)

        self.discriminator = MultiscaleDiscriminator(input_nc = 6)   

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        self.GANloss = lossNet.GANLoss()
        self.visualizer = Visualizer(opt)
        self.totalmeantex = np.load( "./predef/meantex.npy" )
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        
        self.meantex = np.load('/data/home/us000042/lelechen/github/lighting/predef/meantex.npy')
        self.stdtex = np.load('/data/home/us000042/lelechen/github/lighting/predef/stdtex.npy')
        self.meantex = torch.FloatTensor(self.meantex).permute(2, 0,1)
        self.stdtex = torch.FloatTensor(self.stdtex).permute(2,0,1)
        
    def forward(self, A_tex, A_mesh):
        return self.generator(A_tex, A_mesh)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch = batch
        # train generator
        # generate images
        rec_tex_A, rec_mesh_A, code = \
        self(batch['Atex'], batch['Amesh'] )
        map_type = batch['map_type']


        if optimizer_idx ==0:       
            
            # pix loss
            loss_G_pix = 0
            # reconstruction loss
            if not self.opt.no_pix_loss:
                loss_G_pix += self.l1loss(rec_tex_A, batch['Atex']) * self.opt.lambda_pix

            loss_code = 0

            # regularization
            loss_code = ( code ** 2 ).mean()
            
            loss_mesh = self.l2loss(rec_mesh_A, batch['Amesh'] )
            # gan loss
            g_loss = self.GANloss(self.discriminator(  torch.cat((batch['Atex'], rec_tex_A), dim=1) ), True)

            loss = loss_G_pix   + loss_mesh + g_loss + loss_code * 0.1
            tqdm_dict = {'loss_pix': loss_G_pix, 'loss_mesh': loss_mesh, 'loss_GAN': g_loss, 'loss_code': loss_code }
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
            rec_tex_A, _, _ = \
            self(batch['Atex'], batch['Amesh'] )

   
            Atex = batch['Atex'].data[0].cpu()  * self.stdtex + self.meantex 
            Atex = util.tensor2im(Atex  , normalize = False)
            
            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = util.writeText(Atex, batch['A_path'][0])

            Atex = np.ascontiguousarray(Atex, dtype=np.uint8)
            Atex = np.clip(Atex, 0, 255)

            rec_tex_A_vis =rec_tex_A.data[0].cpu() * self.stdtex + self.meantex  
            rec_tex_A_vis = util.tensor2im(rec_tex_A_vis, normalize = False)
            
            rec_tex_A_vis = np.ascontiguousarray(rec_tex_A_vis, dtype=np.uint8)
            rec_tex_A_vis = np.clip(rec_tex_A_vis, 0, 255)

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

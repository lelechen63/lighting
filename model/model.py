
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

# import pickle
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
        self.identity_enc = nn.Sequential(
            nn.Linear( self.enc_input_size, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4,code_n),
            nn.ReLU(True),
            )

        self.expression_enc = nn.Sequential(
            nn.Linear( self.enc_input_size, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4,code_n),
            nn.ReLU(True),
            )
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.resblocks = nn.Sequential(*model)
    def forward(self, tex, mesh):
        tex_encoded = self.CNNencoder(tex)
        tex_encoded = self.resblocks(tex_encoded).view(tex_encoded.shape[0], -1)
        mesh_encoded = self.meshencoder(mesh)
        encoded= torch.cat([mesh_encoded, tex_encoded], 1)
        identity_code = self.identity_enc(encoded)
        expression_code = self.expression_enc(encoded)
        return identity_code, expression_code


class TexMeshDecoder(nn.Module):
    def __init__(self,  tex_shape, linearity, input_nc, code_n, encoder_fc_n, \
                ngf=64, n_downsampling=5, n_blocks=4, norm_layer = nn.BatchNorm2d, \
                padding_type='reflect'):
        super().__init__()

        self.tex_shape = tex_shape
        activation = nn.ReLU(True)   
        self.identity_dec = nn.Sequential(
            nn.Linear( code_n, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            # nn.Linear( ngf*4, ngf*4),
            # nn.ReLU(True),
            nn.Linear( ngf*4,ngf*4),
            nn.ReLU(True),
            )
        self.exp_dec = nn.Sequential(
            nn.Linear( code_n, ngf*4),
            nn.ReLU(True),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(True),
            # nn.Linear( ngf*4, ngf*4),
            # nn.ReLU(True),
            nn.Linear( ngf*4,ngf*4),
            nn.ReLU(True),
            )
        self.tex_fc_dec = nn.Sequential(
            nn.Linear( ngf*4 * 2, ngf*16),
            nn.ReLU(True)
            )
        self.mesh_fc_dec = nn.Sequential(
            nn.Linear( ngf*4 * 2, ngf*4),
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

            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 8), 
            nn.ReLU(True), #4

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

    def forward(self, id_code, exp_code):
        exp_fea = self.exp_dec(exp_code)
        id_fea = self.identity_dec(id_code)
        feature = torch.cat([exp_fea, id_fea], axis = 1)
        rec_mesh = self.mesh_fc_dec(feature)

        tex_dec = self.tex_fc_dec(feature)
        tex_dec = tex_dec.unsqueeze(2).unsqueeze(3).repeat(1, 1, int(self.tex_shape / 128),int(self.tex_shape / 128)) # not sure 
         
        decoded = self.tex_decoder(tex_dec)
        rec_tex = self.output_layer(decoded)
        return rec_tex, rec_mesh


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
    def forward(self, A_tex, A_mesh, B_tex, B_mesh):
        A_id_code, A_exp_code = self.texmeshEnc(A_tex, A_mesh)
        B_id_code, B_exp_code = self.texmeshEnc(B_tex, B_mesh)

        # reconstruction
        rec_tex_A, rec_mesh_A = self.texmeshDec(A_id_code, A_exp_code)
        rec_tex_B, rec_mesh_B = self.texmeshDec(B_id_code, B_exp_code)

        # mismatch
        rec_tex_AB, rec_mesh_AB = self.texmeshDec(A_id_code, B_exp_code)
        rec_tex_BA, rec_mesh_BA = self.texmeshDec(B_id_code, A_exp_code)

        return rec_tex_A, rec_mesh_A, rec_tex_B, rec_mesh_B, rec_tex_AB, rec_mesh_AB, rec_tex_BA, rec_mesh_BA


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


    def forward(self, A_tex, A_mesh, B_tex, B_mesh):
        return self.generator(A_tex, A_mesh, B_tex, B_mesh)

    def training_step(self, batch, batch_idx):

        # train generator
        # generate images
        rec_tex_A, rec_mesh_A, rec_tex_B, rec_mesh_B, \
        rec_tex_AB, rec_mesh_AB, rec_tex_BA, rec_mesh_BA = \
        self(batch['Atex'], batch['Amesh'],batch['Btex'],batch['Bmesh'])
        map_type = batch['map_type']

        # log sampled images
        sample_imgs = rec_tex_A[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, 0)

        # ground truth result (ie: all fake)
       
        # VGG loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            if not self.opt.no_mismatch_loss:
                for i in range(map_type.shape[0]):
                    if map_type[i] == 0: # same id, diff exp, mismatch is decided by exp
                        loss_G_VGG += self.VGGloss(rec_tex_AB[i].unsqueeze(0), batch['Btex'][i].unsqueeze(0)) * self.opt.lambda_feat * self.opt.lambda_mismatch
                        loss_G_VGG += self.VGGloss(rec_tex_BA[i].unsqueeze(0), batch['Atex'][i].unsqueeze(0)) * self.opt.lambda_feat* self.opt.lambda_mismatch
                    else:
                        loss_G_VGG += self.VGGloss(rec_tex_AB[i].unsqueeze(0), batch['Atex'][i].unsqueeze(0)) * self.opt.lambda_feat* self.opt.lambda_mismatch
                        loss_G_VGG += self.VGGloss(rec_tex_BA[i].unsqueeze(0), batch['Btex'][i].unsqueeze(0)) * self.opt.lambda_feat* self.opt.lambda_mismatch
            # reconstruction loss
            
            loss_G_VGG += self.VGGloss(rec_tex_A, batch['Atex']) * self.opt.lambda_feat
            loss_G_VGG +=  self.VGGloss(rec_tex_B, batch['Btex']) * self.opt.lambda_feat
        
        # CLS loss
        loss_G_CLS = 0
        if not self.opt.no_cls_loss:
            if not self.opt.no_mismatch_loss:
                if map_type[i] == 0: # same id, diff exp, mismatch is decided by exp
                    loss_G_CLS += self.CLSloss(rec_tex_AB, batch['Aid'] ,'id' ) * self.opt.lambda_cls* self.opt.lambda_mismatch
                    loss_G_CLS += self.CLSloss(rec_tex_BA, batch['Bid'] ,'id' ) * self.opt.lambda_cls* self.opt.lambda_mismatch

                    loss_G_CLS += self.CLSloss(rec_tex_AB, batch['Bexp'] ,'exp' ) * self.opt.lambda_cls* self.opt.lambda_mismatch
                    loss_G_CLS += self.CLSloss(rec_tex_BA, batch['Aexp'] ,'exp' ) * self.opt.lambda_cls* self.opt.lambda_mismatch
            # reconstruction loss
            loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aid'] , 'id') * self.opt.lambda_cls
            loss_G_CLS += self.CLSloss(rec_tex_A,  batch['Aexp'] , 'exp') * self.opt.lambda_cls

            loss_G_CLS += self.CLSloss(rec_tex_B,  batch['Bid'] , 'id') * self.opt.lambda_cls
            loss_G_CLS += self.CLSloss(rec_tex_B,  batch['Bexp'] , 'exp') * self.opt.lambda_cls

        # pix loss
        loss_G_pix = 0
        # mismatch loss
        if not self.opt.no_mismatch_loss:
            for i in range(map_type.shape[0]):
                if map_type[i] == 0: # same id, diff exp, mismatch is decided by exp
                    loss_G_pix += self.l1loss(rec_tex_AB[i].unsqueeze(0), Btex[i].unsqueeze(0)) * self.opt.lambda_pix* self.opt.lambda_mismatch
                    loss_G_pix += self.l1loss(rec_tex_BA[i].unsqueeze(0), Atex[i].unsqueeze(0)) * self.opt.lambda_pix* self.opt.lambda_mismatch
                else:
                    loss_G_pix += self.l1loss(rec_tex_AB[i].unsqueeze(0), Atex[i].unsqueeze(0)) * self.opt.lambda_pix* self.opt.lambda_mismatch
                    loss_G_pix += self.l1loss(rec_tex_BA[i].unsqueeze(0), Btex[i].unsqueeze(0)) * self.opt.lambda_pix* self.opt.lambda_mismatch
        
        # reconstruction loss
        loss_G_pix += self.l1loss(rec_tex_A, Atex) * self.opt.lambda_pix
        loss_G_pix += self.l1loss(rec_tex_B, Btex) * self.opt.lambda_pix

        #mesh loss
        loss_mesh = 0
        loss_mesh += self.l1loss(rec_mesh_A, Amesh)* self.opt.lambda_mesh
        loss_mesh += self.l1loss(rec_mesh_B, Bmesh)* self.opt.lambda_mesh
        # mismatch loss
        if not self.opt.no_mismatch_loss:
            for i in range(map_type.shape[0]):
                if map_type[i] == 0: # same id, diff exp, mismatch is decided by exp
                    loss_mesh += self.l1loss(rec_mesh_AB[i].unsqueeze(0), Bmesh[i].unsqueeze(0)) * self.opt.lambda_mesh* self.opt.lambda_mismatch
                    loss_mesh += self.criterionPix(rec_mesh_BA[i].unsqueeze(0), Amesh[i].unsqueeze(0)) * self.opt.lambda_mesh* self.opt.lambda_mismatch
                else:
                    loss_mesh += self.l1loss(rec_mesh_AB[i].unsqueeze(0), Amesh[i].unsqueeze(0)) * self.opt.lambda_mesh* self.opt.lambda_mismatch
                    loss_mesh += self.criterionPix(rec_mesh_BA[i].unsqueeze(0), Bmesh[i].unsqueeze(0)) * self.opt.lambda_mesh* self.opt.lambda_mismatch
        
        # adversarial loss is binary cross-entropy
        g_loss = loss_G_pix + loss_G_VGG + loss_G_CLS + loss_mesh
        tqdm_dict = {'loss_pix': loss_G_pix, 'loss_G_VGG': loss_G_VGG, 'loss_G_CLS': loss_G_CLS, 'loss_mesh': loss_mesh }
        output = OrderedDict({
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

        
    def configure_optimizers(self):
        lr = self.opt.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(self.opt.beta1, 0.999))
        return [opt_g], []

    # def on_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)

    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


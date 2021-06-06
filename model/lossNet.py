from torchvision import models
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19()#.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class CLSLoss(nn.Module):
    def __init__(self, opt):
        super(CLSLoss, self).__init__()        
        self.idcls = TexClassifier(opt.loadSize, 301, 64, opt.n_downsample_global, opt.n_blocks_global)
        self.expcls = TexClassifier(opt.loadSize, 20, 64, opt.n_downsample_global, opt.n_blocks_global)
        # self.expcls = self.expcls.cuda()
        # self.idcls = self.idcls.cuda()
        
        self.idcls.load_state_dict(torch.load('/raid/celong/lele/github/render2real/checkpoints/cls/200_net_idcls.pth'))
        self.expcls.load_state_dict(torch.load('/raid/celong/lele/github/render2real/checkpoints/cls/200_net_expcls.pth'))
        
        for param in self.idcls.parameters():
            param.requires_grad = False
        for param in self.expcls.parameters():
            param.requires_grad = False


        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tex, gt_lab, mode):
        device_id = tex.device.index

        if mode == 'id':
            out_lab = self.idcls(tex)
        else:
            out_lab = self.expcls(tex)        
        loss = self.criterion(out_lab, gt_lab.detach())
        return loss


class TexClassifier(nn.Module):
    def __init__(self, tex_size, output_nc, ngf=64, n_downsampling=5, n_blocks=4, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(TexClassifier, self).__init__()        
        activation = nn.ReLU(True)        

        self.CNNencoder = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0),
            norm_layer(ngf), 
            nn.ReLU(True),  

            nn.Conv2d(ngf , ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  # 512

            nn.Conv2d( ngf * 2, ngf  * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 2),
            nn.ReLU(True),  #256

            nn.Conv2d(ngf*2 , ngf  * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 4),
            nn.ReLU(True), # 128

            nn.Conv2d(ngf*4 , ngf  * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 4),
            nn.ReLU(True), # 64

            nn.Conv2d(ngf*4 , ngf  * 8, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 8),
            nn.ReLU(True),  #32

            nn.Conv2d(ngf*8 , ngf  * 8, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 8),
            nn.ReLU(True),  #16

            nn.Conv2d(ngf*8 , ngf  * 16, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf  * 16),
            nn.ReLU(True),  #8

                        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear( ngf * 16 * 4, ngf*4),
            nn.ReLU(),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(),
            nn.Linear( ngf*4, ngf*4),
            nn.ReLU(),
            nn.Linear( ngf*4,output_nc)
                                    )

    def forward(self, tex ):
        fea = self.CNNencoder(tex)
        label = self.fc_layer(fea.view(fea.shape[0], -1))
        return label
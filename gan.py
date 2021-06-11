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
from model.model2 import TexMeshModule as module 
# from model.model2 import TexMeshGANModule as module 

from pytorch_lightning.callbacks import ModelCheckpoint



opt = TrainOptions().parse()
opt.datasetname = "fs_texmesh"
# opt.name = "texmesh_step1_real" 


dm = FacescapeDataModule(opt)
model = module(opt)
print ( opt.gpu_ids)
# trainer = pl.Trainer(gpus= opt.gpu_ids, max_epochs= 200, progress_bar_refresh_rate=20)
# if opt.debug:
# trainer = pl.Trainer(gpus=1,  max_epochs= 10000, progress_bar_refresh_rate=20)

# else:
if len( opt.gpu_ids ) == 1:
    trainer = pl.Trainer(gpus=1,  max_epochs= 10000, progress_bar_refresh_rate=20)

else:
    trainer = pl.Trainer(precision=16,gpus= len( opt.gpu_ids ), accelerator='ddp', max_epochs= 10000, progress_bar_refresh_rate=20)
# trainer = pl.Trainer(gpus=4, accelerator='dp', max_epochs= 10000, progress_bar_refresh_rate=20)

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath= os.path.join(opt.checkpoints_dir, opt.name),
    filename='texmesh-{epoch:02d}-{train_loss:.2f}',
)

trainer.fit(model, dm)
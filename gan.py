import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import MNIST

import pytorch_lightning as pl
from data.data import FacescapeDataModule
from options.step1_train_options import TrainOptions
from model.model import TexMeshModule

from util.visualizer import Visualizer

opt = TrainOptions().parse()
opt.datasetname = "fs_texmesh"
opt.name = "texmesh_step1" 
dm = FacescapeDataModule(opt)
model = TexMeshModule(opt)
print ( opt.gpu_ids)
# trainer = pl.Trainer(gpus= opt.gpu_ids, max_epochs= 200, progress_bar_refresh_rate=20)
trainer = pl.Trainer(gpus=8, accelerator='ddp', max_epochs= 200, progress_bar_refresh_rate=20)


# trainer = pl.Trainer(gpus= 1, max_epochs= 200, progress_bar_refresh_rate=20)

trainer.fit(model, dm)
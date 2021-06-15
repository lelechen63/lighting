
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from PIL import Image

class FacescapeDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()
        self.batch_size = opt.batchSize
        self.num_workers = int(opt.nThreads)

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()

    def prepare_data(self):
        pass
    def get_dataset(self):
        dataset = None
        from data.facescape import FacescapeMeshTexDataset
        dataset = FacescapeMeshTexDataset(self.opt)
        # if self.opt.datasetname == 'fs':
        #     from data.facescape import FacescapeDataset
        #     dataset = FacescapeDataset(self.opt)
        # elif self.opt.datasetname == 'fs_pair':
        #     from data.facescape import FacescapeDirDataset
        #     dataset = FacescapeDirDataset(self.opt)
        # elif self.opt.datasetname == 'fs_texmesh':
        #     from data.facescape import FacescapeMeshTexDataset
        #     dataset = FacescapeMeshTexDataset(self.opt)
        # elif self.opt.datasetname == 'fs_tex':
        #     from data.facescape import FacescapeTexDataset
        #     dataset = FacescapeTexDataset(self.opt)
        print("dataset [%s] was created" % (dataset.name()))
        print ('=================================')
        # dataset.initialize(opt)
        return dataset
        # (self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        self.dataset = self.get_dataset()        

    def train_dataloader(self):
        print ('############ train dataloader ###################')
        return DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)



import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import torch_geometric.transforms as T
from psbody.mesh import Mesh
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from meshnetwork import AE
# from datasets import MeshData
from util import util, mesh_sampling #, writer, train_eval, DataLoader, 

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

model = AE(3,
           [16, 16, 16, 32],
           8,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=6).to(device)
print(model)

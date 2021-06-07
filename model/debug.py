import sys
import json
import numpy as np
import trimesh
import imageio
import openmesh
import cv2

import pyredner
import redner
import math
import pickle
import os

import torch
from tqdm import tqdm
import glob

from skimage.transform import AffineTransform, warp

sys.path.insert(1, '/raid/celong/lele/github/lighting')

from util.render_class import MeshRender
expressions = {
    1: "1_neutral",
    2: "2_smile",
    3: "3_mouth_stretch",
    4: "4_anger",
    5: "5_jaw_left",
    6: "6_jaw_right",
    7: "7_jaw_forward",
    8: "8_mouth_left",
    9: "9_mouth_right",
    10: "10_dimpler",
    11: "11_chin_raiser",
    12: "12_lip_puckerer",
    13: "13_lip_funneler",
    14: "14_sadness",
    15: "15_lip_roll",
    16: "16_grin",
    17: "17_cheek_blowing",
    18: "18_eye_closed",
    19: "19_brow_raiser",
    20: "20_brow_lower"
}

ms = MeshRender()


id_idx = 140
exp_idx = 4
cam_idx = 1
mesh_root = "/raid/celong/FaceScape/textured_meshes"

mesh_path = f"{mesh_root}/{id_idx}/models_reg/{expressions[exp_idx]}.obj"

om_mesh = openmesh.read_trimesh(mesh_path)
om_vertices = np.array(om_mesh.points()).reshape(-1)
om_vertices = torch.from_numpy(om_vertices.astype(np.float32))

img = ms.meshrender(id_idx, exp_idx, om_vertices)

imageio.imwrite("fkass.png", img)
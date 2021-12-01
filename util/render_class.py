import json
import numpy as np
import trimesh
import imageio
import openmesh
import cv2
import redner

import pyredner
import math
import pickle
import os

import torch
from tqdm import tqdm
import glob
from skimage.transform import AffineTransform, warp
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
with open("./predef/Rt_scale_dict.json", 'r') as f:
    Rt_scale_dict = json.load(f)
pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_print_timing(False)
om_indices = np.load("./predef/om_indices.npy")
om_indices = torch.from_numpy(om_indices).type(torch.int32).to(pyredner.get_device())
image_data_root = "/data/home/us000042/lelechen/data/Facescape/jsons"

def shift(image, vector):
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    return shifted.astype(image.dtype)

def meshrender(data_root, id_idx, exp_idx, vertices, cam_idx=1):
    """
    # id_idx: int
    # exp_idx: int
    # vertices: [3*VN] (openmesh ordering, float32 tensor)
    # cam_idx: 1
    # return: rendered image, [h,w,3]
    """
    scale = Rt_scale_dict['1']['%d'%1][0]
    Rt_TU = np.array(Rt_scale_dict['%d'%1]['%d'%1][1])
    # Rt_TU = torch.from_numpy(Rt_TU).type(torch.float32).to(pyredner.get_device())
    Rt_TU = torch.from_numpy(Rt_TU).type(torch.float32).to("cuda:0")

 
    input_vertices = vertices.reshape(-1,3).to("cuda:0")

    # input_vertices = vertices.reshape(-1,3).to(pyredner.get_device())
    input_vertices = (Rt_TU[:3,:3].T @ (input_vertices - Rt_TU[:3,3]).T).T
    input_vertices = input_vertices / scale
    input_vertices = input_vertices.contiguous()

    m = pyredner.Material(diffuse_reflectance = torch.tensor((0.5, 0.5, 0.5), device = pyredner.get_device()))
    # print ('m', m.device)

    obj = pyredner.Object(vertices=input_vertices, indices=om_indices, material=m)
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices)

    # obj.normals = pyredner.compute_vertex_normal(obj.vertices.to(pyredner.get_device()), obj.indices.to(pyredner.get_device())).cpu()
    with open( os.path.join(data_root, 'jsons/1/1_neutral/params.json') , 'r') as f:
        params = json.load(f)
    # img_dir = f"{image_data_root}/{1}/{expressions[1]}"
    # with open(f"{img_dir}/params.json", 'r') as f:
    #     params = json.load(f)

    K = np.array(params['%d_K' % cam_idx])
    Rt = np.array(params['%d_Rt' % cam_idx])
    # dist = np.array(params['%d_distortion' % cam_idx], dtype = float)
    h_src = params['%d_height' % cam_idx]
    w_src = params['%d_width' % cam_idx]

    cx = K[0,2]
    cy = K[1,2]
    dx = cx - 0.5 * w_src
    dy = cy - 0.5 * h_src
    dx = int(dx)
    dy = int(dy)

    c2w = np.eye(4)
    c2w[:3,:3] = Rt[:3,:3].T
    c2w[:3,3] = -Rt[:3,:3].T @ Rt[:3,3]
    c2w = torch.from_numpy(c2w).type(torch.float32)
    K = torch.from_numpy(K).type(torch.float32)

    K[0,2] = 0
    K[1,2] = 0
    K[0,0] = K[0,0] * 2.0 / w_src
    K[1,1] = -K[1,1] * 2.0 / w_src

    # Setup camera
    cam = pyredner.Camera(
        cam_to_world= c2w,
        intrinsic_mat=K,
        clip_near = 1e-2, # needs to > 0
        resolution = (h_src, w_src),
        # distortion_params=distortion,
        camera_type=pyredner.camera_type.perspective,
        fisheye = False
    )
    
    light_dir = torch.tensor([[0.0, 0.0, 1.0]])
    light_dir = (c2w[:3,:3]@light_dir.T).T
    light_dir = light_dir.to("cuda:0")
    # light_dir = light_dir.to(pyredner.get_device())
    lights = [
        pyredner.DirectionalLight(light_dir, torch.tensor([5.0, 5.0, 5.0], device = pyredner.get_device()))
    ]
    
    scene = pyredner.Scene(camera=cam, objects=[obj])
    img = pyredner.render_deferred(scene, lights=lights)

    img = torch.pow(img, 1.0/2.2).cpu().numpy()
    img = shift(img, [-dx, -dy])

    img = img* 255
    img = img.astype(np.uint8)
    return img

# import json
# import numpy as np
# import trimesh
# import imageio
# import openmesh
# import cv2

# import pyredner
# import redner
# import math
# import pickle
# import os

# import torch
# from tqdm import tqdm
# import glob
# from skimage.transform import AffineTransform, warp
# # os.environ["CUDA_VISIBLE_DEVICES"]="0"

# class MeshRender():
#     def __init__(self):
#         self.expressions = {
#             1: "1_neutral",
#             2: "2_smile",
#             3: "3_mouth_stretch",
#             4: "4_anger",
#             5: "5_jaw_left",
#             6: "6_jaw_right",
#             7: "7_jaw_forward",
#             8: "8_mouth_left",
#             9: "9_mouth_right",
#             10: "10_dimpler",
#             11: "11_chin_raiser",
#             12: "12_lip_puckerer",
#             13: "13_lip_funneler",
#             14: "14_sadness",
#             15: "15_lip_roll",
#             16: "16_grin",
#             17: "17_cheek_blowing",
#             18: "18_eye_closed",
#             19: "19_brow_raiser",
#             20: "20_brow_lower"
#         }
#         # pyredner = pyredner
#         with open("./predef/Rt_scale_dict.json", 'r') as f:
#             self.Rt_scale_dict = json.load(f)
#         pyredner.set_use_gpu(torch.cuda.is_available())
#         pyredner.set_print_timing(False)
#         om_indices = np.load("./predef/om_indices.npy")
#         self.om_indices = torch.from_numpy(om_indices).type(torch.int32).to(pyredner.get_device())
#         self.image_data_root = "/data/home/us000042/lelechen/data/Facescape/jsons"

#     def shift(self, image, vector):
#         transform = AffineTransform(translation=vector)
#         shifted = warp(image, transform, mode='wrap', preserve_range=True)
#         return shifted.astype(image.dtype)


#     def meshrender(self,id_idx, exp_idx, vertices, cam_idx=1):
#         """
#         # id_idx: int
#         # exp_idx: int
#         # vertices: [3*VN] (openmesh ordering, float32 tensor)
#         # cam_idx: 1
#         # return: rendered image, [h,w,3]
#         """
#         scale = self.Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][0]
#         Rt_TU = np.array(self.Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][1])
#         # Rt_TU = torch.from_numpy(Rt_TU).type(torch.float32).to(pyredner.get_device())
#         Rt_TU = torch.from_numpy(Rt_TU).type(torch.float32).to("cuda:0")

#         print ('Rt_TU', Rt_TU.device)

#         print ('om_indices', self.om_indices.device)
#         print('+++++++')
#         input_vertices = vertices.reshape(-1,3).to("cuda:0")

#         # input_vertices = vertices.reshape(-1,3).to(pyredner.get_device())
#         input_vertices = (Rt_TU[:3,:3].T @ (input_vertices - Rt_TU[:3,3]).T).T
#         input_vertices = input_vertices / scale
#         input_vertices = input_vertices.contiguous()

#         print ('input_vertices', input_vertices.device)
#         m = pyredner.Material(diffuse_reflectance = torch.tensor((0.5, 0.5, 0.5), device = pyredner.get_device()))
#         # print ('m', m.device)

#         obj = pyredner.Object(vertices=input_vertices, indices=self.om_indices, material=m)
#         print ('obj.vertices', obj.vertices.device)
#         print ('obj.indices', obj.indices.device)
#         obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices)
#         print ('obj.normals', obj.normals.device)

#         # obj.normals = pyredner.compute_vertex_normal(obj.vertices.to(pyredner.get_device()), obj.indices.to(pyredner.get_device())).cpu()

#         img_dir = f"{self.image_data_root}/{id_idx}/{self.expressions[exp_idx]}"
#         with open(f"{img_dir}/params.json", 'r') as f:
#             params = json.load(f)

#         K = np.array(params['%d_K' % cam_idx])
#         Rt = np.array(params['%d_Rt' % cam_idx])
#         # dist = np.array(params['%d_distortion' % cam_idx], dtype = float)
#         h_src = params['%d_height' % cam_idx]
#         w_src = params['%d_width' % cam_idx]

#         cx = K[0,2]
#         cy = K[1,2]
#         dx = cx - 0.5 * w_src
#         dy = cy - 0.5 * h_src
#         dx = int(dx)
#         dy = int(dy)

#         c2w = np.eye(4)
#         c2w[:3,:3] = Rt[:3,:3].T
#         c2w[:3,3] = -Rt[:3,:3].T @ Rt[:3,3]
#         c2w = torch.from_numpy(c2w).type(torch.float32)
#         K = torch.from_numpy(K).type(torch.float32)

#         K[0,2] = 0
#         K[1,2] = 0
#         K[0,0] = K[0,0] * 2.0 / w_src
#         K[1,1] = -K[1,1] * 2.0 / w_src

#         # Setup camera
#         cam = pyredner.Camera(
#             cam_to_world= c2w,
#             intrinsic_mat=K,
#             clip_near = 1e-2, # needs to > 0
#             resolution = (h_src, w_src),
#             # distortion_params=distortion,
#             camera_type=pyredner.camera_type.perspective,
#             fisheye = False
#         )
        
#         light_dir = torch.tensor([[0.0, 0.0, 1.0]])
#         light_dir = (c2w[:3,:3]@light_dir.T).T
#         light_dir = light_dir.to("cuda:0")
#         # light_dir = light_dir.to(pyredner.get_device())
#         print ('light_dir:', light_dir.device)
#         lights = [
#             pyredner.DirectionalLight(light_dir, torch.tensor([5.0, 5.0, 5.0], device = pyredner.get_device()))
#         ]
        
#         scene = pyredner.Scene(camera=cam, objects=[obj])
#         img = pyredner.render_deferred(scene, lights=lights)

#         img = torch.pow(img, 1.0/2.2).cpu().numpy()
#         img = self.shift(img, [-dx, -dy])

#         img = img* 255
#         img = img.astype(np.uint8)
#         print ('**********************************')
#         return img
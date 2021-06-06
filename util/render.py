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

image_data_root = "/raid/celong/FaceScape/fsmview_images"
landmark_root = "/raid/celong/FaceScape/fsmview_landmarks"
mesh_root = "/raid/celong/FaceScape/textured_meshes"
rendering_root = "/raid/celong/FaceScape/fsmview_renderings"

pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.set_print_timing(False)


def shift(image, vector):
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    return shifted.astype(image.dtype)

def read_obj(obj_file):
    verts = []
    uvs = []
    indices = []
    uv_indices = []
    with open(obj_file) as f:
        for line in f.read().splitlines():
            if len(line) > 5 and line[:2] == "v ":
                tks = line.split()
                verts.append([float(tks[1]),float(tks[2]),float(tks[3])])
            if len(line) > 5 and line[:3] == "vt ":
                tks = line.split()
                uvs.append([float(tks[1]),float(tks[2])])
            if len(line) > 5 and line[:2] == "f ":
                tks = line.split()
                vi = []
                ti = []
                for tk in tks[1:]:
                    idx = tk.split("/")
                    vi.append(int(idx[0]))
                    ti.append(int(idx[1]))
                indices.append(vi)
                uv_indices.append(ti)
    return np.array(verts), np.array(uvs), np.array(indices)-1, np.array(uv_indices)-1


# with open("./predef/front_indices.pkl", "rb") as f:
#     indices_front = pickle.load(f)
# with open("./predef/predef_front_faces.pkl", 'rb') as f:
#     faces_front = pickle.load(f)
# f_front = np.array([f for f,_,_,_ in faces_front]) - 1
# f_front = torch.tensor(f_front, device=pyredner.get_device(), dtype=torch.int32)

with open("./predef/Rt_scale_dict.json", 'r') as f:
    Rt_scale_dict = json.load(f)
om_indices = np.load("./predef/om_indices.npy")
om_indices = torch.from_numpy(om_indices).type(torch.int32).to(pyredner.get_device())
    

def render(id_idx, exp_idx, vertices, cam_idx=1):  
    """
    # id_idx: int
    # exp_idx: int
    # vertices: [3*VN] (openmesh ordering, float32 tensor)
    # cam_idx: 1
    # return: rendered image, [h,w,3]
    """

    scale = Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][0]
    Rt_TU = np.array(Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][1])
    Rt_TU = torch.from_numpy(Rt_TU).type(torch.float32).to(pyredner.get_device())
    
    input_vertices = vertices.reshape(-1,3).to(pyredner.get_device())
    input_vertices = (Rt_TU[:3,:3].T @ (input_vertices - Rt_TU[:3,3]).T).T
    input_vertices = input_vertices / scale
    input_vertices = input_vertices.contiguous()

    m = pyredner.Material(diffuse_reflectance = torch.tensor((0.5, 0.5, 0.5), device = pyredner.get_device()))
    obj = pyredner.Object(vertices=input_vertices, indices=om_indices, material=m)
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices)

    img_dir = f"{image_data_root}/{id_idx}/{expressions[exp_idx]}"
    with open(f"{img_dir}/params.json", 'r') as f:
        params = json.load(f)

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
    lights = [
        pyredner.DirectionalLight(light_dir.to(pyredner.get_device()), torch.tensor([5.0, 5.0, 5.0], device = pyredner.get_device()))
    ]
    
    scene = pyredner.Scene(camera=cam, objects=[obj])
    img = pyredner.render_deferred(scene, lights=lights)

    img = torch.pow(img, 1.0/2.2).cpu().numpy()
    img = shift(img, [-dx, -dy])

    return img

if __name__ == '__main__':
    id_idx = 140
    exp_idx = 4
    cam_idx = 1
    
    mesh_path = f"{mesh_root}/{id_idx}/models_reg/{expressions[exp_idx]}.obj"

    om_mesh = openmesh.read_trimesh(mesh_path)
    om_vertices = np.array(om_mesh.points()).reshape(-1)
    om_vertices = torch.from_numpy(om_vertices.astype(np.float32))
    
    img = render(id_idx, exp_idx, om_vertices)
    
    imageio.imwrite("fkass.png", img)

    exit(0)
    for id_idx in range(1,400):
        for exp_idx in range(1,21):
            print(f"Working on id={id_idx}, exp={exp_idx}")

            mesh_path = f"{mesh_root}/{id_idx}/models_reg/{expressions[exp_idx]}.obj"
            if not os.path.exists(mesh_path):
                print(f"[WARN] {mesh_path} not exist!")
                continue
            om_mesh = openmesh.read_trimesh(mesh_path)
            verts = np.array(om_mesh.points())
            if (verts.shape[0] == 0):
                print(f"[WARN] {mesh_path} is empty!")
                continue

            objects = pyredner.load_obj(mesh_path, return_objects=True)
            objects[0].normals = pyredner.compute_vertex_normal(objects[0].vertices, objects[0].indices)
            # objects[0].vertices = objects[0].vertices[indices_front]
            objects[0].vertices = (Rt_TU[:3,:3].T @ (objects[0].vertices - Rt_TU[:3,3]).T).T
            objects[0].vertices = objects[0].vertices / scale
            objects[0].vertices = objects[0].vertices.contiguous()
            # objects[0].indices = f_front

            img_dir = f"{image_data_root}/{id_idx}/{expressions[exp_idx]}"
            with open(f"{img_dir}/params.json", 'r') as f:
                params = json.load(f)
            imgs = glob.glob(f"{img_dir}/*.jpg")
            rendering_dir = f"{rendering_root}/{id_idx}/{expressions[exp_idx]}"
            
            for img in imgs:
                cam_idx = int(os.path.basename(img).split(".")[0])
                
                rendering_path = f"{rendering_dir}/{cam_idx}.png"
                if os.path.exists(rendering_path):
                    continue

                # img_path = f"{img_dir}/{cam_idx}.jpg"
                
                K = np.array(params['%d_K' % cam_idx])
                Rt = np.array(params['%d_Rt' % cam_idx])
                dist = np.array(params['%d_distortion' % cam_idx], dtype = float)
                h_src = params['%d_height' % cam_idx]
                w_src = params['%d_width' % cam_idx]

                cx = K[0,2]
                cy = K[1,2]
                dx = cx - 0.5 * w_src
                dy = cy - 0.5 * h_src
                dx = int(dx)
                dy = int(dy)

                # gt_img = imageio.imread(img_path) / 255.0
                # gt_img = cv2.undistort(gt_img, K, dist)
                # print(gt_img.shape)

                c2w = np.eye(4)
                c2w[:3,:3] = Rt[:3,:3].T
                c2w[:3,3] = -Rt[:3,:3].T @ Rt[:3,3]
                c2w = torch.from_numpy(c2w).type(torch.float32)
                K = torch.from_numpy(K).type(torch.float32)

                K[0,2] = 0
                K[1,2] = 0
                K[0,0] = K[0,0] * 2.0 / w_src
                K[1,1] = -K[1,1] * 2.0 / w_src

                distortion = torch.tensor([dist[0], dist[1], 0, 0, 0, 0, dist[2], dist[3]]).type(torch.float32)
                # print(f"distortion: {distortion}")

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

                scene = pyredner.Scene(camera=cam, objects=objects)
                img = pyredner.render_albedo(scene)
                rendered_full_head_no = torch.pow(img, 1.0/2.2).cpu().numpy()
                # rendered_full_head_pos = shift(rendered_full_head_no, [dx, dy])
                rendered_full_head_neg = shift(rendered_full_head_no, [-dx, -dy])

                # imageio.imsave(f"results/blend_no{id_idx}_exp{exp_idx}_cam{cam_idx}.png", 0.5 * (rendered_full_head_no + gt_img))
                # imageio.imsave(f"results/blend_pos{id_idx}_exp{exp_idx}_cam{cam_idx}.png", 0.5 * (rendered_full_head_pos + gt_img))
                # imageio.imsave(f"results/blend_neg{id_idx}_exp{exp_idx}_cam{cam_idx}.png", 0.5 * (rendered_full_head_neg + gt_img))

                # blend_img = 0.5 * (rendered_full_head + gt_img)

                rendered_full_head = np.clip((255 * rendered_full_head_neg), 0, 255).astype(np.uint8)
                # gt_img = np.clip((255 * gt_img), 0, 255).astype(np.uint8)
                # blend_img = np.clip((255 * blend_img), 0, 255).astype(np.uint8)
                
                if not os.path.exists(rendering_dir):
                    os.makedirs(rendering_dir)
                imageio.imsave(rendering_path, rendered_full_head)
                # imageio.imsave(f"results/head_id{id_idx}_exp{exp_idx}_cam{cam_idx}.png", rendered_full_head)
                # imageio.imsave(f"results/ori_id{id_idx}_exp{exp_idx}_cam{cam_idx}.png", gt_img)
                # imageio.imsave(f"results/blend_id{id_idx}_exp{exp_idx}_cam{cam_idx}.png", blend_img)
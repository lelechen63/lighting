import json
import numpy as np
import trimesh
import imageio
import openmesh
import cv2

import math
import pickle
import os

import torch
from tqdm import tqdm
import glob

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

image_data_root = "/nfs/STG/CodecAvatar/lelechen/Facescape/fsmview_images"
landmark_root = "/nfs/STG/CodecAvatar/lelechen/Facescape/fsmview_landmarks"
mesh_root = "/nfs/STG/CodecAvatar/lelechen/Facescape/textured_meshes"
rendering_root = "/nfs/STG/CodecAvatar/lelechen/Facescape/fsmview_renderings"

def shift(image, vector):
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    return shifted.astype(image.dtype)

if __name__ == '__main__':

    ##################################################################################
    ################################ Single Shot Test ################################
    Rt_scale_dict = json.load(open("/home/uss00022/lelechen/github/lighting/predef/Rt_scale_dict.json", 'r'))
    lm_list_v10 = np.load("/home/uss00022/lelechen/github/lighting/predef/landmark_indices.npz")['v10']
    
    for id_idx in tqdm(range(1,400)):
        for exp_idx in range(1,21):

            img_dir = f"{image_data_root}/{id_idx}/{expressions[exp_idx]}"
            with open(f"{img_dir}/params.json", 'r') as f:
                params = json.load(f)
            print(f"Working on id={id_idx}, exp={exp_idx}")

            scale = Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][0]
            Rt_TU = np.array(Rt_scale_dict['%d'%id_idx]['%d'%exp_idx][1])
        
            # extract landmarks in canonical space
            mesh_path = f"{mesh_root}/{id_idx}/models_reg/{expressions[exp_idx]}.obj"
            om_mesh = openmesh.read_trimesh(mesh_path)
            verts = np.array(om_mesh.points())
            lmks = verts[lm_list_v10,:] # (68,3), in canonical space

            # transform to world space
            lmks = (np.tensordot(Rt_TU[:3,:3].T, (lmks - Rt_TU[:3, 3]).T, 1)).T
            lmks /= scale # (68, 3), in world space

            imgs = glob.glob(f"{img_dir}/*.jpg")
            landmark_dir = f"{landmark_root}/{id_idx}/{expressions[exp_idx]}"
            os.mkdirs(landmark_dir, exists_ok = True)
            for img in imgs:
                cam_idx = int(os.path.basename(img).split(".")[0])

                # projection to screen space
                K = np.array(params['%d_K' % cam_idx])
                Rt = np.array(params['%d_Rt' % cam_idx])
                dist = np.array(params['%d_distortion' % cam_idx], dtype = np.float)
                h_src = params['%d_height' % cam_idx]
                w_src = params['%d_width' % cam_idx]
                R = Rt[:3,:3]
                T = Rt[:3,3:]
                pos = K @ (R @ lmks.T + T)
                coord = pos[:2,:] / pos[2,:] # (2, 68)

                coord = np.transpose(coord, (1, 0))
                
                # np.save( os.path.join( landmark_dir, '%d.npy'%cam_idx), coord)
                # # plot test
                img = cv2.imread( os.path.join(  img_dir , "%d.jpg" % cam_idx ))
                # undist_img = cv2.undistort(img, K, dist)

                for ind in range(68):
                    uv = coord[ind, :]
                    u, v = np.round(uv).astype(np.int)
                    color_draw = cv2.circle(img, (u, v), 10, (100, 100, 100), -1)
                    color_draw = cv2.putText(color_draw, "%02d"%(ind), (u-8, v+4), 
                                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale = 0.4,
                                            color = (255, 255, 255))

                cv2.imwrite( os.path.join(landmark_dir, "%d.jpg" % cam_idx )  , color_draw)


  
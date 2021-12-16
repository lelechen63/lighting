

import dlib 
import cv2
import numpy as np
from utils import _get_full_model_points, solve_pose_by_68_points
import os
import pickle
# dlib’s pre-trained facial landmark detector
import face_alignment
import torch
from tqdm import tqdm 
from skimage import io
def get_exp():
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
    exps = []
    for i in range(1,21):
        exps.append(expressions[i])
    return set(exps)

detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = "cuda:0" ) 

def get_front_list(tt):

    dataroot = '/nfs/STG/CodecAvatar/lelechen/Facescape'
    model_points_68 = _get_full_model_points()
    imgsize = (256,256)
    frontlist = {}
    exps = get_exp()
    for pid in tqdm(os.listdir(  '/nfs/STG/CodecAvatar/lelechen/Facescape/ffhq_aligned_img' )):
        for exp in exps:
            pid = '90'
            exp = '9_mouth_right'
            img_f = os.path.join( dataroot, 'ffhq_aligned_img', pid, exp )
            # frames = []
            new_p = []
            # try:
            for i in range(60):
                try:
                    img_p = os.path.join( img_f, '%d.jpg'%i)
                    image = cv2.imread(img_p)
                    image = cv2.resize(image, imgsize)
                    input_img = io.imread(img_p)
                    preds = detector.get_landmarks(image)
                    if preds is None:
                        new_p.append([])
                    else:
                        new_p.append(preds[0])
                    
                except:
                    print (img_p,'====')
                    continue
            smallyaw = 100
            smallidx = -1
            for i in range(len(new_p)):
                pp = new_p[i]
                if pp == []:
                    continue
                else:
                    pp = np.asarray(pp)
                    pose = solve_pose_by_68_points(pp, imgsize, model_points_68)
                    print (i, pose[0][0])
                    yaw = abs(pose[0][0][0])
                    if yaw < smallyaw:
                        smallyaw = yaw
                        smallidx = i
            frontlist[ pid+ '/models_reg/' + exp] = smallidx
            # except:
            #     print (img_f, '!!!!!!!')
            #     continue
        break
    print (frontlist)
    print (len(frontlist))
    with open( dataroot +   '/compressed/frontlist.pkl', 'wb') as handle:
        pickle.dump(frontlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

get_front_list('test')

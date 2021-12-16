

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
            img_f = os.path.join( dataroot, 'ffhq_aligned_img', pid, exp )
            frames = []
            # try:
            for i in range(60):
                try:
                    img_p = os.path.join( img_f, '%d.jpg'%i)
                    image = cv2.imread(img_p)
                    image = cv2.resize(image, imgsize)
                    # preds = detector.get_landmarks(image)
                    # preds = np.asarray(preds)
                    # print (preds.shape)
                    frames.append(image)
                except:
                    continue
            batch  = np.stack(frames)
            batch = torch.Tensor(batch.transpose(0, 3, 1, 2))
            points = detector.get_landmarks_from_batch(batch)
            print (len(points))
            print (np.asarray(points[0]).shape, '++++++')
            new_p = []
            for k in range(len(points)):
                tmp = []
                for j in range(int(len(points[k])/68)):
                    tmp.append(points[k][68 * j : 68 * (j +1)].tolist())
                new_p.append(tmp)
            print (np.asarray(new_p[0]).shape, '+++----+++')
            smallyaw = 100
            smallidx = -1
            for i in range(len(new_p)):
                img = frames[i]
                for k in range(len(new_p[i])): 
                    pp = new_p[i][k]
                    pp = np.asarray(pp)
                    pose = solve_pose_by_68_points(pp, imgsize, model_points_68)
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



import dlib 
import cv2
import numpy as np
from utils import *
import os
import pickle
# dlib’s pre-trained facial landmark detector
import face_alignment
import torch
detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = "cuda:0" ) 

def get_front_list(tt):
    dataroot = '/nfs/STG/CodecAvatar/lelechen/Facescape'
    all_list =  'compressed/all320_{}list.pkl'.format(tt)
    _file = open(os.path.join(dataroot, all_list), "rb")
    data_list = pickle.load(_file)
    model_points_68 = _get_full_model_points()
    imgsize = (256,256)

    _file.close()
    for data in data_list:
        tmp = data.split('/')
        img_f = os.path.join( dataroot, 'ffhq_aligned_img', tmp[0],tmp[-1])
        frames = []
        for i in range(60):
            try:
                img_p = os.path.join( img_f, '%d.jpg'%i)
                print (img_p)
                img_p = '/nfs/STG/CodecAvatar/lelechen/Facescape/ffhq_aligned_img/1/1_neutral/1.jpg'
                image = cv2.imread(img_p)
                image = cv2.resize(image, imgsize)
                frames.append(image)
            except:
                print ('++++++++')
                continue
        batch  = np.stack(frames)
        batch = torch.Tensor(batch.transpose(0, 3, 1, 2))
        points = self.detector.get_landmarks_from_batch(batch)
        new_p = []
        for k in range(len(points)):
            tmp = []
            for j in range(int(points[k].shape[0]/68)):
                tmp.append(points[k][68 * j : 68 * (j +1)].tolist())
            new_p.append(tmp)
        
        for i in range(len(new_p)):
            img = frames[i]
            for k in range(len(new_p[i])): 
                pp = new_p[i][k]
                pp = np.asarray(pp)
                pose = solve_pose_by_68_points(pp, imgsize, model_points_68)
                print (pose)
                cv2.putText(img, str(pose), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1 )

                cv2.imwrite("./tmp/{}.png"%i, img)
        
        print (ggggg)



            
            

get_front_list('test')

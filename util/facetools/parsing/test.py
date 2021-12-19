from PIL import Image
from face_parsing import parsing, vis_parsing_maps
import numpy as np
import face_alignment
# from eye_parsing.iris_detector import IrisDetector
import dlib
from model import BiSeNet
import os
import pickle
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time, threading


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

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
# idet = IrisDetector()
# idet.set_detector(fa)

n_classes = 19
facenet = BiSeNet(n_classes=n_classes)
facenet.cuda()
facenet.load_state_dict(torch.load('./face_parsing.pth'))
facenet.eval()

base_p = '/nfs/STG/CodecAvatar/lelechen/Facescape/ffhq_aligned_img'
total_ids =  os.listdir(base_p)
total_ids.sort()

def get_parsing_batch( ids ):
    for id_p in ids:
        current_p = os.path.join( base_p , id_p)
        for k in range(len(expressions)):
            motion_p = expressions[k + 1]
            current_p1 = os.path.join( current_p , motion_p)
            for valid_f in range( len ( os.listdir( current_p1 ))):
                img_path = os.path.join( current_p1, str(valid_f) + '.jpg')
                # if os.path.exists(img_path[:-4] +'_mask.png'):
                #     continue
                # parsing_path = img_path.replace('ffhq_aligned_img', 'fsmview_landmarks')[:-4] +'_parsing.png'
                parsing_path = './gg.png'
                print (img_path)
                # try:
                image = Image.open(img_path)
                # res = parsing(image, facenet, idet, img_path[:-4] +'_mask.png')
                res = parsing(image, facenet)
                vis_parsing_maps(image, res, save_parsing_path=parsing_path, save_vis_path ='./gg2.png' ) 
                print('---------')
                # except:
                #     print ('**********')
                #     print (img_path)
                #     continue
                break
            break 
        break
get_parsing_batch(total_ids)
# batch = 7
# for i in range(1):
#     threading.Thread(target = get_parsing_batch, args = (total_ids[batch * i: batch *(i+1)], )).start()

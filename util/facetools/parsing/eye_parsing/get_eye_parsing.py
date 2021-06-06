import cv2
from matplotlib import pyplot as plt
import face_alignment
from iris_detector import IrisDetector
import dlib
import numpy as np 
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0"


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


# def resize_image(im, max_size=768):
#     if np.max(im.shape) > max_size:
#         ratio = max_size / np.max(im.shape)呢
#         print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))})  在好  
root = '/raid/celong/FaceScape/'


def get_imgs(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

idet = IrisDetector()
idet.set_detector(fa)

base_p = '/raid/celong/FaceScape/fsmview_images'
if not os.path.exists( base_p.replace('fsmview_images', 'ffhq_aligned_img') ):
    os.mkdir(base_p.replace('fsmview_images', 'ffhq_aligned_img'))
save_p = base_p.replace('fsmview_images', 'ffhq_aligned_img')

_file = open( '/raid/celong/lele/github/idinvert_pytorch/predef/frontface_list.pkl', "rb")
front_indx = pickle.load(_file)

ids =  os.listdir(base_p)
ids.sort()

for id_p in ids:
    current_p = os.path.join( base_p , id_p)
    front_idx = front_indx[id_p]
   
    for motion_p in os.listdir(current_p):
        current_p1 = os.path.join( current_p , motion_p)
        img_path = os.path.join( current_p1, front_idx + '.jpg')
        lmark_path = img_path.replace('fsmview_images', 'fsmview_landmarks')[:-3] +'npy'
        lmark = None
        parsing_path = lmark_path[:-4] + '_eye.png'

        # debug
        # img_path = "/raid/celong/FaceScape/fsmview_images/1/9_mouth_right/1.jpg"
        # lmark_path = "/raid/celong/FaceScape/fsmview_landmarks/1/9_mouth_right/1.npy"
        # parsing_path = "/raid/celong/FaceScape/fsmview_landmarks/1/9_mouth_right/1_eye.png"

        if os.path.exists(lmark_path):
            lmark = np.load(lmark_path).transpose(1,0)[:,::-1]
        else:
            print ('*********', lmark_path)
            continue

        im = cv2.imread(img_path)[..., ::-1]
        # im = resize_image(im) # Resize image to prevent GPU OOM.
        h, w, _ = im.shape
        try:
            eye_lms = idet.detect_iris(im,lmark)
        except:
            print (img_path, '**************')
            continue
        # Display detection result
        draw = idet.draw_pupil(im, eye_lms[0][0,...]) # draw left eye
        draw = idet.draw_pupil(draw, eye_lms[0][1,...]) # draw right eye

        blank_image = np.zeros((h,w,3), np.uint8)
        lms =   eye_lms[0][0,...].astype(np.int32)[:,::-1]

        cv2.fillConvexPoly(blank_image, lms[:8], (0,0,255))
        cv2.fillConvexPoly(blank_image, lms[8:16], (255,0,0))

        lms = eye_lms[0][1,...].astype(np.int32)[:,::-1]
        cv2.fillConvexPoly(blank_image, lms[:8], (0,0,255))
        cv2.fillConvexPoly(blank_image, lms[8:16], (255,0,0))
        cv2.imwrite(parsing_path, blank_image)
        print (parsing_path)
        # break
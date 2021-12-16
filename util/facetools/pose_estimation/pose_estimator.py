

import dlib 
import cv2
import numpy as np
from utils import *
import os
import pickle
# dlib’s pre-trained facial landmark detector
predictorPath = "/home/uss00022/lelechen/basic/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

def get_front_list(tt):
    dataroot = '/nfs/STG/CodecAvatar/lelechen/Facescape'
    all_list =  'compressed/all320_{}list.pkl'.format(tt)
    _file = open(os.path.join(dataroot, all_list), "rb")
    data_list = pickle.load(_file)

    _file.close()
    for data in data_list:
        tmp = data.split('/')
        img_f = os.path.join( dataroot, 'ffhq_aligned_img', tmp[0],tmp[-1])
        for i in range(60):
            img_p = os.path.join( img_f, '%d.jpg')
            image = cv2.imread(img_p)
            image = cv2.resize(image, (256,256))
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray_image, 1)
            # try:
            for (i, rect) in enumerate(rects): 
                # determine the facial landmarks for the face region (face region is stored in the rect)
                shape = predictor(gray_image, rect)

                # convert the facial landmark (x, y)-coordinates to a NumPy array (68*2)
                shape_array = shape_to_np(shape)
                
                # Show head pose
                image = HPS(image, shape_array)
                
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # show the face number
                cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
                #for (x, y) in shape_array:
                    #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)   # Negative thickness means that a filled circle is to be drawn.
            
                cv2.imwrite("./Result.png", image)
                print (ggggg)



            
            

get_front_list('test')

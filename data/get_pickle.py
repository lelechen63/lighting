import pickle
import os
import random
import openmesh
from PIL import Image
import numpy as np
import sys
sys.path.append("/data/home/us000042/lelechen/github/lighting")
from util.render_class import meshrender
from tqdm import tqdm
import torch
import util.util as util
import matplotlib.pyplot as plt
import cv2
def get_image_pickle():
    
    base_p = '/raid/celong/FaceScape/ffhq_aligned_img'
    save_p = '/raid/celong/FaceScape/ffhq_aligned_img'

    _file = open( '/raid/celong/lele/github/idinvert_pytorch/predef/validface_list.pkl', "rb")
    valid_indx = pickle.load(_file)
    print(len(valid_indx))
    # print (valid_indx.keys())

    hhh = 0
    train_list = []
    test_list = []

    ids =  os.listdir(base_p)
    ids.sort()
    invalid = []
    total = 0
    for id_p in ids:
        current_p = os.path.join( base_p , id_p)
        save_p1 = os.path.join( save_p , id_p)
        all_motions = os.listdir(current_p)
        random.shuffle(all_motions)
        for k, motion_p in enumerate(all_motions):
            current_p1 = os.path.join( current_p , motion_p)
            save_p2 = os.path.join( save_p1 , motion_p)
            if id_p +'__' + motion_p not  in valid_indx.keys():
                continue
            for cam_idx in valid_indx[ id_p +'__' + motion_p ]:
                total +=1
                img_p = os.path.join( save_p2, cam_idx + '.jpg')
                output_p = os.path.join( save_p2 ,cam_idx + '_render.png')
                parsing_p = img_p[:-4].replace('ffhq_aligned_img', 'fsmview_landmarks' ) + '_parsing.png'
                # print (img_p, output_p, parsing_p)
                # if os.path.exists(img_p) and os.path.exists(output_p) and os.path.exists(parsing_p) :
                if os.path.exists(img_p)  and os.path.exists(parsing_p) :
                    # if id_p =='12':
                    # print ( os.path.join( id_p , motion_p, cam_idx + '.jpg'))
                    if k < 17:
                        train_list.append( os.path.join( id_p , motion_p, cam_idx + '.jpg') )
                    else:
                        test_list.append( os.path.join( id_p , motion_p, cam_idx + '.jpg') )

                else:
                    # print (img_p, parsing_p)
                    invalid.append(parsing_p)
                    continue
                # print ('gg')
    print (len(train_list), len(test_list), total,len(invalid))

    with open('/raid/celong/FaceScape/lists/img_alone_train.pkl', 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/raid/celong/FaceScape/lists/img_alone_test.pkl', 'wb') as handle:
        pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_paired_image_pickle():
    _file = open(os.path.join('/raid/celong/FaceScape', "lists/img_alone_train.pkl"), "rb")
    all_train_list = pickle.load(_file)
    _file.close()
    train_list  = {}

    for item in all_train_list:
        tmp = item.split('/')
        pid = tmp[0]
        motion = tmp[1]
        if pid not in train_list.keys():
            train_list[pid] = {}
            train_list[pid][motion] = [item]
        else:
            if motion not in train_list[pid].keys():
                train_list[pid][motion] = [item]
            else:
                train_list[pid][motion].append(item)
        
        if motion not in train_list.keys():
            train_list[motion] = {}
            train_list[motion][pid] =[item]
        else:
            if pid not in train_list[motion].keys():
                train_list[motion][pid] = [item]
            else:
                train_list[motion][pid].append(item)

    print (len(train_list), len(train_list[motion]), len(train_list[motion][pid]))
    print (train_list[motion][pid])

    print (len(train_list), len(train_list[pid]), len(train_list[pid][motion]))
    print (train_list[pid][motion])
    
    with open('/raid/celong/FaceScape/lists/img_dic_train.pkl', 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def get_texmesh_pickle():
    
    base_p = '/raid/celong/FaceScape/textured_meshes'
    train_list = []
    test_list = []
    ids =  os.listdir(base_p)
    ids.sort()
    for id_p in ids:
        print (id_p ,'/', len(ids))
        current_p = os.path.join( base_p , id_p, 'models_reg')
        all_files = os.listdir(current_p)
        all_motions = []
        for f  in all_files:
            if 'jpg' in f:
                all_motions.append(f[:-4])
        random.shuffle(all_motions)
        for k, motion_p in enumerate(all_motions):
            try:
                tex_path = os.path.join('/raid/celong/FaceScape/texture_mapping/target/', id_p, motion_p + '.png')
                mesh_path = os.path.join(current_p, motion_p + '.obj')
                tex = Image.open(tex_path).convert('RGB')
                tex  = np.array(tex ) 
                om_mesh = openmesh.read_trimesh(mesh_path)
                A_vertices = np.array(om_mesh.points())
                if A_vertices.shape[0] == 26317 and tex.shape[0] == 4096:
                    if k < 17:
                        train_list.append( os.path.join( id_p , 'models_reg', motion_p) )
                    else:
                        test_list.append( os.path.join( id_p , 'models_reg',  motion_p) )
                    print(tex_path)
                else:
                    print(A_vertices.shape)
            except:
                continue
        #     if len(train_list) == 50:
        #         break
        # if len(train_list) == 50:
        #     break
        print (len(train_list))
    print (test_list[:10])
    print (len(train_list), len(test_list))

    with open('/raid/celong/FaceScape/lists/texmesh_train.pkl', 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/raid/celong/FaceScape/lists/texmesh_test.pkl', 'wb') as handle:
        pickle.dump(test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_paired_texmesh_pickle():
    base_p = '/raid/celong/FaceScape/textured_meshes'
    ids = []
    # ids =  os.listdir(base_p)
    # ids.sort()
    # id_list = []
    # ids = ids[:300]
    # print(ids)
    for i in range(300):
        ids.append(str(i + 1))
    with open('/raid/celong/FaceScape/lists/ids.pkl', 'wb') as handle:
        pickle.dump(ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

def gettexmesh_pid_expid():
    dataroot = '/raid/celong/FaceScape/'
    _file = open(os.path.join(dataroot, "lists/texmesh_train.pkl"), "rb")
    data_list = pickle.load(_file)
    _file.close()
    pid = set([])
    exp = set([])
    for d in data_list:
        print(d)
        tmp = d.split('/')
        pid.add(str(tmp[0]))
        exp.add(int(tmp[-1].split('_')[0]))
        # break
    pid = sorted(pid)
    exp = sorted(exp)
    print(exp)
    print(pid)
    with open('/raid/celong/FaceScape/lists/ids.pkl', 'wb') as handle:
        pickle.dump(pid, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_mesh_mean_id():
    dataroot = '/data/home/us000042/lelechen/data/Facescape/'
    _file = open(os.path.join(dataroot, "lists/texmesh_train.pkl"), "rb")
    dir_A = os.path.join(dataroot, "textured_meshes")  
    if not os.path.exists( os.path.join(dataroot, "meanmesh")   ):
        os.mkdir(os.path.join(dataroot, "meanmesh"))
    data_list = pickle.load(_file)#[:1]
    _file = open(os.path.join(dataroot, "lists/texmesh_test.pkl"), "rb")
    data_list.extend(pickle.load(_file))
    total_mesh = {}
    cc = 0
    for data in tqdm(data_list):
        cc += 1
        mesh_path = os.path.join( dir_A , data + '.obj')
        om_mesh = openmesh.read_trimesh(mesh_path)
        A_vertices = np.array(om_mesh.points()).reshape(-1)
        
        tmp = data.split('/')
        if tmp[0] not in total_mesh.keys():
            total_mesh[tmp[0]] = [A_vertices]
        else:
            total_mesh[tmp[0]].append(A_vertices)

        # if cc == 100:
        #     break
    for k in total_mesh.keys():
        print (k)
        c_mesh = total_mesh[k]
        c_mesh = np.asarray(c_mesh)
        print (c_mesh.shape, 'c_mesh')
        mean_shape = np.mean(c_mesh, axis=0)
        print (mean_shape.shape)
        mean_shape = torch.FloatTensor(mean_shape)
        np_path = os.path.join(dataroot, "meanmesh" , k + '.npy') 
        np.save( np_path, mean_shape  )
        mean_Amesh = meshrender(int(tmp[0]), int(tmp[-1].split('_')[0]),mean_shape )
        vis_path = os.path.join(dataroot, "meanmesh" , k + '.png') 
        util.save_image(mean_Amesh, vis_path)


def get_meanmesh():
    dataroot = '/data/home/us000042/lelechen/data/Facescape/'
    meanmeshpath = os.path.join(dataroot, "meanmesh")
    total = os.listdir( meanmeshpath)
    meanmesh = []
    for kk in total:
        if kk[-3:] == 'npy':
            meanmesh.append(np.load( os.path.join( meanmeshpath, kk)  ))
    meanmesh = np.asarray(meanmesh)
    print (meanmesh.shape, 'meanmesh')
    meanmesh = np.mean(meanmesh, axis=0)
    save_p = '/data/home/us000042/lelechen/github/lighting/predef/meanmesh.npy'
    np.save( save_p, meanmesh )



def get_mesh_total():
    dataroot = '/data/home/us000042/lelechen/data/Facescape/'
    _file = open(os.path.join(dataroot, "lists/texmesh_test.pkl"), "rb")
    dir_A = os.path.join(dataroot, "textured_meshes")  
    if not os.path.exists( os.path.join(dataroot, "meanmesh")   ):
        os.mkdir(os.path.join(dataroot, "meanmesh"))
    data_list = pickle.load(_file)#[:1]
    # _file = open(os.path.join(dataroot, "lists/texmesh_test.pkl"), "rb")
    totalmeanmesh = np.load( '/data/home/us000042/lelechen/github/lighting/predef/meanmesh.npy' )
    # data_list.extend(pickle.load(_file))
    cc = 0
    big = []
    for data in tqdm(data_list):
        cc += 1
        mesh_path = os.path.join( dir_A , data + '.obj')
        om_mesh = openmesh.read_trimesh(mesh_path)
        A_vertices = np.array(om_mesh.points()).reshape(-1)
        big.append(A_vertices)
    big = np.asarray(big)
    np.save( '/data/home/us000042/lelechen/data/Facescape/bigmeshtest.npy', big )

def getmeshnorm():
    dataroot = '/data/home/us000042/lelechen/data/Facescape/'
    _file = open(os.path.join(dataroot, "lists/texmesh_train.pkl"), "rb")
    data_list = pickle.load(_file)
    _file = open(os.path.join(dataroot, "lists/texmesh_test.pkl"), "rb")
    data_list.extend(pickle.load(_file))

    big = np.load( '/data/home/us000042/lelechen/data/Facescape/bigmesh.npy' )
    totalmeanmesh = np.load( '/data/home/us000042/lelechen/github/lighting/predef/meanmesh.npy' )
    big = big -  totalmeanmesh
    maxv = []
    minv = []
    for i in range(big.shape[0]):
        maxv.append(big[i].max())
        minv.append(big[i].min())
        if big[i].max() > 40 or big[i].min() < -40:
            print( '/data/home/us000042/lelechen/data/Facescape/textured_meshes/' + data_list[i], big[i].max(),big[i].min() )

    plt.plot(maxv,minv, 'o',color='b')
    plt.show()
    plt.savefig('./gg.png')


def get_tex_total():
    dataroot = '/data/home/us000042/lelechen/data/Facescape/'
    _file = open(os.path.join(dataroot, "lists/texmesh_test.pkl"), "rb")
    dir_A = os.path.join(dataroot, "textured_meshes")  
    
    data_list = pickle.load(_file)#[:1]
    # _file = open(os.path.join(dataroot, "lists/texmesh_test.pkl"), "rb")
    totalmeanmesh = np.load( '/data/home/us000042/lelechen/github/lighting/predef/meanmesh.npy' )
    # data_list.extend(pickle.load(_file))
    dir_tex  = os.path.join(dataroot, "texture_mapping", 'target')
    cc = 0
    big = []
    facial_seg = Image.open("/data/home/us000042/lelechen/github/lighting/predef/facial_mask_v10.png")
    facial_seg  = np.array(facial_seg ) / 255.0
    facial_seg = np.expand_dims(facial_seg, axis=2)
    x = 1169-150
    y =600-100
    w =2000
    h = 1334
    l = max(w ,h)
    for data in tqdm(data_list):
        cc += 1
        tmp = data.split('/')
        tex_path = os.path.join( dir_tex , tmp[0], tmp[-1] + '.png')
        tex = Image.open(tex_path).convert('RGB')#.resize(img_size)
        tex  = np.array(tex)
        tex = tex * facial_seg
        tex =  tex[y:y+l,x :x +l,:]
        tex = cv2.resize(tex, (256,256), interpolation = cv2.INTER_AREA)
        big.append(tex)
    big = np.asarray(big)
    np.save( '/data/home/us000042/lelechen/data/Facescape/bigtex256test.npy', big )

def get_texnorm():
    dataroot = '/data/home/us000042/lelechen/data/Facescape/'
    _file = open(os.path.join(dataroot, "lists/texmesh_train.pkl"), "rb")
    data_list = pickle.load(_file)
    _file = open(os.path.join(dataroot, "lists/texmesh_test.pkl"), "rb")
    data_list.extend(pickle.load(_file))

    big = np.load( '/data/home/us000042/lelechen/data/Facescape/bigtex256train.npy' )
    meantex = np.mean(big, axis=0)
    np.save( '/data/home/us000042/lelechen/github/lighting/predef/meanmesh.npy', meantex)
get_texnorm()
# get_mesh_total()
# tmp()
# get_mean()
# gettexmesh_pid_expid()

# get_paired_texmesh_pickle()
# get_texmesh_pickle()
# get_paired_image_pickle()



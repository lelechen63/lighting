import os.path
from data.base_dataset import *
from data.image_folder import make_dataset

from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL
import json
import pickle 
import cv2
import numpy as np
import random
import torch
import openmesh
from tqdm import tqdm
import  os, time
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
def get_anlge_list():
    angle_lists =  open("/raid/celong/lele/github/idinvert_pytorch/predef/angle_list2.txt", 'r')
    total_list = {}
    while True:
        line = angle_lists.readline()[:-1]
        if not line:
            break
        tmp = line.split(',')
        if tmp[0] +'/' + tmp[1] not in total_list.keys():

            total_list[tmp[0] +'/' + tmp[1] ]  = {}
        total_list[tmp[0] +'/' + tmp[1] ][tmp[2]] = [float(tmp[3]),float(tmp[4]), float(tmp[5])]

    return total_list
class FacescapeDirDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        ### input A (renderred image)
        self.dir_A = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input B (real images)
        self.dir_B = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input C (eye parsing images)
        self.dir_C = os.path.join(opt.dataroot, "fsmview_landmarks")

        ### json 
        self.dir_json = os.path.join(opt.dataroot, "fsmview_images")

        # /raid/celong/FaceScape/fsmview_landmarks/99/14_sadness/1_eye.png
        self.exp_set =  get_exp()

        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/img_alone_train.pkl"), "rb")
            
        else:
            _file = open(os.path.join(opt.dataroot, "lists/img_alone_test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)#[:1]
        _file.close()
        
        dic_file = open(os.path.join(opt.dataroot, "lists/img_dic_train.pkl"), "rb")
        self.dic_list = pickle.load(dic_file)#[:10]

        self.angle_list = get_anlge_list()
        
    def __getitem__(self, index):

        tmp = self.data_list[index].split('/')
        A_path = os.path.join( self.dir_A , self.data_list[index] ) 
        mask_path = os.path.join( self.dir_A , self.data_list[index][:-4] + '_mask.png' )
        json_path = os.path.join( self.dir_json , tmp[0], tmp[1], 'params.json' )
        
        f  = open(json_path , 'r')
        params = json.load(f)
        viewpoint = [np.array(params['%s_Rt' %  tmp[2][:-4]]).flatten()]
        ### input mask (binary mask to segment person out)
        mask = cv2.imread(mask_path)[:,:,::-1]
        ### input A (real image)
        A = cv2.imread(A_path)[:,:,::-1]
        A = A * mask
        A = Image.fromarray(np.uint8(A))
        params = get_params(self.opt, A.size)
        transform = get_transform(self.opt, params)      
        A_tensor = transform(A)

        small_index = 0
        A_angle = self.angle_list[tmp[0] +'/' + tmp[1]][tmp[2][:-4]]
        
        pid = tmp[0]
        expresison = tmp[1]

        # randomly get paired image (same identity or same expression)
        toss = random.getrandbits(1)
        # toss 0-> same iden, diff exp
        if toss == 0:
            pool = set(self.dic_list[pid].keys()) - set(expresison)
            B_exp = random.sample(pool, 1)[0]
            B_id = pid
            B_angle_pool = self.angle_list[pid +'/' + B_exp]
        # toss 1 -> same exp, diff iden
        else:
            pool = set(self.dic_list[expresison].keys()) - set(pid)
            B_id = random.sample(pool, 1)[0]
            B_exp = expresison
            B_angle_pool = self.angle_list[B_id +'/' + expresison]
        
        ggg = []
        for i in range(len(B_angle_pool)):
            ggg.append(B_angle_pool[str(i)])
        ggg = np.array(ggg)
        diff = abs(ggg - A_angle).sum(1)
        
        for kk in range(diff.shape[0]):
            small_index = diff.argsort()[kk]
            try:
                # print (small_index)
                B_path =  os.path.join( self.dir_A ,  B_id, B_exp, str(small_index) +'.jpg' )   
                # print (B_path)
                ### input mask (binary mask to segment person out)
                mask_path =os.path.join( self.dir_A ,B_id, B_exp, str(small_index)+ '_mask.png' )   
                # mask = Image.open(mask_path).convert('RGB')
                mask = cv2.imread(mask_path)[:,:,::-1] 
                B = cv2.imread(B_path)[:,:,::-1]
                break
            except:
                continue
        json_path = os.path.join( self.dir_json , B_id, B_exp, 'params.json' )
        f  = open(json_path , 'r')
        params = json.load(f)
        
        viewpoint.append(np.array(params['%d_Rt' %  small_index]).flatten())
        B = B * mask
        B = Image.fromarray(np.uint8(B))
        B_tensor = transform(B)
        viewpoint = np.asarray(viewpoint)
        viewpoint = torch.FloatTensor(viewpoint)

        input_dict = { 'image':A_tensor, 'pair_image': B_tensor, 'pair_type': toss, 'viewpoint' : viewpoint, 'A_path': self.data_list[index][:-4] , 'B_path': os.path.join(B_id, B_exp, str(small_index)) }

        return input_dict

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeDirDataset'


class FacescapeMeshTexDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        ### input A (texture and mesh)   
        self.dir_A = os.path.join(opt.dataroot, "textured_meshes")

        self.dir_tex = '/raid/celong/FaceScape/texture_mapping/target/'

        ### input B (real images)
        self.dir_B = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input C (eye parsing images)
        self.dir_C = os.path.join(opt.dataroot, "fsmview_landmarks")

        ### json 
        self.dir_json = os.path.join(opt.dataroot, "fsmview_images")

        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/texmesh_train.pkl"), "rb")
            
        else:
            _file = open(os.path.join(opt.dataroot, "lists/texmesh_test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)#[:1]
        _file.close()
        
        ids = open(os.path.join(opt.dataroot, "lists/ids.pkl"), "rb")
        self.id_set = set(pickle.load(ids))
        self.exp_set = get_exp()
        print ('===========================')
        print ('id_set:',self.id_set)

        print('!!!!!!!!!')
        print( '0' in self.id_set)

        print('+++++++++++++++++++++++++++')
        print ('exp_set:',self.exp_set)

        print ('===========================')

        # self.facial_seg = cv2.imread("./predef/facial_mask_v10.png")[:,:,::-1]
        self.facial_seg = Image.open("./predef/facial_mask_v10.png")
        # self.facial_seg  = self.facial_seg.resize(self.img_size)
        self.facial_seg  = np.array(self.facial_seg ) / 255.0
        self.facial_seg = np.expand_dims(self.facial_seg, axis=2)
        self.x = 1169-150
        self.y =600-100
        self.w =2000
        self.h = 1334
        self.l = max(self.w ,self.h)
        self.total_tex = {}
        for data in tqdm(self.data_list):
            tmp = data.split('/')
            tex_path = os.path.join( self.dir_tex , tmp[0], tmp[-1] + '.png')
            tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
            tex  = np.array(tex)
            tex = tex * self.facial_seg
            tex =  tex[self.y:self.y+self.l,self.x :self.x +self.l,:]
            tex = cv2.resize(tex, (opt.loadSize,opt.loadSize), interpolation = cv2.INTER_AREA)
            self.total_tex[data] = [tex]
            # self.total_tex[data].append(tex)

            mesh_path = os.path.join( self.dir_A , data + '.obj')
            om_mesh = openmesh.read_trimesh(mesh_path)
            A_vertices = np.array(om_mesh.points()).reshape(-1)
            self.total_tex[data].append(A_vertices)
            if len(self.total_tex) == 4:
                break
        
    def __getitem__(self, index):
        t = time.time()
        tmp = self.data_list[index].split('/')
        A_id = int(tmp[0])
        A_exp = int(tmp[-1].split('_')[0])
        # id_p , 'models_reg', motion_p
        # tex 
        tex_path = os.path.join( self.dir_tex , tmp[0], tmp[-1] + '.png')
        # tex_path = '/raid/celong/FaceScape/texture_mapping/target/1/9_mouth_right.png'
        # mesh 
        # tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
        # tex  = np.array(tex) 
        # # tex = cv2.resize(tex, self.img_size, interpolation = cv2.INTER_AREA)
        # tex = tex * self.facial_seg
        # tex = tex[self.y:self.y+self.l,self.x :self.x +self.l,:]
        tex = self.total_tex[self.data_list[index]][0]
        tex = Image.fromarray(np.uint8(tex))
        params = get_params(self.opt, tex.size)
        transform = get_transform(self.opt, params)      
        A_tex_tensor = transform(tex)

        # mesh_path = os.path.join( self.dir_A , self.data_list[index] + '.obj')
        # # mesh = trimesh.load(mesh_path, process=False)
        # # vertices = mesh.vertices
        # om_mesh = openmesh.read_trimesh(mesh_path)
        # A_vertices = np.array(om_mesh.points()).reshape(-1)
        A_vertices = self.total_tex[self.data_list[index]][1]

        # if A_vertices.shape[0] != 78951:
             
        # vertices=vertices.reshape(-1, 4, 3)
        # A_vertices = vertices[:, 0, :].reshape(-1)

        toss = random.getrandbits(1)
        print('!!!!!+++++!!!!')
        print( '0' in self.id_set)

        # toss 0-> same iden, diff exp
        while True:
            try:
                if toss == 0:
                    pool = self.exp_set - set(tmp[-1])
                    B_exp = random.sample(pool, 1)[0]
                    B_id = tmp[0]
                    # Bid = 20
                # toss 1 -> same exp, diff iden
                else:
                    pool = self.id_set - set(tmp[0])
                    B_id = random.sample(pool, 1)[0]
                    B_exp = tmp[-1]
                    # Bid = 301
                
                # tex
                # DEBUG!!
                tex_index = self.data_list[index]
                # tex_index = os.path.join( B_id , 'models_reg', B_exp  )   
                # tex_path = os.path.join( self.dir_tex , B_id, B_exp + '.png')
                # # tex_path = '/raid/celong/FaceScape/texture_mapping/target/1/9_mouth_right.png'
                # # mesh 
                # tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
                # tex  = np.array(tex ) 
                # tex = tex * self.facial_seg
                # tex = tex[self.y:self.y+self.l,self.x :self.x +self.l,:]
                tex = self.total_tex[tex_index][0]
                tex = Image.fromarray(np.uint8(tex))
                
                B_tex_tensor = transform(tex)
                # mesh_path = os.path.join( self.dir_A , B_id, 'models_reg' , B_exp + '.obj')
                # om_mesh = openmesh.read_trimesh(mesh_path)
                # B_vertices = np.array(om_mesh.points()).reshape(-1)
                B_vertices = self.total_tex[tex_index][1]

                if B_vertices.shape[0] != 78951:
                    print('!!!!',B_vertices.shape )
                    continue
                break
            except:
                print('!!!!!', tex_path)
                continue
        # vertices=vertices.reshape(-1, 4, 3)
        # B_vertices = vertices[:, 0, :].reshape(-1)
        input_dict = { 'Atex': A_tex_tensor, 'Amesh': torch.FloatTensor(A_vertices),
                'A_path': self.data_list[index], 'Btex':B_tex_tensor,
                'Bmesh': torch.FloatTensor(B_vertices), 'B_path': os.path.join( B_id, 'models_reg' , B_exp),
                'map_type':toss, 'Aid': int(A_id) - 1, 'Aexp': int(A_exp) -1,
                'Bid':int(B_id) - 1, 'Bexp':int(B_exp.split('_')[0]) - 1 }

        print (input_dict['Aid'], input_dict['Aexp'], input_dict['Bid'], input_dict['Bexp'] )
        return input_dict

    def __len__(self):
        return len(self.total_tex) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeMeshTexDataset'


class FacescapeTexDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        ### input A (texture and mesh)   
        self.dir_A = os.path.join(opt.dataroot, "textured_meshes")

        self.dir_tex = '/raid/celong/FaceScape/texture_mapping/target/'

        ### input B (real images)
        self.dir_B = os.path.join(opt.dataroot, "ffhq_aligned_img")

        ### input C (eye parsing images)
        self.dir_C = os.path.join(opt.dataroot, "fsmview_landmarks")

        ### json 
        self.dir_json = os.path.join(opt.dataroot, "fsmview_images")

        self.exp_set =  get_exp()

        if opt.isTrain:
            _file = open(os.path.join(opt.dataroot, "lists/texmesh_train.pkl"), "rb")
            
        else:
            _file = open(os.path.join(opt.dataroot, "lists/texmesh_test.pkl"), "rb")
       
        self.data_list = pickle.load(_file)#[:1]
        _file.close()
        
        ids = open(os.path.join(opt.dataroot, "lists/ids.pkl"), "rb")
        self.id_set = set(pickle.load(ids))
        self.exp_set = get_exp()
        # self.facial_seg = cv2.imread("./predef/facial_mask_v10.png")[:,:,::-1]
        self.facial_seg = Image.open("./predef/facial_mask_v10.png")
        # self.facial_seg  = self.facial_seg.resize(self.img_size)
        self.facial_seg  = np.array(self.facial_seg ) / 255.0
        self.facial_seg = np.expand_dims(self.facial_seg, axis=2)
        self.x = 1019
        self.y =500
        self.w =2000
        self.h = 1334
        self.l = max(self.w,self.h)
        self.total_tex = []
        for data in tqdm(self.data_list):
            tmp = data.split('/')
            tex_path = os.path.join( self.dir_tex , tmp[0], tmp[-1] + '.png')
            tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
            tex  = np.array(tex ) 
            tex = tex * self.facial_seg
            tex =  tex[self.y:self.y+self.l,self.x :self.x +self.l,:]
            tex = cv2.resize(tex, (opt.loadSize,opt.loadSize), interpolation = cv2.INTER_AREA)
            self.total_tex.append(tex)
            # if len(self.total_tex) == 33:
            #     break
    def __getitem__(self, index):
        t = time.time()
        tmp = self.data_list[index].split('/')
        # id_p , 'models_reg', motion_p
        # tex 
        tex_path = os.path.join( self.dir_tex , tmp[0], tmp[-1] + '.png')
        # tex_path = '/raid/celong/FaceScape/texture_mapping/target/1/9_mouth_right.png'
        # mesh 
        # tex = Image.open(tex_path).convert('RGB')#.resize(self.img_size)
        # tex  = np.array(tex ) 
        # # tex = cv2.resize(tex, self.img_size, interpolation = cv2.INTER_AREA)
        # tex = tex * self.facial_seg
        # tex = tex[self.y:self.y+self.l,self.x :self.x +self.l,:]
        tex = self.total_tex[index]
        tex = Image.fromarray(np.uint8(tex))
        params = get_params(self.opt, tex.size)
        transform = get_transform(self.opt, params)      
        tex_tensor = transform(tex)

        input_dict = { 'tex':tex_tensor, 'id': int(tmp[0]) - 1, 'exp': int(tmp[-1].split('_')[0] )- 1, 'path': self.data_list[index]}
        # if input_dict['id'] > 300 or input_dict['id']< 0 or input_dict['exp'] > 19  or input_dict['exp']< 0 :
        #     print(input_dict['path'])
        #     print('*************')
       
        return input_dict

    def __len__(self):
        return len(self.total_tex) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeTexDataset'
import pyrender, trimesh, cv2, openmesh, os
import numpy as np
import src.renderer as renderer
import src.camera as camera
import src.utility as util

# read model 
model = trimesh.load_mesh("/nfs/STG/CodecAvatar/lelechen/Facescape/textured_meshes/1/models_reg/1_neutral.obj", process=False)

# get vertices using openmesh, because trimesh doesn't preserve vertex number and order
om_mesh = openmesh.read_trimesh("/nfs/STG/CodecAvatar/lelechen/Facescape/textured_meshes/1/models_reg/1_neutral.obj") 
verts = om_mesh.points()

# set material
model.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)

# set K Rt (cv camera coordinate)
K = np.array([[2000, 0 , 499.5],
              [0, 2000, 499.5],
              [0, 0, 1]])
Rt = np.array([[1, 0 , 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 600]])

# render
_, color = renderer.render_cvcam(model, K, Rt, scale=1.0, 
                                 rend_size=(1000, 1000), flat_shading=True)

# read landmark indices, 'v10' is for TU model, bilinear model 1.0/1.2/1.3
lm_list_v10 = np.load("./predef/landmark_indices.npz")['v10']

# make camera for projection
cam = camera.CamPara(K = K, Rt = Rt)

# draw landmarks
color_draw = color.copy()
for ind, lm_ind in enumerate(lm_list_v10):
    uv = cam.project(verts[lm_ind])
    u, v = np.round(uv).astype(np.int)
    color_draw = cv2.circle(color_draw, (u, v), 10, (100, 100, 100), -1)
    color_draw = cv2.putText(color_draw, "%02d"%(ind), (u-8, v+4), 
                             fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale = 0.4,
                             color = (255, 255, 255))

# save out
os.makedirs("./demo_output/", exist_ok = True)
cv2.imwrite("./demo_output/lm_result.jpg", color_draw)

# util.show_img_arr(color_draw, bgr_mode = True)
# print("results saved to './demo_output/lm_result.jpg'")
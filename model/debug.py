import sys

sys.path.insert(1, '/raid/celong/lele/github/lighting')

from util.render_class import MeshRender

ms = MeshRender()


id_idx = 140
exp_idx = 4
cam_idx = 1

mesh_path = f"{mesh_root}/{id_idx}/models_reg/{expressions[exp_idx]}.obj"

om_mesh = openmesh.read_trimesh(mesh_path)
om_vertices = np.array(om_mesh.points()).reshape(-1)
om_vertices = torch.from_numpy(om_vertices.astype(np.float32))

img = ms.meshrender(id_idx, exp_idx, om_vertices)

imageio.imwrite("fkass.png", img)
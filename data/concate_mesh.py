import os

def concate_mesh():
	ids = os.listdir('/data/home/us000042/lelechen/data/Facescape/textured_meshes')
	for i in ids:
		originalpath = '/data/home/us000042/lelechen/data/Facescape/textured_meshes/%s/models_reg'%i
		files = os.listdir(originalpath)
		augpath = '/data/home/us000042/lelechen/data/Facescape/augmented_meshes/%s/models_reg' % i
		for j in files:
			if j[-3:] =='obj':
				objpath = os.path.join( originalpath, j)
				print (objpath)
				command = 'cp ' + objpath +' ' + augpath
				print(command)
concate_mesh()
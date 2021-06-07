### Using labels only
#  CUDA_VISIBLE_DEVICES=6,7,5,4,3,2,1,0  python -m pdb -c continue step1_texmesh_train.py --datasetname fs_texmesh --name texmesh_step1 --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 8

  CUDA_VISIBLE_DEVICES=2,3,1,0  python gan.py --datasetname fs_texmesh  --name texmesh_step1 --gpu_ids 0,1,2,3 --batchSize 16 --lr 0.0005
  # --no_mismatch_loss --no_cls_loss --no_mesh_loss --no_vgg_loss
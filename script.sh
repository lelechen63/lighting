### Using labels only
#  CUDA_VISIBLE_DEVICES=6,7,5,4,3,2,1,0  python -m pdb -c continue step1_texmesh_train.py --datasetname fs_texmesh --name texmesh_step1 --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 8

# CUDA_VISIBLE_DEVICES=2 python gan.py  --datasetname fs_tex  --name texgan --gpu_ids 0 --batchSize 2 --lr 0.0005 --isTrain 


CUDA_VISIBLE_DEVICES=2 python gan.py    --name texgan --gpu_ids 0 --batchSize 1 --lr 0.0005 --isTrain --debug

  # --no_mismatch_loss --no_cls_loss --no_mesh_loss --no_vgg_loss --no_pix_loss
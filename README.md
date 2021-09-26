# lighting

train: CUDA_VISIBLE_DEVICES=2 python gan.py --name texgan --gpu_ids 0 --batchSize 1   --lr 0.0005    #--isTrain

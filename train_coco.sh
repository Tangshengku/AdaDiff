# the training setting
num_processes=4  # the number of gpus you have, e.g., 2
train_script=train_t2i_discrete.py  # the train script, one of <train.py|train_ldm.py|train_ldm_discrete.py|train_t2i_discrete.py>
                       # train.py: training on pixel space
                       # train_ldm.py: training on latent space with continuous timesteps
                       # train_ldm_discrete.py: training on latent space with discrete timesteps
                       # train_t2i_discrete.py: text-to-image training on latent space
config=configs/mscoco_uvit_small.py # the training configuration
                                      # you can change other hyperparameters by modifying the configuration file
pretrained_weight=/home/dongk/dkgroup/tsk/projects/U-ViT/ckpt/mscoco_uvit_small.pth
# launch training
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py\
         --config=configs/mscoco_uvit_small.py --is_train=True --pretrained_weight=${pretrained_weight}

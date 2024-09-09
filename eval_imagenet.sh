# the evaluation setting
export num_processes=4  # the number of gpus you have, e.g., 2
export eval_script=eval.py  # the evaluation script, one of <eval.py|eval_ldm.py|eval_ldm_discrete.py|eval_t2i_discrete.py>
                     # eval.py: for models trained with train.py (i.e., pixel space models)
                     # eval_ldm.py: for models trained with train_ldm.py (i.e., latent space models with continuous timesteps)
                     # eval_ldm_discrete.py: for models trained with train_ldm_discrete.py (i.e., latent space models with discrete timesteps)
                     # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
export config=configs/celeba64_uvit_small.py  # the training configuration

threshold=0.92

# launch evaluation --multi_gpu
accelerate launch  --num_processes 1 --mixed_precision fp16 eval_ldm.py \
         --config=configs/imagenet256_uvit_large.py \
         --exit_threshold=${threshold} \
          --nnet_path=/home/dongk/dkgroup/tsk/projects/U-ViT/workdir/imagenet256_uvit_large/deediff/ckpts/110000.ckpt/nnet_ema.pth
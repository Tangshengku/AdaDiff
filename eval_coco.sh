
threshold=0.87

accelerate launch --multi_gpu --num_processes 4  --main_process_port 29501 --mixed_precision fp16 eval_t2i_discrete.py\
            --config=configs/mscoco_uvit_small.py \
            --exit_threshold=${threshold} \
            --nnet_path=/home/dongk/dkgroup/tsk/projects/U-ViT/workdir/330000.ckpt/nnet_ema.pth

# Stage 1 (pixel branch, 256x512) as trained for the dev release checkpoint mp3d_stage1_pixel_256x512
# (checkpoint-24000 of this recipe, selected by WS-PSNR on the validation split mp3d_double_256_val).
# omni_gs_160x320_mp3d_cylinder_pixel_256.py with batch 4 per GPU (3 GPUs), lr 4e-4 (best of the
# 2e-4 / 4e-4 / 8e-4 probes), checkpoints every 1500 steps; 15 epochs, seed 0 as in the base config.
# Train: train.py --entry mp3d_double_256 --switch ddp_forward=true (see README, "Reproducing the dev release").
# The backbone starts from the PanSplat checkpoint: set model.backbone_ckpt (README, "Weights").
_base_ = ['../omni_gs_160x320_mp3d_cylinder_pixel_256.py']

lr = 4e-4
save_freq = 1500
val_freq = 1500
resume_from = False
dataset_params = dict(batch_size_train=4)
model = dict(dataset_params=dict(batch_size_train=4))  # the copy the base config put into the model

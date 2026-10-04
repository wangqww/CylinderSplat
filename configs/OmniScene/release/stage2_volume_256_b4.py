# Stage 2 (volume branch, 256x512; pixel branch frozen) as trained for the dev release checkpoint
# mp3d_stage2_volume_256x512 (checkpoint-24000, selected on mp3d_double_256_val).
# omni_gs_160x320_mp3d_cylinder_volume_256.py with batch 4 per GPU (3 GPUs), lr 4e-4 (best probe),
# checkpoints every 1500 steps; 15 epochs, seed 0 as in the base config.
# Train from the stage-1 weights: --resume-from <stage 1> --transfer stage1_to_stage2 --switch ddp_forward=true.
_base_ = ['../omni_gs_160x320_mp3d_cylinder_volume_256.py']

lr = 4e-4
save_freq = 1500
val_freq = 1500
resume_from = False
dataset_params = dict(batch_size_train=4)
model = dict(dataset_params=dict(batch_size_train=4))  # the copy the base config put into the model

# 360Loc fine-tune (256x512) as trained for the dev release checkpoint loc360_finetune_256x512 (the final
# weights, checkpoint-21670 after 10 epochs; PanSplat's protocol tests the last weights).
# omni_gs_160x320_360Loc_cylinder_all_256.py with lr 2e-4 (the stage-3 lr), 10 epochs and seed 1111.
# Train from the stage-3 weights on 3 GPUs (batch 2 each):
#   train.py --entry loc360_all_256 --resume-from <stage 3> --transfer mp3d_all_256_to_loc360_pan2 --save-final
#     --switch ddp_forward=true --switch loc360_interleave=true --switch depth_valid_mask=true
_base_ = ['../omni_gs_160x320_360Loc_cylinder_all_256.py']

lr = 2e-4
max_epochs = 10
seed = 1111
resume_from = False
dataset_params = dict(seed=1111)

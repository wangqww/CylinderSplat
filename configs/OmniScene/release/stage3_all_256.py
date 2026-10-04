# Stage 3 (both branches jointly, 256x512) as trained for the dev release checkpoint mp3d_stage3_joint_256x512
# (checkpoint-78000, selected on mp3d_double_256_val). omni_gs_160x320_mp3d_cylinder_all_256.py unchanged
# (batch 2 per GPU on 3 GPUs, lr 2e-4 = the best probe, 25 epochs, seed 0) except that it does not resume
# from the author's path: pass --resume-from <stage 2> --transfer stage2_to_stage3 --switch ddp_forward=true.
_base_ = ['../omni_gs_160x320_mp3d_cylinder_all_256.py']

resume_from = False

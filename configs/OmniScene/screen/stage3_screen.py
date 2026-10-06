# Base of the stage-3 fine-tunes: a continuation of the released stage-3 checkpoint (mp3d_stage3_joint_256x512) on
# row mp3d_double_256_screen (one process, batch 2, OneCycle over onecycle_total_steps with max_lr = lr, seed 1111).
# long_c0.py / long_s.py set the schedule actually used; launch through scripts/long_arm.sh.
_base_ = ['../release/stage3_all_256.py']

seed = 1111
onecycle_total_steps = 6100
save_freq = 3000

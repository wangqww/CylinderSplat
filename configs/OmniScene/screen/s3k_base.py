# Volume sparsity, stage-3 base: the LS recipe (long_s.py: lr 5e-5, batch 2, the LS switches, depth weight 0.1) for
# 8,000 steps on a OneCycle over 8,100 steps, checkpoints every 4,000 steps, from the LS checkpoint (scripts/long_arm.sh
# with CYLINDERSPLAT_INIT, transfer exact; README, "Volume sparsity").
_base_ = ['./long_s.py']

onecycle_total_steps = 8100
save_freq = 4000

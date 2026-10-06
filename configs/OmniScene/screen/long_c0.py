# The released stage-3 checkpoint fine-tuned 20,000 steps (two passes over the training sequences in the released
# order) on a OneCycle over 20,100 steps with peak lr 5e-5, checkpoints every 5,000 steps, prune_invisible on. Base of
# the LS recipe (long_s.py). Launch with scripts/long_arm.sh <config> <gpu>: trains --max-steps 20000 --save-final,
# evaluates every checkpoint on val, picks one on val and reads test once for it.
_base_ = ['./stage3_screen.py']

lr = 5e-5
onecycle_total_steps = 20100
save_freq = 5000
switches = dict(prune_invisible=True)

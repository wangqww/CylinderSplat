# LS recipe (README, "LS fine-tune"): long_c0 + sampling_align, ws_loss and lpips_input_range, and the fused depth
# weight 0.1 (a loss override, model.loss_args), validated as one bundle against long_c0 and the released stage 3
# (README table). Transfer exact from the released stage 3: no switch adds parameters.
_base_ = ['./long_c0.py']

switches = dict(lpips_input_range=True, sampling_align=True, ws_loss=True)
model = dict(loss_args=dict(weight_depth_abs=0.1))

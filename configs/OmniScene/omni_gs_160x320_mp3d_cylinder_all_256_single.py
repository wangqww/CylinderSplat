# D3 arm (plan §7, Phase-2 screen): the stage-3 all_256 model built with num_frames=1 for the
# single-view MP3D loader. Everything else is omni_gs_160x320_mp3d_cylinder_all_256.py; mmengine
# merges the dicts below into it (_base_), so only these keys change:
#   exp_name, resume_from, model.pixel_gs.num_frames = 1.
# Init: REF-T1's 2-view all_256 checkpoint, loaded with --transfer d3_single_view (same names and
# shapes: the UNet shapes do not depend on num_frames). The pair is this config without and with
# --switch v1_identity_pose=true:
#   accelerate launch --config-file configs/accelerate/accel_1proc.yaml train.py \
#       --entry screen_mp3d_single_256 \
#       --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256_single.py \
#       --run-id screen_d3_on --transfer d3_single_view --switch v1_identity_pose=true
# The plan's Pan2 v=1 arm (PixelGaussian360Loc(num_frames=1) on 360Loc) still needs a single-view
# 360Loc loader, which does not exist yet.

_base_ = [
    './omni_gs_160x320_mp3d_cylinder_all_256.py',
]

exp_name = "omni_gs_160x320_mp3d_cylinder_single_all_256"

resume_from = '/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/checkpoint-48000/model.safetensors'

model = dict(
    pixel_gs=dict(
        num_frames=1,
    ),
)

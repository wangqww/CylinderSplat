# The joint model (all_256) built with num_frames=1 for the single-view MP3D loader. Everything else is
# omni_gs_160x320_mp3d_cylinder_all_256.py (mmengine merges the dicts below into it), so only exp_name,
# resume_from and model.pixel_gs.num_frames change. Train with --entry mp3d_single_256 from a two-view all_256
# checkpoint (--transfer exact: no weight shape depends on num_frames).

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

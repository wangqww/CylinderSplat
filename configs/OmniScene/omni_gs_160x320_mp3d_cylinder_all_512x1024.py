# CylinderSplat stage 4: the joint model (all_256 architecture) at 512x1024, initialised from the
# stage-3 (256x512) checkpoint. A full copy of omni_gs_160x320_mp3d_cylinder_all_256.py with only these
# values changed:
#   resolution = [512, 1024]          (camera_args.resolution, loss_args.perceptual_resolution,
#                                       pixel_gs.image_height and dataset_params.resolution follow it)
#   dataset_params batch_size_train 2 -> 1, batch_size_test 4 -> 1 (batch_size_val stays 1)
#   max_epochs 25 -> 10, resume_from -> False, exp_name
#   lr stays 2e-4 (the stage-3 lr); save_freq and val_freq stay 3000.
# Train with the rows mp3d_double_512_ddp3 / mp3d_double_512_ddp4 (train.py, --transfer stage3_to_stage4_512)
# and evaluate with mp3d_double_512_full (test) / mp3d_double_512_full_val (validation). Batch 1 needs about
# 26 GB per GPU (L40); it does not fit a 24 GB RTX 4090 for training (evaluation does).

import math

_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

exp_name = 'omni_gs_160x320_mp3d_cylinder_double_all_512x1024'
output_dir = "/data/qiwei/nips25/workdirs"

lr = 2e-4  # the lr of the selected stage-3 (256x512) run; the per-run configs set the run's own lr
grad_max_norm = 1.0
print_freq = 100
save_freq = 3000
val_freq = 3000
max_epochs = 10
save_epoch_freq = -1

lr_scheduler_type = "constant_with_warmup"
max_train_steps = 1000
volume_train_steps = 18000
warmup_steps = 1000
mixed_precision = "no"
gradient_accumulation_steps = 1
# resume_from = '/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random/checkpoint-36000/model.safetensors'
# resume_from = '/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256/checkpoint-36000/model.safetensors'
resume_from = False  # the init is passed with --resume-from <stage-3 selection> --transfer stage3_to_stage4_512
# resume_from = False
report_to = "tensorboard"

volume_only = False
use_checkpoint = True
seed = 0
use_center, use_first, use_last = True, False, False
resolution = [512, 1024]
# resolution = [80, 80]
# point_cloud_range = [-20.0, -20.0, -3.0, 20.0, 20.0, 3.0]

point_cloud_range = [0.0, 0.0, -7.0, 10.0, 6.28, 3.0] # r, phi, z
scale_theta = 1
scale_r = 1
scale_z = 1


dataset_params = dict(
    dataset_name="nuScenesDataset",
    seed=seed,
    resolution=resolution,
    pc_range=point_cloud_range,
    use_center=use_center,
    use_first=use_first,
    use_last=use_last,
    batch_size_train=1,  # per process; the per-run configs set the batch the memory probe chose
    batch_size_val=1,
    batch_size_test=1,
    num_workers=32,
    num_workers_val=32,
    num_workers_test=32
)

near = 0.1
far = 15.0
camera_args = dict(
    resolution=resolution,
    znear=near,
    zfar=far
)

eval_args = dict(
    save_vis=True,
    save_ply=True
)

loss_args = dict(
    recon_loss_type="l2",
    recon_loss_vol_type="l2_mask",
    perceptual_loss_vol_type="mask",
    depth_abs_loss_vol_type="mask",
    mask_dptm=True,
    perceptual_resolution=[resolution[0], resolution[1]],
    weight_recon=1.0,
    weight_perceptual=0.05,
    weight_depth_abs=1.0,
    weight_recon_vol=0.1,
    weight_perceptual_vol=0.005,
    weight_depth_abs_vol=0.1,
    weight_volume_loss=0.0 #0.1
)

pc_range = point_cloud_range
pc_xrange, pc_yrange, pc_zrange = pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]

_dim_ = 128
num_heads = 8
num_layers = 1
_ffn_dim_ = _dim_ * 2

tpv_theta_ = 128  # theta
tpv_r_ = 16  # r
tpv_z_ = 64  # z

# tpv_theta_ = 256  # theta
# tpv_r_ = 16  # r
# tpv_z_ = 128  # z

gpv = 3 # TODO: Change to 3

near_num_points_in_pillar = [32, 8, 64] # thetar ztheta rz
near_num_points = [64, 16, 128]

# near_num_points_in_pillar = [32, 8, 64] # thetar ztheta rz
# near_num_points = [64, 16, 128]


hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_theta_,
            tpv_w=tpv_r_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1),
        dict(
            type='TPVImageCrossAttention',
            pc_range=point_cloud_range,
            dropout=0.1,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=near_num_points,
                num_z_anchors=near_num_points_in_pillar,
                num_levels=1,
                floor_sampling_offset=False,
                tpv_h=tpv_theta_,
                tpv_w=tpv_r_,
                tpv_z=tpv_z_),
            embed_dims=_dim_,
            tpv_h=tpv_theta_,
            tpv_w=tpv_r_,
            tpv_z=tpv_z_)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
    # operation_order=('self_attn', 'norm', 'ffn', 'norm'),
)

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_theta_,
            tpv_w=tpv_r_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm'))

model = dict(
    type='OmniGaussianCylinderAll',
    use_checkpoint=use_checkpoint,
    point_cloud_range=point_cloud_range,
    with_pixel=True,
    volume_only=volume_only,
    backbone=dict(
        type='BackboneResnet',
        feature_channels=[128, 96, 64, 32],
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_cross_attn=False,
        num_head=1,
    ),
    pixel_gs=dict(
        type="PixelGaussian",
        use_checkpoint=use_checkpoint,
        image_height=resolution[0],
        patchs_height=1,
        patchs_width=1,
        gh_cnn_layers=3,
        gaussians_per_pixel=1,
    ),
    volume_gs=dict(
        type="VolumeGaussianCylinder",
        use_checkpoint=use_checkpoint,
        encoder=dict(
            type='TPVFormerEncoderCylinder',
            tpv_theta=tpv_theta_,
            tpv_r=tpv_r_,
            tpv_z=tpv_z_,
            num_feature_levels=1,
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=near_num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer, self_cross_layer, self_layer
            ],
            embed_dims=_dim_,
            positional_encoding=dict(
                type='TPVFormerPositionalEncoding',
                num_feats=[32, 48, 48],
                h=tpv_theta_,
                w=tpv_r_,
                z=tpv_z_)),
        gs_decoder = dict(
            type='VolumeGaussianDecoderCylinder',
            tpv_theta=tpv_theta_,
            tpv_r=tpv_r_,
            tpv_z=tpv_z_,
            pc_range=point_cloud_range,
            gs_dim=14,
            in_dims=_dim_,
            hidden_dims=2*_dim_,
            out_dims=_dim_,
            scale_theta=scale_theta,
            scale_r=scale_r,
            scale_z=scale_z,
            gpv=gpv,
            offset_max=[0.5, 0.5, 0.5],
            scale_max=[0.5, 0.5, 0.5],
        )
    ),
    camera_args=camera_args,
    loss_args=loss_args,
    dataset_params=dataset_params
)

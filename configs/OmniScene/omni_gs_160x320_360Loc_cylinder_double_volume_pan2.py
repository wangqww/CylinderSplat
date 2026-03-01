import math

_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

exp_name = "omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2"
output_dir = "/data/qiwei/nips25/workdirs"

lr = 1e-4 #1e-4
grad_max_norm = 1.0
print_freq = 100
save_freq = 3000
val_freq = 3000
max_epochs = 30
save_epoch_freq = -1

lr_scheduler_type = "constant_with_warmup"
max_train_steps = 5000
volume_train_steps = 18000
warmup_steps = 500
mixed_precision = "no"
gradient_accumulation_steps = 1
resume_from = "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_z3/checkpoint-36000/model.safetensors"
# resume_from = False
report_to = "tensorboard"

volume_only = False
use_checkpoint = True
seed = 0
use_center, use_first, use_last = True, False, False
resolution = [160, 320]
# resolution = [80, 80]
# point_cloud_range = [-20.0, -20.0, -3.0, 20.0, 20.0, 3.0]

near_point_cloud_range = [0.0, 0.0, -3.0, 16.0, 6.28, 3.0] # r, phi, z
far_point_cloud_range = [0.0, 0.0, -3.0, 16.0, 6.28, 3.0]
point_cloud_range = [0.0, 0.0, -3.0, 16.0, 6.28, 3.0] # r, phi, z
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
    batch_size_train=6,
    batch_size_val=6,
    batch_size_test=6,
    num_workers=32,
    num_workers_val=32,
    num_workers_test=32
)

num_cams = 2
near = 0.1
far = 15.0
camera_args = dict(
    resolution=resolution,
    znear=near,
    zfar=far
)

eval_args = dict(
    save_vis=True,
    save_ply=False
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
    weight_depth_abs=0.1,
    weight_recon_vol=0,
    weight_perceptual_vol=0,
    weight_depth_abs_vol=0,
    weight_volume_loss=0.05 #0.1
)

pc_range = point_cloud_range
pc_xrange, pc_yrange, pc_zrange = pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]

_dim_ = 128
num_heads = 8
num_layers = 1
patch_sizes=[8, 8, 4, 2]
_ffn_dim_ = _dim_ * 2

near_tpv_theta_ = 64  # theta
near_tpv_r_ = 16  # r
near_tpv_z_ = 32  # z

far_tpv_theta_ = 64  # theta
far_tpv_r_ = 16  # r
far_tpv_z_ = 32  # z

gpv = 3

near_num_points_in_pillar = [16, 8, 32] # thetar ztheta rz
near_num_points = [32, 16, 64]

far_num_points_in_pillar = [32, 4, 32]
far_num_points = [64, 8, 64]

# near_num_points_in_pillar = [4, 4, 4]
# near_num_points = [8, 8, 8]

# far_num_points_in_pillar = [16, 16, 16]
# far_num_points = [32, 32, 32]


hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

near_self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=near_tpv_theta_,
            tpv_w=near_tpv_r_,
            tpv_z=near_tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1),
        dict(
            type='TPVImageCrossAttention',
            pc_range=near_point_cloud_range,
            num_cams=num_cams,
            dropout=0.1,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=near_num_points,
                num_z_anchors=near_num_points_in_pillar,
                num_levels=1,
                floor_sampling_offset=False,
                tpv_h=near_tpv_theta_,
                tpv_w=near_tpv_r_,
                tpv_z=near_tpv_z_),
            embed_dims=_dim_,
            tpv_h=near_tpv_theta_,
            tpv_w=near_tpv_r_,
            tpv_z=near_tpv_z_)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
    # operation_order=('self_attn', 'norm', 'ffn', 'norm'),
    )

near_self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=near_tpv_theta_,
            tpv_w=near_tpv_r_,
            tpv_z=near_tpv_z_,
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

far_self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=far_tpv_theta_,
            tpv_w=far_tpv_r_,
            tpv_z=far_tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
            dropout=0.1),
        dict(
            type='TPVImageCrossAttention',
            pc_range=far_point_cloud_range,
            num_cams=num_cams,
            dropout=0.1,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=far_num_points,
                num_z_anchors=far_num_points_in_pillar,
                num_levels=1,
                floor_sampling_offset=False,
                tpv_h=far_tpv_theta_,
                tpv_w=far_tpv_r_,
                tpv_z=far_tpv_z_),
            embed_dims=_dim_,
            tpv_h=far_tpv_theta_,
            tpv_w=far_tpv_r_,
            tpv_z=far_tpv_z_)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
    # operation_order=('self_attn', 'norm', 'ffn', 'norm'),
    )

far_self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=far_tpv_theta_,
            tpv_w=far_tpv_r_,
            tpv_z=far_tpv_z_,
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
    type='OmniGaussianCylinderVolume360LocPan2',
    use_checkpoint=use_checkpoint,
    near_point_cloud_range=near_point_cloud_range,
    far_point_cloud_range=far_point_cloud_range,
    with_pixel=True,
    volume_only=volume_only,
    backbone=dict(
        type='BackboneResnet',
        d_in=3,),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    pixel_gs=dict(
        type="PixelGaussian360Loc",
        use_checkpoint=use_checkpoint,
        down_block=dict(
            type='MVDownsample2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        up_block=dict(
            type='MVUpsample2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        mid_block=dict(
            type='MVMiddle2D',
            num_layers=num_layers,
            resnet_act_fn="silu",
            resnet_groups=32,
            num_attention_heads=num_heads,
            num_views=num_cams),
        patch_sizes=patch_sizes,
        in_embed_dim=_dim_,
        out_embed_dims=[_dim_, _dim_*2, _dim_*4, _dim_*4],
        num_cams=num_cams,
        near=near,
        far=far),
    near_volume_gs=dict(
        type="VolumeGaussianCylinder",
        use_checkpoint=use_checkpoint,
        encoder=dict(
            type='TPVFormerEncoderCylinder',
            tpv_theta=near_tpv_theta_,
            tpv_r=near_tpv_r_,
            tpv_z=near_tpv_z_,
            num_feature_levels=1,
            num_layers=3,
            pc_range=near_point_cloud_range,
            num_cams=num_cams,
            num_points_in_pillar=near_num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                near_self_cross_layer, near_self_cross_layer, near_self_layer
            ],
            embed_dims=_dim_,
            positional_encoding=dict(
                type='TPVFormerPositionalEncoding',
                num_feats=[32, 48, 48],
                h=near_tpv_theta_,
                w=near_tpv_r_,
                z=near_tpv_z_)),
        gs_decoder = dict(
            type='VolumeGaussianDecoderCylinderOri',
            tpv_theta=near_tpv_theta_,
            tpv_r=near_tpv_r_,
            tpv_z=near_tpv_z_,
            pc_range=near_point_cloud_range,
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
    far_volume_gs=dict(
        type="VolumeGaussianCylinder",
        use_checkpoint=use_checkpoint,
        encoder=dict(
            type='TPVFormerEncoderCylinder',
            tpv_theta=far_tpv_theta_,
            tpv_r=far_tpv_r_,
            tpv_z=far_tpv_z_,
            num_feature_levels=1,
            num_layers=3,
            pc_range=far_point_cloud_range,
            num_cams=num_cams,
            num_points_in_pillar=far_num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                far_self_cross_layer, far_self_cross_layer, far_self_layer
            ],
            embed_dims=_dim_,
            positional_encoding=dict(
                type='TPVFormerPositionalEncoding',
                num_feats=[32, 64, 32],
                h=far_tpv_theta_,
                w=far_tpv_r_,
                z=far_tpv_z_)),
        gs_decoder = dict(
            type='VolumeGaussianDecoderCylinderOri',
            tpv_theta=far_tpv_theta_,
            tpv_r=far_tpv_r_,
            tpv_z=far_tpv_z_,
            pc_range=far_point_cloud_range,
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

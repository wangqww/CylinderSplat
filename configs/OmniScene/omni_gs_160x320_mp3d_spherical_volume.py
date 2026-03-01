import math

_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

exp_name = "omni_gs_160x320_mp3d_spherical_Double_Volume"
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
resume_from = '/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel1/checkpoint-36000/model.safetensors'
# resume_from = False
report_to = "tensorboard"

volume_only = False
use_checkpoint = True
seed = 0
use_center, use_first, use_last = True, False, False
resolution = [160, 320]
# resolution = [80, 80]

point_cloud_range = [0.0, 0.0, 0.0, 6.28, 3.14, 10.0]

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
    weight_recon_vol=1.0,
    weight_perceptual_vol=0.05,
    weight_depth_abs_vol=0.1,
    weight_volume_loss=0.0 #0.1
)

pc_range = point_cloud_range
pc_xrange, pc_yrange, pc_zrange = pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]

_dim_ = 128
num_heads = 8
num_layers = 1
patch_sizes=[8, 8, 4, 2]
_ffn_dim_ = _dim_ * 2

tpv_theta_ = 64 # theta
tpv_phi_ = 32 # r
tpv_r_ = 16 # phi

gpv = 3

num_points_in_pillar = [8, 16, 32]
num_points = [16, 32, 64]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_theta_,
            tpv_w=tpv_phi_,
            tpv_z=tpv_r_,
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
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=1,
                floor_sampling_offset=False,
                tpv_h=tpv_theta_,
                tpv_w=tpv_phi_,
                tpv_z=tpv_r_),
            embed_dims=_dim_,
            tpv_h=tpv_theta_,
            tpv_w=tpv_phi_,
            tpv_z=tpv_r_)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_theta_,
            tpv_w=tpv_phi_,
            tpv_z=tpv_r_,
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
    type='OmniGaussianSphericalVolume',
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
    ),
    volume_gs=dict(
        type="VolumeGaussianSpherical",
        use_checkpoint=use_checkpoint,
        encoder=dict(
            type='TPVFormerEncoderSpherical',
            tpv_theta=tpv_theta_,
            tpv_phi=tpv_phi_,
            tpv_r=tpv_r_,
            num_feature_levels=1,
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=num_points_in_pillar,
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
                w=tpv_phi_,
                z=tpv_r_,
            )
        ),
        gs_decoder = dict(
            type='VolumeGaussianDecoderSpherical',
            tpv_theta=tpv_theta_,
            tpv_phi=tpv_phi_,
            tpv_r=tpv_r_,
            pc_range=point_cloud_range,
            gs_dim=14,
            in_dims=_dim_,
            hidden_dims=2*_dim_,
            out_dims=_dim_,
            gpv=gpv
        )
    ),
    camera_args=camera_args,
    loss_args=loss_args,
    dataset_params=dataset_params
)

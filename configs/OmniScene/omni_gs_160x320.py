_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

exp_name = "omni_gs_160x320"
output_dir = "workdirs"

lr = 1e-4
grad_max_norm = 1.0
print_freq = 100
save_freq = 5000
val_freq = 5000
max_epochs = 30
save_epoch_freq = -1

lr_scheduler_type = "constant_with_warmup"
max_train_steps = 5000
warmup_steps = 1000
mixed_precision = "no"
gradient_accumulation_steps = 1
resume_from = "latest"
report_to = "tensorboard"

volume_only = False
use_checkpoint = True
seed = 0
use_center, use_first, use_last = True, False, False
resolution = [160, 320]
# resolution = [80, 80]
# point_cloud_range = [-50.0, -50.0, -3.0, 50.0, 50.0, 12.0]
point_cloud_range = [-30.0, -20.0, -30.0, 30.0, 3.0, 30.0]
dataset_params = dict(
    dataset_name="nuScenesDataset",
    seed=seed,
    resolution=resolution,
    pc_range=point_cloud_range,
    use_center=use_center,
    use_first=use_first,
    use_last=use_last,
    batch_size_train=2,
    batch_size_val=2,
    batch_size_test=2,
    num_workers=32,
    num_workers_val=32,
    num_workers_test=32
)

num_cams = 1
near = 0.1
far = 1000.0
camera_args = dict(
    resolution=resolution,
    znear=near,
    zfar=far
)

eval_args = dict(
    save_vis=False,
    save_ply=False
)

loss_args = dict(
    recon_loss_type="l2",
    recon_loss_vol_type="l2_mask",
    perceptual_loss_vol_type="mask",
    depth_abs_loss_vol_type="mask_self",
    mask_dptm=True,
    perceptual_resolution=[resolution[0], resolution[1]],
    weight_recon=1.0,
    weight_perceptual=0.05,
    weight_depth_abs=0.01,
    weight_recon_vol=1.0,
    weight_perceptual_vol=0.05,
    weight_depth_abs_vol=0.01,
)

pc_range = point_cloud_range
pc_xrange, pc_yrange, pc_zrange = pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]

_dim_ = 128
num_heads = 8
num_layers = 1
patch_sizes=[8, 8, 4, 2]
_ffn_dim_ = _dim_ * 2

tpv_h_ = 16
tpv_w_ = 64
tpv_z_ = 64
scale_h = 1
scale_w = 1
scale_z = 1
gpv = 3

# num_points_in_pillar = [8, 16, 16]
# num_points = [16, 32, 32]
num_points_in_pillar = [16, 16, 8]
num_points = [32, 32, 16]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
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
            num_cams=1,
            dropout=0.1,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=1,
                floor_sampling_offset=False,
                tpv_h=tpv_h_,
                tpv_w=tpv_w_,
                tpv_z=tpv_z_),
            embed_dims=_dim_,
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_)
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
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
    type='OmniGaussianOriginal',
    use_checkpoint=use_checkpoint,
    with_pixel=True,
    volume_only=volume_only,
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/dino_resnet50_pretrain.pth',
            prefix=None)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    pixel_gs=dict(
        type="PixelGaussian",
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
    volume_gs=dict(
        type="VolumeGaussianOriginal",
        use_checkpoint=use_checkpoint,
        encoder=dict(
            type='TPVFormerEncoderOriginal',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_feature_levels=1,
            num_layers=3,
            pc_range=point_cloud_range,
            num_cams=1,
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
                h=tpv_h_,
                w=tpv_w_,
                z=tpv_z_)),
        gs_decoder = dict(
            type='VolumeGaussianDecoderOriginal',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            pc_range=point_cloud_range,
            gs_dim=14,
            in_dims=_dim_,
            hidden_dims=2*_dim_,
            out_dims=_dim_,
            scale_h=scale_h,
            scale_w=scale_w,
            scale_z=scale_z,
            gpv=gpv,
            offset_max=[2 * pc_xrange / (tpv_w_*scale_w), 2 * pc_yrange / (tpv_h_*scale_h), 2 * pc_zrange / (tpv_z_*scale_z)],
            scale_max=[2 * pc_xrange / (tpv_w_*scale_w), 2 * pc_yrange / (tpv_h_*scale_h), 2 * pc_zrange / (tpv_z_*scale_z)]
        )
    ),
    camera_args=camera_args,
    loss_args=loss_args,
    dataset_params=dataset_params
)

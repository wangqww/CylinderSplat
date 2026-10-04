"""Evaluation rows for evaluate.py (--dataset).

Each row fixes which loader and split are read, where the batch size comes
from, which scene keys are reported, which metrics are computed and printed
(the per-batch, per-scene and Total line formats), how a checkpoint is loaded,
and whether PLY files can be written. Every line prints each value under its
own label; WS-PSNR is always labelled `wspsnr`.

Metric names (tools/metrics.py):
  psnr    plain PSNR of images clipped to [0, 1] (compute_psnr)
  wspsnr  WS-PSNR, cos(latitude) weights (= sin of the polar angle,
          sin((row + 0.5) * pi / H)), no clipping (WSPSNR.ws_psnr);
          the PSNR the paper reports for panoramas
  ssim    skimage SSIM, Gaussian 11x11 window, on CPU (compute_ssim)
  lpips   LPIPS-VGG, normalize=True (compute_lpips)
  pcc     Pearson correlation of the rendered depth with gts['depth']
  abs, silog, rmse, delta1..3
          depth vs the real GT depth (gts['depth_gt']) under gts['mask_gt'],
          unaligned (evaluate.py --align-depth adds median-aligned columns)
  depthsim
          seam continuity of the rendered depth (left vs right image edge)

View indices are positions in the loader's target list. `novel_views` are
the targets that are not context views (evaluate.py --novel-only).
"""

_MP3D_SCENES = (
    "m3d_0.1",
    "m3d_0.25",
    "m3d_0.5",
    "m3d_0.75",
    "m3d_1.0",
    "residential_0.15",
    "replica_0.5",
)

_FULL_METRICS = (
    "psnr", "wspsnr", "ssim", "lpips", "pcc",
    "abs", "silog", "rmse", "delta1", "delta2", "delta3", "depthsim",
)

_FULL_BATCH_LINE = (
    "[Eval] Batch %d-%d: psnr: %.3f, wspsnr: %.3f, ssim: %.4f, lpips: %.4f, pcc: %.4f, abs: %.4f, "
    "silog: %.4f, rmse: %.4f, delta1: %.4f, delta2: %.4f, delta3: %.4f, depthsim: %.4f"
)
_FULL_SCENE_LINE = (
    " {} psnr: {:.3f}, wspsnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, pcc: {:.4f}, abs: {:.4f}, "
    "silog: {:.4f}, rmse: {:.4f}, delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}, depthsim: {:.4f}"
)
_FULL_TOTAL_LINE = (
    "Finish evluation ({:d} s). Total psnr: {:.3f}, wspsnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, "
    "pcc: {:.4f}, abs: {:.4f}, silog: {:.4f}, rmse: {:.4f}, delta1: {:.4f}, delta2: {:.4f}, "
    "delta3: {:.4f}, depthsim: {:.4f}."
)

_IMAGE_METRICS = ("psnr", "wspsnr", "ssim", "lpips", "pcc")
_IMAGE_BATCH_LINE = "[Eval] Batch %d-%d: psnr: %.3f, wspsnr: %.3f, ssim: %.4f, lpips: %.4f, pcc: %.4f"


def _mp3d_256(loader_module, context_views, novel_views, example_config):
    return dict(
        loader=(loader_module, "load_MP3D_data"),
        stage="test",
        batch_size_key="batch_size_test",
        scene_keys=_MP3D_SCENES,
        context_views=context_views,
        target_views=(0, 1, 2),
        novel_views=novel_views,
        depth_metrics=True,
        pcc_per_view=True,
        batch_metrics=_FULL_METRICS,
        scene_metrics=_FULL_METRICS,
        total_metrics=_FULL_METRICS,
        batch_line=_FULL_BATCH_LINE,
        scene_line=_FULL_SCENE_LINE,
        total_line=_FULL_TOTAL_LINE,
        load="filter",
        summary="table",
        vis_name="Batch_{}_Sampe_{}_Scene_{}",
        save_ply=True,
        example_config=example_config,
    )


EVAL_ENTRIES = {
    # MP3D two-view test sets at 256x512 (stages 1-3), context [0, 2].
    "mp3d_double_256": _mp3d_256(
        "data.mp3d_dataloader_double_256",
        context_views=(0, 2), novel_views=(1,),
        example_config="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py",
    ),
    # Model selection only (the paper reports the test sets): the MP3D *validation* split,
    # png_render_val_1024x512_seq_len_3_m3d_dist_0.5 (1.0 m baseline), with the metric code of
    # mp3d_double_256. The loader pairs its single validation root with the first test set's
    # name/distance, so every validation scene is labelled m3d_0.1 although the baseline is 1.0 m.
    "mp3d_double_256_val": dict(
        _mp3d_256(
            "data.mp3d_dataloader_double_256",
            context_views=(0, 2), novel_views=(1,),
            example_config="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py",
        ),
        stage="val",
        scene_keys=("m3d_0.1",),
    ),
    # One input view (context [1]); the model must be built with num_frames=1.
    "mp3d_single_256": _mp3d_256(
        "data.mp3d_dataloader_single_256",
        context_views=(1,), novel_views=(0, 2),
        example_config="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py",
    ),
    # Stage 4 (512x1024; train.py rows mp3d_double_512_ddp3/4): the rows mp3d_double_256 and
    # mp3d_double_256_val on the 512x1024 loader, nothing else changed: the full metric set
    # (PSNR, WS-PSNR, SSIM, LPIPS, per-view PCC, depth metrics), the three-frame protocol with
    # context [0, 2], batch_size_test, the same scene keys. The model is all_256 built at
    # resolution = [512, 1024] (the stage-4 config). The mp3d_double_512 row below reports a
    # reduced set instead (no plain PSNR, no depth metrics, PCC pooled over views).
    "mp3d_double_512_full": _mp3d_256(
        "data.mp3d_dataloader_double_512",
        context_views=(0, 2), novel_views=(1,),
        example_config="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py",
    ),
    # Stage-4 model selection on the MP3D validation split at 512x1024. As in
    # mp3d_double_256_val, the loader pairs its single validation root with the first test set's
    # name/distance, so every validation scene is labelled m3d_0.1 although the baseline is 1.0 m.
    "mp3d_double_512_full_val": dict(
        _mp3d_256(
            "data.mp3d_dataloader_double_512",
            context_views=(0, 2), novel_views=(1,),
            example_config="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py",
        ),
        stage="val",
        scene_keys=("m3d_0.1",),
    ),
    # MP3D two-view test sets at 512x1024 (the all_512 / pixel_512 models): WS-PSNR, SSIM, LPIPS
    # and PCC pooled over views; no plain PSNR, no depth metrics.
    "mp3d_double_512": dict(
        loader=("data.mp3d_dataloader_double_512", "load_MP3D_data"),
        stage="test",
        batch_size_key="batch_size_test",
        scene_keys=_MP3D_SCENES,
        context_views=(0, 2),
        target_views=(0, 1, 2),
        novel_views=(1,),
        depth_metrics=False,
        pcc_per_view=False,
        batch_metrics=("wspsnr", "ssim", "lpips", "pcc"),
        scene_metrics=("wspsnr", "ssim", "lpips"),
        total_metrics=("wspsnr", "ssim", "lpips", "pcc"),
        batch_line="[Eval] Batch %d-%d: wspsnr: %.3f, ssim: %.4f, lpips: %.4f, pcc: %.4f",
        scene_line=" {} wspsnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}.",
        total_line="Finish evluation ({:d} s). Total wspsnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, pcc: {:.4f}.",
        load="filter",
        summary="n_params",
        vis_name="Batch_{}_Sampe_{}_Scene_{}",
        save_ply=True,
        example_config="configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512.py",
    ),
    # 360Loc 256x512 on the 'val' split (atrium), batch_size_train, no scene keys
    # (the batches carry none), totals only. Targets are four consecutive frames
    # [l, l+1, l+2, l+3] with context [l, l+3], so the novel views are the two
    # interior frames (see Dataset360Loc in data/loc360_dataloader_double_all_512.py).
    "loc360_double_256": dict(
        loader=("data.loc360_dataloader_double_all_512", "load_360Loc_data"),
        stage="val",
        batch_size_key="batch_size_train",
        scene_keys=None,
        context_views=(0, 3),
        target_views=(0, 1, 2, 3),
        novel_views=(1, 2),
        depth_metrics=False,
        pcc_per_view=False,
        batch_metrics=_IMAGE_METRICS,
        scene_metrics=None,
        total_metrics=_IMAGE_METRICS,
        batch_line=_IMAGE_BATCH_LINE,
        scene_line=None,
        total_line=("Finish evluation ({:d} s). Total psnr: {:.3f}, wspsnr: {:.3f}, ssim: {:.4f}, "
                    "lpips: {:.4f}, pcc: {:.4f}."),
        # Loaded with accelerator.load_state(strict=False), which also restores the
        # RNG states saved with the checkpoint.
        load="accelerate_state",
        summary="n_params",
        vis_name="Batch_{}_Sampe_{}",
        # No --save-ply: the 360Loc model's preds["gaussian"] holds the per-view pixel
        # Gaussians ((b v) hw c), not one set per sample.
        save_ply=False,
        example_config="configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py",
    ),
    # Kansas/VIGOR test list, one scene key. The Total line omits wspsnr
    # (it is still written to metrics.json).
    "vigor_double": dict(
        loader=("data.vigor_dataloader_double", "load_VIGOR_data"),
        stage="test",
        batch_size_key="batch_size_test",
        scene_keys=("VIGOR",),
        context_views=(0, 2),
        target_views=(0, 1, 2),
        novel_views=(1,),
        depth_metrics=False,
        pcc_per_view=True,
        batch_metrics=_IMAGE_METRICS,
        scene_metrics=_IMAGE_METRICS,
        total_metrics=("psnr", "ssim", "lpips", "pcc"),
        batch_line=_IMAGE_BATCH_LINE,
        scene_line=" {} psnr: {:.3f}, wspsnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, pcc: {:.4f}",
        total_line="Finish evluation ({:d} s). Total psnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, pcc: {:.4f}.",
        load="filter",
        summary="table",
        vis_name="Batch_{}_Sampe_{}_Scene_{}",
        save_ply=True,
        example_config="configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_all.py",
    ),
}

# 360Loc with the paper's PCC reference: loc360_double_256 (same split, samples, targets, metric
# code and line formats) on a loader whose outputs['depth'] is the Depth Anywhere pseudo-GT
# (depthanywhere/*_depth_anywhere.png) instead of the UniK3D prior the model receives as input. Only PCC can change.
EVAL_ENTRIES["loc360_double_256_da"] = dict(
    EVAL_ENTRIES["loc360_double_256"],
    loader=("data.loc360_dataloader_da", "load_360Loc_data_da"),
)

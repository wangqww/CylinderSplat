
import os, time, argparse, os.path as osp, numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
import math
import data.dataloader as datasets

import mmcv
import mmengine
import imageio
from mmengine import MMLogger
from mmengine.config import Config
import logging

from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration
from tools.metrics import compute_psnr, compute_ssim, compute_lpips, compute_pcc, compute_absrel
from tools.visualization import depths_to_colors

from data.mp3d_dataloader_trible import load_MP3D_data

import warnings
warnings.filterwarnings("ignore")

def inverse_sigmoid(x):
    return torch.log(x/(1-x))
    
def pass_print(*args, **kwargs):
    pass

def create_logger(log_file=None, is_main_process=False, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def main(args):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.output_dir = args.output_dir
    logger_mm = MMLogger.get_instance('mmengine', log_level='WARNING')

    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, 
        logging_dir=None
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='omni-gs', 
            # config=config,
            init_kwargs={
                "wandb":{'name': cfg.exp_name},
            }
        )

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)

    dataset_config = cfg.dataset_params

    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        cfg.dump(osp.join(args.output_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.output_dir, f'{timestamp}.log')
    if not osp.exists(osp.dirname(log_file)):
        os.makedirs(osp.dirname(log_file), exist_ok=True)
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)

    # build model
    from builder import builder as model_builder
    
    my_model = model_builder.build(cfg.model).to(accelerator.device)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')

    # generate datasets
    val_dataloader = load_MP3D_data(dataset_config.batch_size_train, stage='test')

    my_model, val_dataloader = accelerator.prepare(
        my_model, val_dataloader
    )

    # Potentially load in the weights and states from a previous save
    if args.load_from:
        cfg.load_from = args.load_from
    if cfg.load_from:
        path = cfg.load_from
    else:
        path = None

    if path:
        path = os.path.join(args.output_dir, path)
        accelerator.print(f"Loading from checkpoint {path}")
        accelerator.load_state(path, map_location='cpu', strict=False)
        global_iter = int(path.split("-")[1])
        print(f'Successfully loaded from iter{global_iter}')
    else:
        print('Can\'t find checkpoint {}. Randomly initialize model parameters anyway.'.format(args.load_from))
    
    print('work dir: ', args.output_dir)
    
    # Evaluation
    print_freq = cfg.print_freq
    scene_res = {
        "m3d_0.1": [],
        "m3d_0.25": [],
        "m3d_0.5": [],
        "m3d_0.75": [],
        "m3d_1.0": [],
        "residential_0.15": [],
        "replica_0.5": [],
    }
    #time.sleep(10)
    time_s = time.time()
    with torch.no_grad():
        my_model.eval()
        total_psnr, total_ssim, total_lpips, total_pcc = 0.0, 0.0, 0.0, 0.0
        total_absrel, total_rmse, total_absrel_ref, total_rmse_ref = 0.0, 0.0, 0.0, 0.0
        for i_iter, batch in enumerate(val_dataloader):
            data_time_e = time.time()
            # preds, gts = my_model.module.forward_test(batch)
            preds, gts = my_model.forward_test(batch)
            bs = preds["img"].shape[0]
            pred_gaussians = preds["gaussian"]
            pred_imgs = preds["img"]
            pred_depths = preds["depth"]
            gt_imgs = gts["img"]
            gt_depths = gts["depth"]
            gt_depths_m = gts["depth_m"]
            # compute metrics and save results
            # pnsr
            bv_psnr = compute_psnr(
                rearrange(gt_imgs, "b v c h w -> (b v) c h w"),
                rearrange(pred_imgs, "b v c h w -> (b v) c h w")).view(bs, -1)
            bv_psnr_mean = bv_psnr.mean()
            total_psnr += bv_psnr_mean
            # ssim
            bv_ssim = compute_ssim(
                rearrange(gt_imgs, "b v c h w -> (b v) c h w"),
                rearrange(pred_imgs, "b v c h w -> (b v) c h w")).view(bs, -1)
            bv_ssim_mean = bv_ssim.mean()
            total_ssim += bv_ssim_mean
            # lpips
            bv_lpips = compute_lpips(
                rearrange(gt_imgs, "b v c h w -> (b v) c h w"),
                rearrange(pred_imgs, "b v c h w -> (b v) c h w")).view(bs, -1)
            bv_lpips_mean = bv_lpips.mean()
            total_lpips += bv_lpips_mean
            # pcc
            bv_pcc = compute_pcc(
                rearrange(gt_depths, "b v c h w -> (b v c) h w"),
                rearrange(pred_depths, "b v h w -> (b v) h w")
            )
            bv_pcc_mean = bv_pcc.mean()
            total_pcc += bv_pcc_mean
            logger.info('[Eval] Batch %d-%d: psnr: %.3f, ssim: %.4f, lpips: %.4f, pcc: %.4f'%(
                    i_iter, bv_psnr_mean.device.index, bv_psnr_mean, bv_ssim_mean, bv_lpips_mean, bv_pcc_mean))
            output_dir = os.path.join(cfg.output_dir, str(global_iter))
            os.makedirs(output_dir, exist_ok=True)
            if cfg.eval_args.save_ply:
                for b in range(bs):
                    gaussians = pred_gaussians[b]
                    ply_path = osp.join(output_dir, "Batch_{}_Sampe_{}_Scene_{}.ply".format(i_iter, b, batch['scene'][b]))
                    if not osp.exists(osp.dirname(ply_path)):
                        os.makedirs(osp.dirname(ply_path))
                    save_ply(gaussians, ply_path, crop_range=None)
            if cfg.eval_args.save_vis:
                for b in range(bs):
                    # get psnr for this batch sample
                    v_psnr = bv_psnr[b]
                    v_psnr_mean = v_psnr.mean()
                    v_psnr_str = "%.2f" % v_psnr_mean.item()
                    # get ssim for this batch sample
                    v_ssim = bv_ssim[b]
                    v_ssim_mean = v_ssim.mean()
                    # get lpips for this batch sample
                    v_lpips = bv_lpips[b]
                    v_lpips_mean = v_lpips.mean()

                    # save scene metric
                    scene_name = batch['scene'][b]
                    scene_res[scene_name].append({
                        "psnr": v_psnr_mean,
                        "ssim": v_ssim_mean,
                        "lpips": v_lpips_mean
                    })
                    # save visualization results
                    v_pred_imgs = pred_imgs[b]
                    v_pred_depths = pred_depths[b].clamp(0.0, 140.0)
                    v_gt_depths = gt_depths[b].clamp(0.0, 140.0)
                    v_gt_imgs = gt_imgs[b]
                    cat_img_gt = rearrange(v_gt_imgs, "v c h w -> c h (v w)")
                    cat_img_pred = rearrange(v_pred_imgs, "v c h w -> c h (v w)")
                    grid_img = torch.cat([cat_img_gt, cat_img_pred], dim=1)
                    grid_img = (grid_img.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                    grid_depth = depths_to_colors(v_pred_depths)
                    gt_depth = depths_to_colors(v_gt_depths.squeeze(1))
                    grid_all = np.concatenate([grid_img, grid_depth, gt_depth], axis=0)
                    imageio.imwrite(osp.join(output_dir, "Batch_{}_Sampe_{}_Scene_{}.png".format(i_iter, b, batch['scene'][b])), grid_all)
        
        torch.cuda.empty_cache()
        for s in scene_res:
            res = scene_res[s]
            s_psnr = 0
            s_ssim = 0
            s_lpips = 0
            for m in res:
                s_psnr = s_psnr + m['psnr'].item()
                s_ssim = s_ssim + m['ssim'].item()
                s_lpips = s_lpips + m['lpips'].item()
            s_psnr = s_psnr / len(res)
            s_ssim = s_ssim / len(res)
            s_lpips = s_lpips / len(res)
            logger.info(" {} psnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}.".format(
            s,
            s_psnr,
            s_ssim,
            s_lpips))

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_ssim = accelerator.gather_for_metrics(total_ssim).mean()
        total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
        total_pcc = accelerator.gather_for_metrics(total_pcc).mean()
        time_e = time.time()
        logger.info("Finish evluation ({:d} s). Total psnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, pcc: {:.4f}.".format(
            int(time_e - time_s),
            total_psnr.item() / len(val_dataloader),
            total_ssim.item() / len(val_dataloader),
            total_lpips.item() / len(val_dataloader),
            total_pcc.item() / len(val_dataloader)))
        
        # benchmarker = my_model.module.benchmarker
        benchmarker = my_model.benchmarker
        for tag, times in benchmarker.execution_times.items():
            logger.info(
                f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

def save_ply(gaussians, path, crop_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 12.0], compatible=True):
    # gaussians: [B, N, 14]
    # compatible: save pre-activated gaussians as in the original paper
    gaussians = torch.cat([gaussians[:, 0:3],
                           gaussians[:, 6:7],
                           gaussians[:, 11:14],
                           gaussians[:, 7:11],
                           gaussians[:, 3:6]], dim=-1)

    from plyfile import PlyData, PlyElement
    
    means3D = gaussians[:, 0:3].contiguous().float()
    opacity = gaussians[:, 3:4].contiguous().float()
    scales = gaussians[:, 4:7].contiguous().float()
    rotations = gaussians[:, 7:11].contiguous().float()
    shs = gaussians[:, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

    if crop_range is not None:
        x_start, y_start, z_start, x_end, y_end, z_end = crop_range
        mask = (means3D[:, 0] > x_start) & (means3D[:, 0] < x_end) & \
               (means3D[:, 1] > y_start) & (means3D[:, 1] < y_end) & \
               (means3D[:, 2] > z_start) & (means3D[:, 2] < z_end)
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

    # prune by opacity
    mask = opacity.squeeze(-1) >= 0.005
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotations.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    PlyData([el]).write(path)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--load-from', type=str, default=None)

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)
    
    main(args)

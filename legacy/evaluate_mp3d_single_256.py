
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
from tools.metrics import compute_psnr, compute_ssim, compute_lpips, compute_pcc, compute_absrel, WSPSNR
from tools.visualization import depths_to_colors
from safetensors.torch import load_file

from data.mp3d_dataloader_single_256 import load_MP3D_data
import torch.nn as nn
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


def continuity(x, gt):
    # x, gt shape: [B, V, H, W]
    s = x[:, :, :, 0]   # 左边缘
    e = x[:, :, :, -1]  # 右边缘
    s_gt = gt[:, :, :, 0]
    e_gt = gt[:, :, :, -1]

    # 计算边缘差值的差异
    diff = torch.abs((s - e) - (s_gt - e_gt)).mean(dim=-1)
    return diff

def get_model_summary(model: nn.Module) -> str:
    """
    生成一个详细的模型摘要，模仿 PyTorch Lightning 的格式。

    Args:
        model (nn.Module): 需要分析的 PyTorch 模型。

    Returns:
        str: 格式化好的模型摘要字符串。
    """
    
    # --- 1. 统计所有参数和模组状态 ---
    total_params = 0
    trainable_params = 0
    train_mode_modules = 0
    eval_mode_modules = 0

    # 使用 model.modules() 递归遍历所有子模组
    for module in model.modules():
        if module.training:
            train_mode_modules += 1
        else:
            eval_mode_modules += 1
            
    # 使用 model.parameters() 遍历所有参数
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            
    non_trainable_params = total_params - trainable_params
    
    # --- 2. 准备顶层结构表 ---
    # 使用 model.named_children() 只遍历模型的直接子模组
    table_data = []
    top_level_total_params = 0
    
    # 辅助函数，用于计算一个模组内的总参数
    def count_module_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    for i, (name, module) in enumerate(model.named_children()):
        params = count_module_params(module)
        top_level_total_params += params
        mode = "train" if module.training else "eval"
        table_data.append(
            (
                i,
                name,
                module.__class__.__name__,
                f"{params / 1e6:.1f} M" if params > 0 else "0",
                mode
            )
        )
        
    # --- 3. 格式化输出字符串 ---
    
    # 定义列宽
    col_widths = [3, 25, 25, 12, 8] # 序号, 名称, 类型, 参数, 模式

    # 构建分隔线
    separator = "-" * (sum(col_widths) + len(col_widths) - 1)
    
    # 拼接结果
    summary_lines = []
    
    # 表头
    header = f"{' ':<{col_widths[0]}} | {'Name':<{col_widths[1]}} | {'Type':<{col_widths[2]}} | {'Params':>{col_widths[3]}} | {'Mode':<{col_widths[4]}}"
    summary_lines.append(header)
    summary_lines.append(separator)

    # 表格内容
    for row in table_data:
        line = f"{row[0]:<{col_widths[0]}} | {row[1]:<{col_widths[1]}} | {row[2]:<{col_widths[2]}} | {row[3]:>{col_widths[3]}} | {row[4]:<{col_widths[4]}}"
        summary_lines.append(line)
        
    summary_lines.append(separator)

    # 参数统计
    summary_lines.append(f"{trainable_params / 1e6:<.1f} M    Trainable params")
    summary_lines.append(f"{non_trainable_params / 1e6:<.1f} M    Non-trainable params")
    summary_lines.append(f"{total_params / 1e6:<.1f} M    Total params")
    
    # 内存和模组统计
    # 假设为 FP32, 每个参数占 4 bytes
    memory_mb = total_params * 4 / 1e6 
    summary_lines.append(f"{memory_mb:<.3f}   Total estimated model params size (MB)")
    summary_lines.append(f"{train_mode_modules:<8}  Modules in train mode")
    summary_lines.append(f"{eval_mode_modules:<8}  Modules in eval mode")
    
    return "\n".join(summary_lines)

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

    # generate datasets
    val_dataloader = load_MP3D_data(dataset_config.batch_size_test, stage='test')

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
        full_path = os.path.join(args.output_dir, path, 'model.safetensors')
        accelerator.print(f"Resuming from checkpoint {full_path}")
        state_dict = load_file(full_path, device="cpu")
        model_dict = my_model.state_dict()

        global_iter = int(path.split("-")[1])

        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        my_model.load_state_dict(model_dict)
        accelerator.print("Model weights loaded successfully before prepare().")

    else:
        print('Can\'t find checkpoint {}. Randomly initialize model parameters anyway.'.format(args.load_from))
    
    print('work dir: ', args.output_dir)
    
    # n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    # if logger is not None:
    #     logger.info(f'Number of params: {n_parameters}')

    # 生成并打印摘要
    summary_str = get_model_summary(my_model)
    print(summary_str)

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
    wspsnr_calculator = WSPSNR()
    time_s = time.time()
    with torch.no_grad():
        my_model.eval()
        total_psnr, total_wspsnr, total_ssim, total_lpips, total_pcc, total_abs, total_silog, total_rmse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_delta1, total_delta2, total_delta3 = 0.0, 0.0, 0.0
        total_depthsim = 0.0
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
            real_gt_depths = gts['depth_gt'].squeeze(2)
            real_mask_gt = gts['mask_gt'].squeeze(2)
            # compute metrics and save results
            # depthsim
            bv_depthsim = continuity(pred_depths, real_gt_depths).view(bs, -1)
            bv_depthsim_mean = bv_depthsim.mean()
            total_depthsim += bv_depthsim_mean

            # pnsr
            bv_psnr = compute_psnr(
                rearrange(gt_imgs, "b v c h w -> (b v) c h w"),
                rearrange(pred_imgs, "b v c h w -> (b v) c h w")).view(bs, -1)
            bv_psnr_mean = bv_psnr.mean()
            total_psnr += bv_psnr_mean            
            # wspsnr
            bv_wspsnr = wspsnr_calculator.ws_psnr(
                rearrange(gt_imgs, "b v c h w -> (b v) h w c"),
                rearrange(pred_imgs, "b v c h w -> (b v) h w c"),
                max_val=1.0).view(bs, -1)
            bv_wspsnr_mean = bv_wspsnr.mean()
            total_wspsnr += bv_wspsnr_mean
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
                rearrange(pred_depths, "b v h w -> (b v) h w")).view(bs, -1)
            bv_pcc_mean = bv_pcc.mean()
            total_pcc += bv_pcc_mean

            # ================= 1. 尺度对齐 (Scale Alignment) =================
            # 为了防止破坏原始数据，先 clone 一份
            pred_depths_aligned = pred_depths.clone()

            B, V, H, W = real_gt_depths.shape

            # 必须逐张图片计算 scale，因为每张图的尺度差异可能不同
            for b in range(B):
                for v in range(V):
                    mask = real_mask_gt[b, v]  # [H, W]
                    
                    # 如果该图没有有效像素，跳过
                    if mask.sum() < 1:
                        continue
                        
                    # 提取有效像素值
                    valid_gt = real_gt_depths[b, v][mask]
                    valid_pred = pred_depths[b, v][mask]
                    
                    # 计算中值 (Median)
                    gt_median = valid_gt.median()
                    pred_median = valid_pred.median()
                    
                    # 防止除以 0
                    if pred_median > 1e-8:
                        scale = gt_median / pred_median
                        # 将该张图的预测值进行缩放对齐
                        pred_depths_aligned[b, v] = pred_depths[b, v] * scale
            pred_depths_aligned = pred_depths
            # ================= 2. 使用对齐后的数据计算指标 =================
            # abs_rel
            abs_diff = (real_gt_depths - pred_depths_aligned).abs()
            # 只在 mask 区域计算相对误差，非 mask 区域置为 0
            rel_err_map = (abs_diff / (real_gt_depths + 1e-8)) * real_mask_gt.float()

            # 2. 计算有效误差的总和 (Sum) -> 形状变为 [B, V]
            # 假设最后两个维度是 H 和 W (即 dim=-1 和 dim=-2)
            error_sum = rel_err_map.sum(dim=(-2, -1)) 

            # 3. 计算有效像素的个数 (Count) -> 形状变为 [B, V]
            valid_count = real_mask_gt.float().sum(dim=(-2, -1))

            # 4. 计算平均值 (Mean = Sum / Count) -> 形状 [B, V]
            # 加 1e-8 是为了防止某张图完全没有有效像素导致除以0
            bv_abs = error_sum / (valid_count + 1e-8)
            bv_abs_mean = bv_abs.mean()
            total_abs += bv_abs_mean


            # SILog
            # 1. 计算对数差 (只在 mask 区域)
            # clamp 防止 log(0)
            log_diff = (torch.log(pred_depths.clamp(min=1e-8)) - torch.log(real_gt_depths.clamp(min=1e-8))) * real_mask_gt.float()

            # 2. 有效像素数
            num_valid = real_mask_gt.float().sum(dim=(-2, -1)) + 1e-8

            # 3. 计算第一项: sum(d^2) / n
            term1 = (log_diff ** 2).sum(dim=(-2, -1)) / num_valid

            # 4. 计算第二项: (sum(d))^2 / n^2
            term2 = (log_diff.sum(dim=(-2, -1)) ** 2) / (num_valid ** 2)

            # 5. SILog = sqrt(term1 - term2) * 100 (通常乘以100)
            bv_silog = torch.sqrt((term1 - term2).abs()) * 100
            bv_silog_mean = bv_silog.mean()
            total_silog += bv_silog_mean
            # rmse
            # 1. 计算逐像素的平方误差 (Squared Error Map)
            sq_diff = (real_gt_depths - pred_depths_aligned) ** 2

            # 2. 应用 Mask：只保留有效区域的平方误差，无效区域置 0
            masked_sq_diff = sq_diff * real_mask_gt.float()

            # 3. 在空间维度求和 (Sum) -> 形状变为 [B, V]
            sum_sq_error = masked_sq_diff.sum(dim=(-2, -1))

            # 4. 计算有效像素个数 (Count) -> 形状变为 [B, V]
            valid_count = real_mask_gt.float().sum(dim=(-2, -1))

            # 5. 计算均方误差 (MSE = Sum / Count) -> 形状 [B, V]
            # 加 1e-8 防止除零
            mse = sum_sq_error / (valid_count + 1e-8)

            # 6. 开根号得到 RMSE (Sqrt) -> 形状 [B, V]
            bv_rmse = torch.sqrt(mse)
            bv_rmse_mean = bv_rmse.mean()
            total_rmse += bv_rmse_mean

            # ================= 3. 计算准确率指标 (Accuracy Metrics) =================
            # 1. 计算逐像素的比率阈值图 (Pixel-wise Threshold Map)
            # 公式: thresh = max(gt / pred, pred / gt)
            # 使用对齐后的预测值 pred_depths_aligned
            # 分母加 1e-8 防止除零
            r1 = real_gt_depths / (pred_depths_aligned + 1e-8)
            r2 = pred_depths_aligned / (real_gt_depths + 1e-8)
            thresh = torch.max(r1, r2)

            # 2. 计算有效像素的总个数 (分母) -> 形状 [B, V]
            valid_count = real_mask_gt.float().sum(dim=(-2, -1))
            
            # 3. 确保只统计 Mask 区域内的点
            # 注意：必须与 real_mask_gt 做逻辑与运算，排除无效区域可能的异常值
            valid_mask_bool = real_mask_gt.bool()

            # ----- δ1 < 1.25 -----
            # 统计同时满足 (thresh < 1.25) 和 (是有效像素) 的点
            delta1_correct = (thresh < 1.25) & valid_mask_bool
            # 空间维度求和 -> [B, V]
            delta1_count = delta1_correct.float().sum(dim=(-2, -1))
            # 计算比例
            bv_delta1 = delta1_count / (valid_count + 1e-8)
            bv_delta1_mean = bv_delta1.mean()
            total_delta1 += bv_delta1_mean

            # ----- δ2 < 1.25^2 -----
            delta2_correct = (thresh < 1.25 ** 2) & valid_mask_bool
            delta2_count = delta2_correct.float().sum(dim=(-2, -1))
            bv_delta2 = delta2_count / (valid_count + 1e-8)
            bv_delta2_mean = bv_delta2.mean()
            total_delta2 += bv_delta2_mean

            # ----- δ3 < 1.25^3 -----
            delta3_correct = (thresh < 1.25 ** 3) & valid_mask_bool
            delta3_count = delta3_correct.float().sum(dim=(-2, -1))
            bv_delta3 = delta3_count / (valid_count + 1e-8)
            bv_delta3_mean = bv_delta3.mean()
            total_delta3 += bv_delta3_mean

            # bv_pcc_m = compute_pcc(
            #     rearrange(gt_depths, "b v c h w -> (b v c) h w"),
            #     rearrange(gt_depths_m, "b v c h w -> (b v c) h w")).view(bs, -1)
            # bv_pcc_m_mean = bv_pcc_m.mean()
            # total_pcc_m += bv_pcc_m_mean
            logger.info('[Eval] Batch %d-%d: psnr: %.3f, wspsnr: %.3f, ssim: %.4f, lpips: %.4f, pcc: %.4f, abs: %.4f, silog: %.4f, rmse: %.4f, delta1: %.4f, delta2: %.4f, delta3: %.4f, depthsim: %.4f'%(
                    i_iter, bv_psnr_mean.device.index, bv_psnr_mean, bv_wspsnr_mean, bv_ssim_mean, bv_lpips_mean, bv_pcc_mean, bv_abs_mean, bv_silog_mean, bv_rmse_mean, bv_delta1_mean, bv_delta2_mean, bv_delta3_mean, bv_depthsim_mean))
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
                    # get wspsnr for this batch sample
                    v_wspsnr = bv_wspsnr[b]
                    v_wspsnr_mean = v_wspsnr.mean()
                    v_psnr_str = "%.2f" % v_psnr_mean.item()
                    # get ssim for this batch sample
                    v_ssim = bv_ssim[b]
                    v_ssim_mean = v_ssim.mean()
                    # get lpips for this batch sample
                    v_lpips = bv_lpips[b]
                    v_lpips_mean = v_lpips.mean()

                    v_pcc = bv_pcc[b]
                    v_pcc_mean = v_pcc.mean()

                    v_abs = bv_abs[b]
                    v_abs_mean = v_abs.mean()

                    v_silog = bv_silog[b]
                    v_silog_mean = v_silog.mean()

                    v_rmse = bv_rmse[b]
                    v_rmse_mean = v_rmse.mean()

                    v_delta1 = bv_delta1[b]
                    v_delta1_mean = v_delta1.mean()

                    v_delta2 = bv_delta2[b]
                    v_delta2_mean = v_delta2.mean()

                    v_delta3 = bv_delta3[b]
                    v_delta3_mean = v_delta3.mean()
                    
                    v_depthsim = bv_depthsim[b]
                    v_depthsim_mean = v_depthsim.mean()
                    # save scene metric
                    scene_name = batch['scene'][b]
                    scene_res[scene_name].append({
                        "psnr": v_psnr_mean,
                        'wspsnr': v_wspsnr_mean,
                        "ssim": v_ssim_mean,
                        "lpips": v_lpips_mean,
                        'pcc': v_pcc_mean,
                        'abs': v_abs_mean,
                        'silog': v_silog_mean,
                        'rmse': v_rmse_mean,
                        'delta1': v_delta1_mean,
                        'delta2': v_delta2_mean,
                        'delta3': v_delta3_mean,
                        'depthsim': v_depthsim_mean,
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
            s_wspsnr = 0
            s_ssim = 0
            s_lpips = 0
            s_pcc = 0
            s_abs = 0
            s_silog = 0
            s_rmse = 0
            s_delta1 = 0
            s_delta2 = 0
            s_delta3 = 0
            s_depthsim = 0
            for m in res:
                s_psnr = s_psnr + m['psnr'].item()
                s_wspsnr = s_wspsnr + m['wspsnr'].item()
                s_ssim = s_ssim + m['ssim'].item()
                s_lpips = s_lpips + m['lpips'].item()
                s_pcc = s_pcc + m['pcc'].item()
                s_abs = s_abs + m['abs'].item()
                s_silog = s_silog + m['silog'].item()
                s_rmse = s_rmse + m['rmse'].item()
                s_delta1 = s_delta1 + m['delta1'].item()
                s_delta2 = s_delta2 + m['delta2'].item()
                s_delta3 = s_delta3 + m['delta3'].item()
                s_depthsim = s_depthsim + m['depthsim'].item()
            s_psnr = s_psnr / len(res)
            s_wspsnr = s_wspsnr / len(res)
            s_ssim = s_ssim / len(res)
            s_lpips = s_lpips / len(res)
            s_pcc = s_pcc / len(res)
            s_abs = s_abs / len(res)
            s_silog = s_silog / len(res)
            s_rmse = s_rmse / len(res)
            s_delta1 = s_delta1 / len(res)
            s_delta2 = s_delta2 / len(res)
            s_delta3 = s_delta3 / len(res)
            s_depthsim = s_depthsim / len(res)
            logger.info(" {} psnr: {:.3f}, wspsnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, pcc: {:.4f}, abs: {:.4f}, silog: {:.4f}, rmse: {:.4f}, delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}, depthsim: {:.4f}".format(
            s,
            s_psnr,
            s_wspsnr,
            s_ssim,
            s_lpips,
            s_pcc,
            s_abs,
            s_silog,
            s_rmse,
            s_delta1,
            s_delta2,
            s_delta3,
            s_depthsim
            ))

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_wspsnr = accelerator.gather_for_metrics(total_wspsnr).mean()
        total_ssim = accelerator.gather_for_metrics(total_ssim).mean()
        total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
        total_pcc = accelerator.gather_for_metrics(total_pcc).mean()
        total_abs = accelerator.gather_for_metrics(total_abs).mean()
        total_silog = accelerator.gather_for_metrics(total_silog).mean()
        total_rmse = accelerator.gather_for_metrics(total_rmse).mean()
        total_delta1 = accelerator.gather_for_metrics(total_delta1).mean()
        total_delta2 = accelerator.gather_for_metrics(total_delta2).mean()
        total_delta3 = accelerator.gather_for_metrics(total_delta3).mean()
        total_depthsim = accelerator.gather_for_metrics(total_depthsim).mean()

        time_e = time.time()
        logger.info("Finish evluation ({:d} s). Total psnr: {:.3f}, ssim: {:.4f}, lpips: {:.4f}, pcc: {:.4f}, abs: {:.4f}, silog: {:.4f}, rmse: {:.4f}, delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}, depthsim: {:.4f}.".format(
            int(time_e - time_s),
            total_psnr.item() / len(val_dataloader),
            total_wspsnr.item() / len(val_dataloader),
            total_ssim.item() / len(val_dataloader),
            total_lpips.item() / len(val_dataloader),
            total_pcc.item() / len(val_dataloader),
            total_abs.item() / len(val_dataloader),
            total_silog.item() / len(val_dataloader),
            total_rmse.item() / len(val_dataloader),
            total_delta1.item() / len(val_dataloader),
            total_delta2.item() / len(val_dataloader),
            total_delta3.item() / len(val_dataloader),
            total_depthsim.item() / len(val_dataloader)
        ))

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

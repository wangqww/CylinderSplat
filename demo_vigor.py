
import os, time, argparse, os.path as osp, numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, pack
import math
import data.dataloader as datasets

import mmcv
import mmengine
import imageio
from mmengine import MMLogger
from mmengine.config import Config
import logging

import moviepy.editor as mpy
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration
from tools.visualization import depths_to_colors

from data.vigor_dataloader import load_vigor_data
import warnings
warnings.filterwarnings("ignore")

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
        os.makedirs(osp.dirname(log_file))
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)

    # build model
    from builder import builder as model_builder
    
    my_model = model_builder.build(cfg.model).to(accelerator.device)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')

    # generate datasets
    # dataset = getattr(datasets, dataset_config.dataset_name)
    # demo_dataset = dataset(dataset_config.resolution, split="demo",
    #                       use_center=dataset_config.use_center,
    #                       use_first=dataset_config.use_first,
    #                       use_last=dataset_config.use_last)
    # demo_dataloader = DataLoader(
    #     demo_dataset, dataset_config.batch_size_test, shuffle=False,
    #     num_workers=dataset_config.num_workers_test
    # )

    train_dataloader, val_dataloader = load_vigor_data(dataset_config.batch_size_test)

    my_model, demo_dataloader = accelerator.prepare(
        my_model, train_dataloader
    )

    # Potentially load in the weights and states from a previous save
    if args.load_from:
        cfg.load_from = args.load_from
    # if cfg.load_from:
    #     path = cfg.load_from
    else:
        path = None

    if path:
        accelerator.print(f"Loading from checkpoint {path}")
        accelerator.load_state(path, map_location='cpu', strict=False)
        global_iter = int(path.split("-")[1])
        print(f'Successfully loaded from iter{global_iter}')
    else:
        print('Can\'t find checkpoint {}. Randomly initialize model parameters anyway.'.format(args.load_from))
    
    print('work dir: ', args.output_dir)
    
    # Evaluation
    print_freq = cfg.print_freq

    #time.sleep(10)
    time_s = time.time()
    with torch.no_grad():
        my_model.eval()
        for i_iter, batch in enumerate(demo_dataloader):
            data_time_e = time.time()
            if torch.cuda.device_count() > 1:
                preds, bin_tokens = my_model.module.forward_demo(batch)
            else:
                preds, bin_tokens = my_model.forward_demo(batch)
            bs = preds["img"].shape[0]
            pred_imgs = preds["img"]
            pred_depths = preds["depth"]
            logger.info('[Eval] Batch %d-%d'%(
                    i_iter, pred_depths.device.index))

            for b in range(bs):
                bin_token = bin_tokens[b]
                # dump rgb view
                dump_path = osp.join(cfg.output_dir, "{}_rgb.mp4".format(bin_token))
                video = (pred_imgs[b].clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
                video_rec = wandb.Video(video[None], fps=30, format="mp4")
                video_tensor = video_rec._prepare_video(video_rec.data)
                clip = mpy.ImageSequenceClip(list(video_tensor), fps=30)
                clip.write_videofile(dump_path, codec='libx264', preset='medium', logger=None)
                # dump depth view
                dump_path_dpt = osp.join(cfg.output_dir, "{}_depth.mp4".format(bin_token))
                pred_depth = pred_depths[b].clamp(0.0, 100.0)
                max_val = float(pred_depth.max())
                video_dpt = depths_to_colors(pred_depths[b], concat="frame", max_val=max_val)
                video_dpt = video_dpt.transpose((0, 3, 1, 2))
                video_rec_dpt = wandb.Video(video_dpt[None], fps=30, format="mp4")
                video_tensor_dpt = video_rec_dpt._prepare_video(video_rec_dpt.data)
                clip_dpt = mpy.ImageSequenceClip(list(video_tensor_dpt), fps=30)
                clip_dpt.write_videofile(dump_path_dpt, codec='libx264', preset='medium', logger=None)
        
        torch.cuda.empty_cache()

        time_e = time.time()
        logger.info("Finish demo ({:d} s).".format(
            int(time_e - time_s)))

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


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

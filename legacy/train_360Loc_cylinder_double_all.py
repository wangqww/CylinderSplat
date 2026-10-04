import os, time, argparse, os.path as osp, numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
import math
import data.dataloader as datasets
from safetensors.torch import load_file

import mmcv
import mmengine
from mmengine import MMLogger
from mmengine.config import Config
import logging

from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import set_seed, convert_outputs_to_fp32, DistributedType, ProjectConfiguration, InitProcessGroupKwargs

from data.loc360_dataloader_double_all import load_360Loc_data
# from data.mp3d_dataloader_double import load_MP3D_data
# from data.vigor_dataloader_cube import load_vigor_data

import warnings
warnings.filterwarnings("ignore")

def create_logger(log_file=None, is_main_process=False, log_level=logging.INFO):
    if not is_main_process:
        return None
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if is_main_process else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if is_main_process else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if is_main_process else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def main(args):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    logger_mm = MMLogger.get_instance('mmengine', log_level='WARNING')
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir, 
        logging_dir=os.path.join(cfg.work_dir, 'logs')
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
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

    max_num_epochs = cfg.max_epochs

    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    if not osp.exists(osp.dirname(log_file)):
        os.makedirs(osp.dirname(log_file))
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)
    if logger is not None:
        logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from builder import builder as model_builder
    
    my_model = model_builder.build(cfg.model).to(accelerator.device)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')

    optimizers = my_model.configure_optimizers(cfg.lr)
    optimizer = optimizers[0]
    
    #scheduler = get_scheduler(
    #    cfg.lr_scheduler_type,
    #    optimizer=optimizer,
    #    num_warmup_steps=cfg.warmup_steps*accelerator.num_processes,
    #    num_training_steps=cfg.max_train_steps*accelerator.num_processes,
    #)
    # consine lr scheduler
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        1 / (cfg.warmup_steps*accelerator.num_processes),
        1,
        total_iters=cfg.warmup_steps*accelerator.num_processes,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_train_steps*accelerator.num_processes, eta_min=cfg.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, scheduler], milestones=[cfg.warmup_steps*accelerator.num_processes])

    # generate datasets
    # dataset = getattr(datasets, dataset_config.dataset_name)
    # train_dataset = dataset(dataset_config.resolution, split="train",
    #                         use_center=dataset_config.use_center,
    #                         use_first=dataset_config.use_first,
    #                         use_last=dataset_config.use_last)
    # val_dataset = dataset(dataset_config.resolution, split="val",
    #                       use_center=dataset_config.use_center,
    #                       use_first=dataset_config.use_first,
    #                       use_last=dataset_config.use_last)
    # train_dataloader = DataLoader(
    #     train_dataset, dataset_config.batch_size_train, shuffle=True,
    #     num_workers=dataset_config.num_workers
    # )
    # val_dataloader = DataLoader(
    #     val_dataset, dataset_config.batch_size_val, shuffle=False,
    #     num_workers=dataset_config.num_workers_val
    # )

    train_dataloader = load_360Loc_data(dataset_config.batch_size_train, stage='train')
    val_dataloader = load_360Loc_data(dataset_config.batch_size_train, stage='val')
    my_model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        my_model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    
    # resume and load
    epoch = 0
    global_iter = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if cfg.resume_from:
        if cfg.resume_from != "latest":
            path = os.path.basename(cfg.resume_from)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            else:
                path = None

    path = cfg.resume_from
    if path:
        accelerator.print(f"Resuming from checkpoint {path}")
        state_dict = load_file(path, device="cpu")
        my_model.load_state_dict(state_dict)
        accelerator.print("Model weights loaded successfully before prepare().")
    
    print('work dir: ', args.work_dir)
    
    # training
    print_freq = cfg.print_freq

    while epoch < max_num_epochs:
        my_model.train()
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, batch in enumerate(train_dataloader):
            # forward + backward + optimize
            data_time_e = time.time()
            with accelerator.accumulate(my_model):
                optimizer.zero_grad()
                loss, log, _, _, _, _, _, _, _ = my_model.forward(batch, "train", iter=global_iter, iter_end=cfg.max_train_steps)
                # loss, log, _, _, _, _, _, _, _ = my_model.module.forward(batch, "train", iter=global_iter, iter_end=cfg.max_train_steps)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                optimizer.step()
                scheduler.step()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            accelerator.wait_for_everyone()
            if accelerator.sync_gradients and accelerator.is_main_process:
                if global_iter > 0 and global_iter % cfg.save_freq == 0:
                    if accelerator.is_main_process:
                        save_file_name = os.path.join(os.path.abspath(args.work_dir), f'checkpoint-{global_iter}')
                        accelerator.save_state(save_file_name)
                        dst_file = osp.join(args.work_dir, 'latest')
                        mmengine.utils.symlink(save_file_name, dst_file)
                        if logger is not None:
                            logger.info('[TRAIN] Save latest state dict to {}.'.format(save_file_name))
                
                if global_iter > 0 and global_iter % cfg.val_freq == 0:
                    my_model.eval()
                    if accelerator.is_main_process:
                        for i_iter_val, batch_val in enumerate(val_dataloader):
                            val_batch_save_dir = osp.join(cfg.output_dir, cfg.exp_name, "validation",
                                                "step-{}/batch-{}".format(global_iter, i_iter_val))
                            log_val = my_model.validation_step(batch_val, val_batch_save_dir)
                            # log_val = my_model.module.validation_step(batch_val, val_batch_save_dir)
                            log.update(log_val)
                    my_model.train()
            
            time_e = time.time()

            # print loss log regularly
            if global_iter % print_freq == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]['lr']
                losses_str = ""
                for loss_k, loss_v in log.items():
                    losses_str += ("%s: %.3f, " % (loss_k, loss_v))
                if logger is not None:
                    logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f, %s grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                        epoch, i_iter, len(train_dataloader), 
                        loss.item(), losses_str, grad_norm, lr,
                        time_e - time_s, data_time_e - data_time_s
                    ))

            global_iter += 1

            # dump loss log to tensorboard
            accelerator.log(log, step=global_iter)

            data_time_s = time.time()
            time_s = time.time()

        epoch += 1

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()

    SEED = 42 # 你可以选择任何固定的整数
    torch.manual_seed(SEED)

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)
    
    main(args)

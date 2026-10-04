"""CylinderSplat training entry point; replaces the legacy train_*.py scripts (now in legacy/).

    accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py \
        --entry mp3d_double_256 \
        --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
        --run-id all256_repro --resume-from <volume_256>/checkpoint-36000 --transfer stage2_to_stage3

    accelerate launch --config-file configs/accelerate/accel_1proc.yaml train.py \
        --entry mp3d_double_160 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py --run-id all160

    accelerate launch --config-file configs/accelerate/accel_1proc.yaml train.py \
        --entry screen_mp3d_all_256 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
        --run-id screen_c0_s43 --resume-from <all_256>/checkpoint-48000 --seed 43

    accelerate launch --config-file configs/accelerate/accel_1proc.yaml train.py \
        --entry screen_mp3d_single_256 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256_single.py \
        --run-id screen_d3_on --transfer d3_single_view --switch v1_identity_pose=true

    accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py \
        --entry mp3d_double_512_ddp3 --py-config <all_256 with resolution = [512, 1024]> \
        --run-id all512x1024 --resume-from <stage-3 run>/checkpoint-N --transfer stage3_to_stage4_512 \
        --switch ddp_forward=true

Each table-T --entry is a row of configs/entries.py (ENTRIES) and reproduces one legacy
trainer exactly: loader and DataLoader keywords, scheduler, forward call, validation,
process count and setup order. The Phase-2 screen rows (SCREEN_ENTRIES, plan §7) reuse a
table-T row's loader with their own process count, forward call and OneCycle length;
only they accept --screen-steps and --seed. The stage-4 rows (STAGE4_ENTRIES) train on the
512x1024 loader of mp3d_double_512 with the recipe of mp3d_double_256, on 3 or 4
processes. Added on top: the write guard (every output
under the run dir, the process cwd in <run dir>/cwd), validation output in
<run dir>/validation under torch.no_grad(), the weights-only resume rule
(tools/resume.py, --transfer), the opt-in switches (tools/switches.py, --switch),
--profile-steps and --max-steps.
"""

import os, time, argparse, json, statistics, importlib, importlib.util, os.path as osp
import torch
from torch.utils.data import DataLoader, RandomSampler

import mmengine
from mmengine import MMLogger
from mmengine.config import Config
import logging

from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, InitProcessGroupKwargs, DistributedDataParallelKwargs

from tools import switches as switch_registry
from tools.write_guard import (check_write_roots, default_run_dir, prepare_run_dir, enter_scratch_cwd,
                               guard_save_path)
from tools.resume import load_weights_only, refuse_source_inside

import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = osp.dirname(osp.abspath(__file__))
ENTRIES_FILE = osp.join(REPO_ROOT, 'configs', 'entries.py')


def load_entries(path=ENTRIES_FILE):
    # configs/ is not a package; load the table by path.
    spec = importlib.util.spec_from_file_location('cylindersplat_entries', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entries = load_entries()
SCREEN_ENTRIES = entries.SCREEN_ENTRIES
STAGE4_ENTRIES = entries.STAGE4_ENTRIES
ENTRIES = dict(entries.ENTRIES, **SCREEN_ENTRIES, **STAGE4_ENTRIES)  # table T, the Phase-2 screen rows, stage 4
TRANSFERS = entries.TRANSFERS


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


def write_roots(work_dir, py_config):
    """Every path train.py writes to or under, by name; all are checked before anything is written.

    checkpoint-N dirs and the timestamped log file sit directly in work_dir and are
    re-checked when they are written.
    """
    return {
        'work_dir': work_dir,  # Accelerate project_dir, checkpoint-N, <timestamp>.log
        'logging_dir': osp.join(work_dir, 'logs'),  # Accelerate trackers
        'config_dump': osp.join(work_dir, osp.basename(py_config)),
        'switches': osp.join(work_dir, 'switches.json'),
        'latest': osp.join(work_dir, 'latest'),
        'validation': osp.join(work_dir, 'validation'),
        'scratch_cwd': osp.join(work_dir, 'cwd'),
    }


def build_dataloader(entry, loader_module, stage, batch_size, shuffle_train=False, dataset_extra=None):
    """One stage's DataLoader with the keywords of the row's legacy load_*() factory.

    Default: the same dataset, sequential order, generator seed, worker_init_fn,
    persistent_workers and num_workers (table T, C3) as the factory. shuffle_train
    (D1b, map-style rows only) adds a RandomSampler whose generator is seeded like
    the loader's train generator; Accelerate shards its batches across processes and
    keeps that generator in sync, so every epoch is a new permutation. dataset_extra
    adds dataset keywords (loc360_interleave: interleave=True on the 360Loc train split).
    """
    spec = entry['loader']
    stage_spec = spec['stages'].get(stage, spec['stages']['other'])
    dataset = getattr(loader_module, spec['dataset_class'])(**spec['dataset_kwargs'][stage], **(dataset_extra or {}))
    sampler = None
    if shuffle_train and stage == 'train':
        sampler = RandomSampler(dataset, generator=loader_module.get_generator(stage_spec['seed']))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=spec['num_workers'],
        generator=loader_module.get_generator(stage_spec['seed']),
        worker_init_fn=loader_module.worker_init_fn,
        persistent_workers=stage_spec['persistent_workers'],
        shuffle=False,
        sampler=sampler,
    )


def build_loaders(entry, loader_module, dataset_config, shuffle_train, interleave=False):
    train_dataloader = build_dataloader(entry, loader_module, 'train',
                                       dataset_config[entry['batch_size']['train']], shuffle_train,
                                       dict(interleave=True) if interleave else None)
    val_dataloader = build_dataloader(entry, loader_module, 'val', dataset_config[entry['batch_size']['val']])
    return train_dataloader, val_dataloader


def build_onecycle_scheduler(optimizer, cfg, train_dataloader, max_num_epochs):
    # PanSplat Scheduler (train_mp3d_cylinder_double_256.py:131-142)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=len(train_dataloader) * max_num_epochs + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    return scheduler


def build_warmup_cosine_scheduler(optimizer, cfg, accelerator):
    # consine lr scheduler (train_mp3d_cylinder_double.py:117-124)
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        1 / (cfg.warmup_steps*accelerator.num_processes),
        1,
        total_iters=cfg.warmup_steps*accelerator.num_processes,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_train_steps*accelerator.num_processes, eta_min=cfg.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, scheduler], milestones=[cfg.warmup_steps*accelerator.num_processes])
    return scheduler


def build_onecycle_screen_scheduler(optimizer, cfg):
    # Phase-2 screen (plan §7): the PanSplat OneCycleLR above, total_steps = screen steps + 100
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=cfg.screen_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    return scheduler


def build_scheduler(entry, optimizer, cfg, accelerator, train_dataloader, max_num_epochs):
    if entry['scheduler'] == 'onecycle':
        return build_onecycle_scheduler(optimizer, cfg, train_dataloader, max_num_epochs)
    if entry['scheduler'] == 'warmup_cosine':
        return build_warmup_cosine_scheduler(optimizer, cfg, accelerator)
    if entry['scheduler'] == 'onecycle_screen':
        return build_onecycle_screen_scheduler(optimizer, cfg)
    raise ValueError(f"unknown scheduler kind {entry['scheduler']!r}")


def seed_processes(cfg, accelerator):
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)


def init_trackers(cfg, accelerator):
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='omni-gs',
            init_kwargs={
                "wandb": {'name': cfg.exp_name},
            }
        )


def build_model(cfg, accelerator):
    from builder import builder as model_builder

    return model_builder.build(cfg.model).to(accelerator.device)


def apply_render_switches(switch_values):
    """D8: renderer opacity pruning threshold (0 = off, the released behaviour)."""
    from model import gaussian

    gaussian.set_prune_opacity(switch_values['prune_opacity'])


def describe_entry(entry):
    if entry.get('screen'):
        return f"runs the Phase-2 screen recipe on the {entry['screen']['base']} loader"
    if entry.get('stage4'):
        s4 = entry['stage4']
        return f"runs stage 4 (the {s4['recipe_row']} recipe on the {s4['loader_row']} loader)"
    return f"reproduces {entry['legacy_script']}"


def save_checkpoint(accelerator, work_dir, latest, global_iter, logger):
    save_file_name = guard_save_path(os.path.join(os.path.abspath(work_dir), f'checkpoint-{global_iter}'))
    dst_file = guard_save_path(latest, allow_symlink=True)  # mmengine.utils.symlink replaces the link
    accelerator.save_state(save_file_name)
    mmengine.utils.symlink(save_file_name, dst_file)
    if logger is not None:
        logger.info('[TRAIN] Save latest state dict to {}.'.format(save_file_name))


def check_row_switches(entry_name, entry, switch_values):
    if switch_values['shuffle_train'] and entry['loader']['iterable']:
        raise SystemExit(
            f"--entry {entry_name}: shuffle_train needs a map-style dataset; "
            f"{entry['loader']['dataset_class']} is an IterableDataset that already shuffles its train split")
    if switch_values.get('loc360_interleave') and entry['loader']['dataset_class'] != 'Dataset360Loc':
        raise SystemExit(
            f"--entry {entry_name}: loc360_interleave needs the 360Loc loader (Dataset360Loc), "
            f"not {entry['loader']['dataset_class']}")


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def report_profile(step_ms, accelerator, work_dir):
    """C4: wall time of the timed train steps (forward, backward, clip, optimizer and scheduler step)."""
    summary = dict(steps=len(step_ms), mean_ms=statistics.fmean(step_ms), median_ms=statistics.median(step_ms))
    if len(step_ms) > 1:
        summary.update(mean_ms_after_first=statistics.fmean(step_ms[1:]),
                       median_ms_after_first=statistics.median(step_ms[1:]))
    rank = accelerator.process_index
    print(f"[profile] rank {rank}/{accelerator.num_processes}: "
          + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    path = guard_save_path(osp.join(work_dir, f'profile_steps_rank{rank}.json'))
    with open(path, 'w') as f:
        json.dump(dict(summary, step_ms=step_ms), f, indent=2)


def main(args, entry, loader_module):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    screen = entry.get('screen')
    if screen:
        # Phase-2 screen (plan §7); parse_args has filled in the row's defaults. The seed
        # reaches set_seed(cfg.seed + local_process_index) below; __main__ used it for
        # torch.manual_seed. cfg.dataset_params.seed (a copy taken when the config was
        # parsed) is read by neither train.py nor the loaders and is left as it is.
        cfg.screen_steps = args.screen_steps
        cfg.seed = args.seed
    if args.resume_from:
        cfg.resume_from = args.resume_from
    elif args.no_resume:
        cfg.resume_from = False
    if cfg.resume_from:
        cfg.resume_from = osp.abspath(cfg.resume_from)  # before the move into <work dir>/cwd
    # Switches go into cfg.model before the model is built; all at default = the legacy run.
    switch_values = switch_registry.apply(cfg, args.switch)
    check_row_switches(args.entry, entry, switch_values)
    transfer = TRANSFERS[args.transfer]

    # INV-3: nothing is written before every write root has passed the guard.
    roots = write_roots(args.work_dir, args.py_config)
    check_write_roots(roots.values())
    if cfg.resume_from:
        refuse_source_inside(cfg.resume_from, args.work_dir)

    logger_mm = MMLogger.get_instance('mmengine', log_level='WARNING')
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    kwargs_handlers = [kwargs]
    if switch_values['ddp_forward']:
        # D1: forward through the DDP wrapper; parameters that do not reach the loss must not stall the all-reduce.
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir,
        logging_dir=os.path.join(cfg.work_dir, 'logs')
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers
    )
    if accelerator.num_processes != entry['num_processes']:
        raise SystemExit(
            f"--entry {args.entry} {describe_entry(entry)} with {entry['num_processes']} process(es); "
            f"this launch has {accelerator.num_processes}. Use configs/accelerate/accel_{entry['num_processes']}proc.yaml.")

    # Relative writes by the models (debug PNGs) land in <work dir>/cwd.
    enter_scratch_cwd(prepare_run_dir(args.work_dir, extra_write_roots=list(roots.values())))
    if accelerator.is_main_process:
        with open(guard_save_path(roots['switches']), 'w') as f:
            json.dump(switch_values, f, indent=2, sort_keys=True)

    dataset_config = cfg.dataset_params
    max_num_epochs = cfg.max_epochs
    shuffle_train = switch_values['shuffle_train']
    interleave = switch_values['loc360_interleave']
    loaders_first = entry['setup_order'] == 'loaders_before_model'
    if loaders_first:
        # *_256 trainers: seed, loaders, trackers
        seed_processes(cfg, accelerator)
        train_dataloader, val_dataloader = build_loaders(entry, loader_module, dataset_config, shuffle_train, interleave)
        init_trackers(cfg, accelerator)
    else:
        # 160 / 512 trainers: trackers, seed; the loaders are built after the scheduler
        init_trackers(cfg, accelerator)
        seed_processes(cfg, accelerator)
        train_dataloader = val_dataloader = None

    # configure logger
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(guard_save_path(roots['config_dump']))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = guard_save_path(osp.join(args.work_dir, f'{timestamp}.log'))
    if not osp.exists(osp.dirname(log_file)):
        os.makedirs(osp.dirname(log_file))
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)
    if logger is not None:
        logger.info(f'Config:\n{cfg.pretty_text}')
        logger.info(f'Entry: {args.entry} ({describe_entry(entry)}); switches: {switch_values}')
        if screen:
            logger.info(f'Screen: {cfg.screen_steps} steps, seed {cfg.seed}')

    # build model
    my_model = build_model(cfg, accelerator)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f'Number of params: {n_parameters}')
    apply_render_switches(switch_values)

    optimizers = my_model.configure_optimizers(cfg.lr)
    optimizer = optimizers[0]
    scheduler = build_scheduler(entry, optimizer, cfg, accelerator, train_dataloader, max_num_epochs)

    if not loaders_first:
        train_dataloader, val_dataloader = build_loaders(entry, loader_module, dataset_config, shuffle_train, interleave)

    path = cfg.resume_from
    if path:
        accelerator.print(f"Resuming from checkpoint {path} (weights only, transfer '{args.transfer}')")
        report = load_weights_only(my_model, path, transfer['allowed_missing'], transfer['allowed_extra'])
        accelerator.print(f"Model weights loaded successfully before prepare(): {len(report['matched'])} tensors, "
                          f"{len(report['extra'])} allowed extra, {len(report['missing'])} allowed missing, "
                          f"{len(report['aliased'])} tied.")

    my_model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        my_model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # resume and load
    epoch = 0
    global_iter = 0

    print('work dir: ', args.work_dir)

    # training
    print_freq = cfg.print_freq
    forward_mode = 'ddp' if switch_values['ddp_forward'] else entry['train_forward']
    validation_mode = entry['validation']
    val_root = roots['validation']
    profile_ms = []
    stop = False
    max_steps = args.max_steps
    if screen:
        # A screen run is exactly cfg.screen_steps steps (fewer with a smaller --max-steps),
        # however many epochs that takes; the scheduler never reaches its total_steps.
        max_steps = min(max_steps, cfg.screen_steps) if max_steps else cfg.screen_steps

    while (epoch < max_num_epochs or screen) and not stop:
        epoch_start_iter = global_iter
        if shuffle_train and hasattr(train_dataloader, 'set_epoch'):
            train_dataloader.set_epoch(epoch)
        my_model.train()
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, batch in enumerate(train_dataloader):
            # forward + backward + optimize
            data_time_e = time.time()
            timing = len(profile_ms) < args.profile_steps
            if timing:
                _sync(accelerator.device)
                step_t0 = time.perf_counter()
            with accelerator.accumulate(my_model):
                optimizer.zero_grad()
                if forward_mode == 'module':
                    # legacy *_256 call: bypasses the DDP wrapper, gradients are not synchronised
                    loss, log, _, _, _, _, _, _, _ = my_model.module.forward(batch, "train", iter=global_iter, iter_end=cfg.max_train_steps)
                elif forward_mode == 'plain':
                    loss, log, _, _, _, _, _, _, _ = my_model.forward(batch, "train", iter=global_iter, iter_end=cfg.max_train_steps)
                else:
                    # D1 ddp_forward: through the wrapper, so DDP all-reduces the gradients
                    loss, log, _, _, _, _, _, _, _ = my_model(batch, "train", iter=global_iter, iter_end=cfg.max_train_steps)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                optimizer.step()
                scheduler.step()
            if timing:
                _sync(accelerator.device)
                profile_ms.append((time.perf_counter() - step_t0) * 1000.0)
                if len(profile_ms) == args.profile_steps:
                    report_profile(profile_ms, accelerator, args.work_dir)

            # Checks if the accelerator has performed an optimization step behind the scenes
            accelerator.wait_for_everyone()
            if accelerator.sync_gradients and accelerator.is_main_process:
                if global_iter > 0 and global_iter % cfg.save_freq == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(accelerator, args.work_dir, roots['latest'], global_iter, logger)

                if validation_mode is not None and global_iter > 0 and global_iter % cfg.val_freq == 0:
                    my_model.eval()
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            for i_iter_val, batch_val in enumerate(val_dataloader):
                                val_batch_save_dir = guard_save_path(osp.join(
                                    val_root, "step-{}/batch-{}".format(global_iter, i_iter_val)))
                                if validation_mode == 'module':
                                    log_val = my_model.module.validation_step(batch_val, val_batch_save_dir)
                                else:
                                    log_val = my_model.validation_step(batch_val, val_batch_save_dir)
                                log.update(log_val)
                    my_model.train()

            if forward_mode == 'ddp':
                # D1: the next DDP forward is a collective; hold every rank here until rank 0 is back from
                # its save / validation, or the others wait inside NCCL and time out after 1800 s.
                accelerator.wait_for_everyone()

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

            if max_steps and global_iter >= max_steps:
                stop = True
                break

        if screen and global_iter == epoch_start_iter:
            raise SystemExit(f"--entry {args.entry}: the train loader yielded no batch in epoch {epoch}")
        epoch += 1

    if (screen or args.save_final) and accelerator.is_main_process:
        # The periodic saves only happen at steps below the budget; keep the screen's final
        # weights (after global_iter steps) as checkpoint-<global_iter>. The other ranks wait below.
        # --save-final does the same for any row (the final weights of a full schedule).
        save_checkpoint(accelerator, args.work_dir, roots['latest'], global_iter, logger)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='CylinderSplat training; one --entry per row of configs/entries.py')
    parser.add_argument('--entry', required=True, choices=sorted(ENTRIES),
                        help='row of configs/entries.py (the legacy trainer to reproduce)')
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--run-id', type=str, default=None,
                        help='run name; the work dir defaults to <runs root>/<run-id>')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='every output of the run (default: tools.write_guard.default_run_dir(run_id))')
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument('--resume-from', type=str, default='',
                        help='checkpoint dir or model.safetensors (read-only); overrides cfg.resume_from')
    resume.add_argument('--no-resume', action='store_true',
                        help='ignore cfg.resume_from and start from the initialised weights')
    parser.add_argument('--transfer', default='exact', choices=sorted(TRANSFERS),
                        help='names the resume checkpoint may miss or add (configs/entries.py TRANSFERS)')
    parser.add_argument('--switch', action='append', default=[], metavar='NAME=VALUE',
                        help='opt-in switch (tools/switches.py), repeatable; overrides the config')
    parser.add_argument('--profile-steps', type=int, default=0,
                        help='time the first N train steps with torch.cuda.synchronize; 0 = off')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='stop after N train steps (equivalence / timing runs); 0 = full schedule')
    parser.add_argument('--save-final', action='store_true',
                        help='after the last step, save the final weights as checkpoint-<steps> (screen rows always do)')
    parser.add_argument('--screen-steps', type=int, default=None,
                        help="screen rows only: steps of the Phase-2 screen run and OneCycleLR total_steps - 100 "
                             "(default: the row's, 6000)")
    parser.add_argument('--seed', type=int, default=None,
                        help="screen rows only: replaces SEED of torch.manual_seed and cfg.seed of "
                             "set_seed(seed + local rank) (default: the row's, 42); the loader seeds stay")
    args = parser.parse_args(argv)
    screen = ENTRIES[args.entry].get('screen')
    if screen:
        if args.screen_steps is None:
            args.screen_steps = screen['steps']
        if args.seed is None:
            args.seed = screen['seed']
        if args.screen_steps <= 0:
            parser.error('--screen-steps must be positive')
        if not 0 <= args.seed < 2 ** 31:
            parser.error('--seed must be in [0, 2**31)')
    else:
        for flag, value in (('--screen-steps', args.screen_steps), ('--seed', args.seed)):
            if value is not None:
                parser.error(f"{flag} is only valid for the screen rows ({', '.join(sorted(SCREEN_ENTRIES))}); "
                             f"--entry {args.entry} {describe_entry(ENTRIES[args.entry])} and keeps its seeding")
    if not args.work_dir:
        if not args.run_id:
            parser.error('give --run-id or --work-dir')
        args.work_dir = default_run_dir(args.run_id)
    # Absolute before the process moves into <work dir>/cwd.
    args.work_dir = osp.abspath(args.work_dir)
    args.py_config = osp.abspath(args.py_config)
    if args.resume_from:
        args.resume_from = osp.abspath(args.resume_from)
    return args


if __name__ == '__main__':
    # Training settings
    args = parse_args()
    entry = ENTRIES[args.entry]

    # The legacy trainers import these at module level: data/dataloader.py sets the
    # process-wide cv2 options (setNumThreads(0), OpenCL off) that the loader workers inherit.
    import data.dataloader  # noqa: F401
    loader_module = importlib.import_module(entry['loader']['module'])

    SEED = 42 # 你可以选择任何固定的整数
    if args.seed is not None:
        SEED = args.seed  # screen rows only (parse_args); main() puts the same value into cfg.seed
    torch.manual_seed(SEED)

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    main(args, entry, loader_module)

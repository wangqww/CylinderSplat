"""CylinderSplat training entry point: one --entry per row of configs/entries.py.

    accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py \
        --entry mp3d_double_256 \
        --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
        --run-id all256 --resume-from <volume_256>/checkpoint-36000 --transfer stage2_to_stage3

    accelerate launch --config-file configs/accelerate/accel_1proc.yaml train.py \
        --entry kansas_double_160 --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_all.py --run-id kansas

    accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py \
        --entry mp3d_double_512_ddp3 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py \
        --run-id all512x1024 --resume-from <stage-3 run>/checkpoint-N --transfer stage3_to_stage4_512 \
        --switch ddp_forward=true

A row (configs/entries.py) fixes the loader and its DataLoader keywords, the batch-size
keys, the scheduler, how the training forward and validation are called, the process
count and the setup order. The stage-4 rows (STAGE4_ENTRIES) train on the 512x1024
loader of mp3d_double_512 with the recipe of mp3d_double_256, on 3 or 4 processes.

Every output goes under the work dir (--work-dir, or <runs root>/<run-id>): checkpoint-N,
the `latest` link, logs, the dumped config, switches.json and validation images. The
process works in <work dir>/cwd, so relative writes by the models stay inside the run.
Resuming loads weights only, under the name contract of tools/resume.py (--transfer);
optimizer and scheduler start fresh. Opt-in switches: tools/switches.py (--switch).
"""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import os.path as osp
import re
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

import mmengine
from mmengine import MMLogger
from mmengine.config import Config

from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, InitProcessGroupKwargs, DistributedDataParallelKwargs

from tools import switches as switch_registry
from tools.resume import load_weights_only, refuse_source_inside

import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = osp.dirname(osp.abspath(__file__))
ENTRIES_FILE = osp.join(REPO_ROOT, "configs", "entries.py")
# --run-id NAME writes to <runs root>/NAME.
RUNS_ROOT = os.environ.get("CYLINDERSPLAT_RUNS_ROOT", osp.join(REPO_ROOT, "workdirs"))
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_entries(path=ENTRIES_FILE):
    # configs/ is not a package; load the table by path.
    spec = importlib.util.spec_from_file_location("cylindersplat_entries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entries = load_entries()
STAGE4_ENTRIES = entries.STAGE4_ENTRIES
ENTRIES = dict(entries.ENTRIES, **STAGE4_ENTRIES)
TRANSFERS = entries.TRANSFERS


def create_logger(log_file=None, is_main_process=False, log_level=logging.INFO):
    if not is_main_process:
        return None
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
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


def default_run_dir(run_id):
    """<runs root>/<run_id>, absolute so that it does not move with the chdir into the scratch cwd."""
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run id {run_id!r}: use letters, digits, '.', '_' or '-'")
    return osp.abspath(osp.expanduser(osp.join(RUNS_ROOT, run_id)))


def enter_scratch_cwd(work_dir):
    """Create <work_dir>/cwd and make it the process cwd, so relative writes by the models stay in the run.

    The repository root goes on sys.path first, so imports keep working after the chdir.
    """
    scratch = osp.join(work_dir, "cwd")
    os.makedirs(scratch, exist_ok=True)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.chdir(scratch)


def build_dataloader(entry, loader_module, stage, batch_size, dataset_extra=None):
    """One stage's DataLoader: the row's dataset, read in order, with the stage's generator seed.

    dataset_extra adds dataset keywords (loc360_interleave: interleave=True on the 360Loc train split).
    """
    spec = entry["loader"]
    stage_spec = spec["stages"].get(stage, spec["stages"]["other"])
    dataset = getattr(loader_module, spec["dataset_class"])(**spec["dataset_kwargs"][stage], **(dataset_extra or {}))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=spec["num_workers"],
        generator=loader_module.get_generator(stage_spec["seed"]),
        worker_init_fn=loader_module.worker_init_fn,
        persistent_workers=stage_spec["persistent_workers"],
        shuffle=False,
    )


def build_loaders(entry, loader_module, dataset_config, interleave=False):
    train_dataloader = build_dataloader(
        entry,
        loader_module,
        "train",
        dataset_config[entry["batch_size"]["train"]],
        dict(interleave=True) if interleave else None,
    )
    val_dataloader = build_dataloader(entry, loader_module, "val", dataset_config[entry["batch_size"]["val"]])
    return train_dataloader, val_dataloader


def build_onecycle_scheduler(optimizer, cfg, train_dataloader, max_num_epochs):
    # PanSplat's OneCycle schedule over the whole run.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=len(train_dataloader) * max_num_epochs + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    return scheduler


def build_warmup_cosine_scheduler(optimizer, cfg, accelerator):
    # Linear warm-up, then cosine decay to 0.1 * lr.
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        1 / (cfg.warmup_steps * accelerator.num_processes),
        1,
        total_iters=cfg.warmup_steps * accelerator.num_processes,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_train_steps * accelerator.num_processes, eta_min=cfg.lr * 0.1
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm_up, scheduler], milestones=[cfg.warmup_steps * accelerator.num_processes]
    )
    return scheduler


def build_onecycle_steps_scheduler(optimizer, cfg):
    # The same OneCycle over a fixed number of steps (cfg.onecycle_total_steps): the fine-tunes of row
    # mp3d_double_256_screen, which stop at --max-steps instead of after max_epochs passes over the loader.
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=cfg.onecycle_total_steps,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )


def build_scheduler(entry, optimizer, cfg, accelerator, train_dataloader, max_num_epochs):
    if entry["scheduler"] == "onecycle":
        return build_onecycle_scheduler(optimizer, cfg, train_dataloader, max_num_epochs)
    if entry["scheduler"] == "onecycle_steps":
        return build_onecycle_steps_scheduler(optimizer, cfg)
    if entry["scheduler"] == "warmup_cosine":
        return build_warmup_cosine_scheduler(optimizer, cfg, accelerator)
    raise ValueError(f"unknown scheduler kind {entry['scheduler']!r}")


def seed_processes(cfg, accelerator):
    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)


def init_trackers(cfg, accelerator):
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="omni-gs",
            init_kwargs={
                "wandb": {"name": cfg.exp_name},
            },
        )


def build_model(cfg, accelerator):
    from builder import builder as model_builder

    return model_builder.build(cfg.model).to(accelerator.device)


def describe_entry(entry):
    if entry.get("stage4"):
        s4 = entry["stage4"]
        return f"runs stage 4 (the {s4['recipe_row']} recipe on the {s4['loader_row']} loader)"
    return f"trains on {entry['loader']['dataset_class']} from {entry['loader']['module']}"


class SparsityMonitor:
    """Stop rules of the volume sparsity budget (switch volume_sparsity, model/volume/sparsity.py) with the model's
    sparsity_args: a non-finite loss, or - when collapse_floor is set - from collapse_from on, a trailing
    collapse_window-step mean of the rendered volume share below collapse_floor x the trailing mean of its target
    (pruning that runs away from the budget). A failed arm writes <work dir>/arm_failed.json and exits with code 7
    before any further save (scripts/long_arm.sh then skips every evaluation and the test read). Inactive without the
    switch; with it, a run whose config sets no budget_start, or that runs on more than one process (a stop on one rank
    would leave the others waiting in the gradient all-reduce), is refused before the first step."""

    def __init__(self, model, work_dir, num_processes=1):
        self.active = bool(getattr(model, "volume_sparsity", False))
        self.args = dict(getattr(model, "sparsity_args", {}) or {})
        if self.active and num_processes != 1:
            raise SystemExit("volume_sparsity runs on one process (row mp3d_double_256_screen), "
                             f"not {num_processes}")
        if self.active and self.args.get("budget_start") is None:
            raise SystemExit("volume_sparsity: sparsity_args.budget_start is not set (the ceiling of the init's rendered "
                             "volume share; e.g. train s3k_ls_sp3.py, not its base s3k_sp3d.py)")
        self.work_dir = work_dir
        self.trail = []

    def fail(self, step, reason, **details):
        with open(osp.join(self.work_dir, "arm_failed.json"), "w") as f:
            json.dump(dict(step=step, reason=reason, **details), f, indent=2)
        print(f"volume sparsity: arm failed at step {step}: {reason}", flush=True)
        sys.exit(7)

    def check(self, step, loss, log):
        if not self.active:
            return
        if not torch.isfinite(loss.detach()).all():
            self.fail(step, "non-finite loss")
        floor = self.args.get("collapse_floor")
        if floor is None:
            return
        window = self.args.get("collapse_window", 100)
        self.trail = (self.trail + [(log["train/volume_share"], log["train/volume_share_target"])])[-window:]
        if step >= self.args.get("collapse_from", 1000) and len(self.trail) == window:
            share = sum(b for b, _ in self.trail) / window
            target = sum(r for _, r in self.trail) / window
            if share < floor * target:
                self.fail(step, "volume share collapsed", mean_volume_share=share, mean_target=target, steps=window)


def save_checkpoint(accelerator, work_dir, global_iter, logger):
    save_file_name = osp.join(osp.abspath(work_dir), f"checkpoint-{global_iter}")
    accelerator.save_state(save_file_name)
    mmengine.utils.symlink(save_file_name, osp.join(work_dir, "latest"))
    if logger is not None:
        logger.info("[TRAIN] Save latest state dict to {}.".format(save_file_name))


def check_row_switches(entry_name, entry, switch_values):
    if switch_values["loc360_interleave"] and entry["loader"]["dataset_class"] != "Dataset360Loc":
        raise SystemExit(
            f"--entry {entry_name}: loc360_interleave needs the 360Loc loader (Dataset360Loc), "
            f"not {entry['loader']['dataset_class']}"
        )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def report_profile(step_ms, accelerator, work_dir):
    """Wall time of the timed train steps (forward, backward, clip, optimizer and scheduler step)."""
    summary = dict(steps=len(step_ms), mean_ms=statistics.fmean(step_ms), median_ms=statistics.median(step_ms))
    if len(step_ms) > 1:
        summary.update(
            mean_ms_after_first=statistics.fmean(step_ms[1:]), median_ms_after_first=statistics.median(step_ms[1:])
        )
    rank = accelerator.process_index
    print(
        f"[profile] rank {rank}/{accelerator.num_processes}: "
        + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items())
    )
    with open(osp.join(work_dir, f"profile_steps_rank{rank}.json"), "w") as f:
        json.dump(dict(summary, step_ms=step_ms), f, indent=2)


def main(args, entry, loader_module):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    if args.resume_from:
        cfg.resume_from = args.resume_from
    elif args.no_resume:
        cfg.resume_from = False
    if cfg.resume_from:
        cfg.resume_from = osp.abspath(cfg.resume_from)  # before the move into <work dir>/cwd
    # Switches go into cfg.model before the model is built; all at default = the released behaviour.
    switch_values = switch_registry.apply(cfg, args.switch)
    check_row_switches(args.entry, entry, switch_values)
    transfer = TRANSFERS[args.transfer]
    if cfg.resume_from:
        refuse_source_inside(cfg.resume_from, args.work_dir)

    MMLogger.get_instance("mmengine", log_level="WARNING")
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=1800))]
    if switch_values["ddp_forward"]:
        # Forward through the DDP wrapper; parameters that do not reach the loss must not stall the all-reduce.
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir, logging_dir=osp.join(cfg.work_dir, "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers,
    )
    if accelerator.num_processes != entry["num_processes"]:
        raise SystemExit(
            f"--entry {args.entry} {describe_entry(entry)} with {entry['num_processes']} process(es); "
            f"this launch has {accelerator.num_processes}. "
            f"Use configs/accelerate/accel_{entry['num_processes']}proc.yaml."
        )

    enter_scratch_cwd(args.work_dir)
    if accelerator.is_main_process:
        with open(osp.join(args.work_dir, "switches.json"), "w") as f:
            json.dump(switch_values, f, indent=2, sort_keys=True)

    dataset_config = cfg.dataset_params
    max_num_epochs = cfg.max_epochs
    interleave = switch_values["loc360_interleave"]
    loaders_first = entry["setup_order"] == "loaders_before_model"
    if loaders_first:
        # 256 rows and stage 4: seed, loaders, trackers
        seed_processes(cfg, accelerator)
        train_dataloader, val_dataloader = build_loaders(entry, loader_module, dataset_config, interleave)
        init_trackers(cfg, accelerator)
    else:
        # 160 / 512 rows: trackers, seed; the loaders are built after the scheduler
        init_trackers(cfg, accelerator)
        seed_processes(cfg, accelerator)
        train_dataloader = val_dataloader = None

    # configure logger
    if accelerator.is_main_process:
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(args.work_dir, f"{timestamp}.log")
    logger = create_logger(log_file=log_file, is_main_process=accelerator.is_main_process)
    if logger is not None:
        logger.info(f"Config:\n{cfg.pretty_text}")
        logger.info(f"Entry: {args.entry} ({describe_entry(entry)}); switches: {switch_values}")

    # build model
    my_model = build_model(cfg, accelerator)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f"Number of params: {n_parameters}")

    optimizers = my_model.configure_optimizers(cfg.lr)
    optimizer = optimizers[0]
    scheduler = build_scheduler(entry, optimizer, cfg, accelerator, train_dataloader, max_num_epochs)

    if not loaders_first:
        train_dataloader, val_dataloader = build_loaders(entry, loader_module, dataset_config, interleave)

    path = cfg.resume_from
    if path:
        accelerator.print(f"Resuming from checkpoint {path} (weights only, transfer '{args.transfer}')")
        report = load_weights_only(my_model, path, transfer["allowed_missing"], transfer["allowed_extra"])
        accelerator.print(
            f"Model weights loaded successfully before prepare(): {len(report['matched'])} tensors, "
            f"{len(report['extra'])} allowed extra, {len(report['missing'])} allowed missing, "
            f"{len(report['aliased'])} tied."
        )

    my_model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        my_model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    epoch = 0
    global_iter = 0
    unwrapped = my_model.module if hasattr(my_model, "module") else my_model  # the DDP wrapper keeps it in .module
    sparsity_monitor = SparsityMonitor(unwrapped, args.work_dir, accelerator.num_processes)

    print("work dir: ", args.work_dir)

    # training
    print_freq = cfg.print_freq
    forward_mode = "ddp" if switch_values["ddp_forward"] else entry["train_forward"]
    validation_mode = entry["validation"]
    val_root = osp.join(args.work_dir, "validation")
    profile_ms = []
    stop = False
    max_steps = args.max_steps

    while epoch < max_num_epochs and not stop:
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
                if forward_mode == "module":
                    # Bypasses the DDP wrapper: gradients are not synchronised across ranks.
                    loss, log, _, _, _, _, _, _, _ = my_model.module.forward(
                        batch, "train", iter=global_iter, iter_end=cfg.max_train_steps
                    )
                elif forward_mode == "plain":
                    loss, log, _, _, _, _, _, _, _ = my_model.forward(
                        batch, "train", iter=global_iter, iter_end=cfg.max_train_steps
                    )
                else:
                    # ddp_forward: through the wrapper, so DDP all-reduces the gradients.
                    loss, log, _, _, _, _, _, _, _ = my_model(
                        batch, "train", iter=global_iter, iter_end=cfg.max_train_steps
                    )

                sparsity_monitor.check(global_iter, loss, log)
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
                    save_checkpoint(accelerator, args.work_dir, global_iter, logger)

                if validation_mode is not None and global_iter > 0 and global_iter % cfg.val_freq == 0:
                    my_model.eval()
                    with torch.no_grad():
                        for i_iter_val, batch_val in enumerate(val_dataloader):
                            val_batch_save_dir = osp.join(val_root, "step-{}/batch-{}".format(global_iter, i_iter_val))
                            if validation_mode == "module":
                                log_val = my_model.module.validation_step(batch_val, val_batch_save_dir)
                            else:
                                log_val = my_model.validation_step(batch_val, val_batch_save_dir)
                            log.update(log_val)
                    my_model.train()

            if forward_mode == "ddp":
                # The next DDP forward is a collective; hold every rank here until rank 0 is back from
                # its save / validation, or the others wait inside NCCL and time out after 1800 s.
                accelerator.wait_for_everyone()

            time_e = time.time()

            # print loss log regularly
            if global_iter % print_freq == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                losses_str = ""
                for loss_k, loss_v in log.items():
                    losses_str += "%s: %.3f, " % (loss_k, loss_v)
                if logger is not None:
                    logger.info(
                        "[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f, %s grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)"
                        % (
                            epoch,
                            i_iter,
                            len(train_dataloader),
                            loss.item(),
                            losses_str,
                            grad_norm,
                            lr,
                            time_e - time_s,
                            data_time_e - data_time_s,
                        )
                    )

            global_iter += 1

            # dump loss log to tensorboard
            accelerator.log(log, step=global_iter)

            data_time_s = time.time()
            time_s = time.time()

            if max_steps and global_iter >= max_steps:
                stop = True
                break

        epoch += 1

    if args.save_final and accelerator.is_main_process:
        # The periodic saves skip the last step; keep the final weights as checkpoint-<global_iter>.
        # The other ranks wait below.
        save_checkpoint(accelerator, args.work_dir, global_iter, logger)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CylinderSplat training; one --entry per row of configs/entries.py")
    parser.add_argument(
        "--entry",
        required=True,
        choices=sorted(ENTRIES),
        help="row of configs/entries.py (loader, scheduler, forward call, process count)",
    )
    parser.add_argument("--py-config", required=True)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="run name; the work dir defaults to <runs root>/<run-id> (runs root: "
        "$CYLINDERSPLAT_RUNS_ROOT, else workdirs/ in the repository)",
    )
    parser.add_argument("--work-dir", type=str, default=None, help="every output of the run")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="checkpoint dir or model.safetensors (read-only); overrides cfg.resume_from",
    )
    resume.add_argument(
        "--no-resume", action="store_true", help="ignore cfg.resume_from and start from the initialised weights"
    )
    parser.add_argument(
        "--transfer",
        default="exact",
        choices=sorted(TRANSFERS),
        help="names the resume checkpoint may miss or add (configs/entries.py TRANSFERS)",
    )
    parser.add_argument(
        "--switch",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="opt-in switch (tools/switches.py), repeatable; overrides the config",
    )
    parser.add_argument(
        "--profile-steps", type=int, default=0, help="time the first N train steps with torch.cuda.synchronize; 0 = off"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after N train steps (short check or timing runs); 0 = full schedule",
    )
    parser.add_argument(
        "--save-final", action="store_true", help="after the last step, save the final weights as checkpoint-<steps>"
    )
    args = parser.parse_args(argv)
    if not args.work_dir:
        if not args.run_id:
            parser.error("give --run-id or --work-dir")
        try:
            args.work_dir = default_run_dir(args.run_id)
        except ValueError as e:
            parser.error(str(e))
    # Absolute before the process moves into <work dir>/cwd.
    args.work_dir = osp.abspath(args.work_dir)
    args.py_config = osp.abspath(args.py_config)
    if args.resume_from:
        args.resume_from = osp.abspath(args.resume_from)
    return args


if __name__ == "__main__":
    # Training settings
    args = parse_args()
    entry = ENTRIES[args.entry]

    # Process-wide OpenCV options, inherited by the loader workers.
    import cv2

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    loader_module = importlib.import_module(entry["loader"]["module"])

    SEED = 42
    torch.manual_seed(SEED)

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    main(args, entry, loader_module)

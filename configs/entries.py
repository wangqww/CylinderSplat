"""Training rows for train.py (--entry) and the weight-transfer lists of the resume rule.

Pure data, no imports: train.py and the tests load this file by path. A row fixes
everything about a run that the model config does not: the loader module, dataset
class and DataLoader keywords, where the batch sizes come from, the learning-rate
schedule, the setup order, how the training forward and validation are called, and
the process count. Nothing here reads cfg.dataset_name: the row chosen with --entry
decides the loader.
"""


# ---------------------------------------------------------------------------
# Keywords of the DataLoader(...) that train.py builds, keyword -> source
# expression. Only the num_workers literal differs between loaders; the loader
# modules' load_*() factories build the same DataLoader.
# ---------------------------------------------------------------------------
def _dataloader_call(num_workers):
    return {
        "batch_size": "batch_size",
        "num_workers": str(num_workers),
        "generator": "get_generator(seed)",
        "worker_init_fn": "worker_init_fn",
        "persistent_workers": "persistent_workers",
        "shuffle": "False",
    }


# Per-stage generator seed and persistent_workers ("other" = any other stage).
# The DataLoader always gets shuffle=False: map-style datasets are read in order,
# the 360Loc IterableDataset shuffles its train split itself.
LOADER_STAGES = {
    "train": dict(seed=1234, persistent_workers=True),
    "val": dict(seed=3456, persistent_workers=True),
    "test": dict(seed=2345, persistent_workers=False),
    "other": dict(seed=6789, persistent_workers=True),
}

# ---------------------------------------------------------------------------
# Scheduler assignments of train.py, in order, as source expressions.
# ---------------------------------------------------------------------------
SCHEDULERS = {
    # OneCycle (PanSplat's schedule) over the whole run: the 256 rows and stage 4.
    "onecycle": [
        dict(target="scheduler", call="torch.optim.lr_scheduler.OneCycleLR",
             args=["optimizer"],
             kwargs={
                 "max_lr": "cfg.lr",
                 "total_steps": "len(train_dataloader) * max_num_epochs + 100",
                 "pct_start": "0.01",
                 "cycle_momentum": "False",
                 "anneal_strategy": "'cos'",
                 "div_factor": "25.0",
                 "final_div_factor": "10000.0",
             }),
    ],
    # The same OneCycle over a fixed step count from the config: the screen row.
    "onecycle_steps": [
        dict(target="scheduler", call="torch.optim.lr_scheduler.OneCycleLR",
             args=["optimizer"],
             kwargs={
                 "max_lr": "cfg.lr",
                 "total_steps": "cfg.onecycle_total_steps",
                 "pct_start": "0.01",
                 "cycle_momentum": "False",
                 "anneal_strategy": "'cos'",
                 "div_factor": "25.0",
                 "final_div_factor": "10000.0",
             }),
    ],
    # Linear warm-up then cosine: the 160 and 512 rows.
    "warmup_cosine": [
        dict(target="warm_up", call="torch.optim.lr_scheduler.LinearLR",
             args=["optimizer", "1 / (cfg.warmup_steps * accelerator.num_processes)", "1"],
             kwargs={"total_iters": "cfg.warmup_steps * accelerator.num_processes"}),
        dict(target="scheduler", call="torch.optim.lr_scheduler.CosineAnnealingLR",
             args=["optimizer"],
             kwargs={"T_max": "cfg.max_train_steps * accelerator.num_processes", "eta_min": "cfg.lr * 0.1"}),
        dict(target="scheduler", call="torch.optim.lr_scheduler.SequentialLR",
             args=["optimizer"],
             kwargs={"schedulers": "[warm_up, scheduler]",
                     "milestones": "[cfg.warmup_steps * accelerator.num_processes]"}),
    ],
}

# Arguments of the training forward and of validation_step (the same for every row).
TRAIN_FORWARD_ARGS = (["batch", "'train'"], {"iter": "global_iter", "iter_end": "cfg.max_train_steps"})
VALIDATION_STEP_ARGS = (["batch_val", "val_batch_save_dir"], {})

# Setup orders (what runs between Accelerator() and accelerator.prepare()):
#   loaders_before_model: set_seed, loaders, init_trackers, config dump + log, model,
#                         optimizer, scheduler, resume     (the 256 rows and stage 4)
#   model_before_loaders: init_trackers, set_seed, config dump + log, model, optimizer,
#                         scheduler, loaders, resume       (the 160 and 512 rows)

# ---------------------------------------------------------------------------
# Rows. Fields:
#   configs          the configs trained with this row (informative)
#   loader           module / factory / dataset class, the dataset kwargs per stage,
#                    the DataLoader(...) keywords, num_workers, shuffle, per-stage
#                    seeds, and whether the dataset is an IterableDataset
#   batch_size       dataset_params key each loader's batch size is read from
#   scheduler        key into SCHEDULERS
#   setup_order      see above
#   train_forward    'module' = my_model.module.forward (bypasses DDP, no gradient sync;
#                    --switch ddp_forward=true calls the wrapper instead),
#                    'plain'  = my_model.forward
#   num_processes    the only process count train.py accepts for the row
#   validation       'module' / 'plain' = every val_freq via my_model[.module].validation_step,
#                    None = no validation loop (the val loader is still built and prepared)
# train.py never sets CUDA_VISIBLE_DEVICES: the GPUs come from the launch config.
# ---------------------------------------------------------------------------
ENTRIES = {
    # MP3D two-view, 256x512: stages 1-3.
    "mp3d_double_256": dict(
        configs=[
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_mgs.py",
            "configs/OmniScene/release/stage1_pixel_256_b4.py",
            "configs/OmniScene/release/stage2_volume_256_b4.py",
            "configs/OmniScene/release/stage3_all_256.py",
        ],
        loader=dict(
            module="data.mp3d_dataloader_double_256",
            factory="load_MP3D_data",
            dataset_class="DatasetMP3D",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(32),
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),
        scheduler="onecycle",
        setup_order="loaders_before_model",
        train_forward="module",
        num_processes=3,
        validation="module",
    ),
    # MP3D single view (one context view per sample), 256x512.
    "mp3d_single_256": dict(
        configs=[
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256_single.py",
        ],
        loader=dict(
            module="data.mp3d_dataloader_single_256",
            factory="load_MP3D_data",
            dataset_class="DatasetMP3D",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(32),
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),
        scheduler="onecycle",
        setup_order="loaders_before_model",
        train_forward="module",
        num_processes=3,
        validation="module",
    ),
    # 360Loc fine-tune, 256x512 (the loader module is named *_512 for historical reasons).
    "loc360_all_256": dict(
        configs=[
            "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py",
            "configs/OmniScene/release/loc360_finetune_256.py",
        ],
        loader=dict(
            module="data.loc360_dataloader_double_all_512",
            factory="load_360Loc_data",
            dataset_class="Dataset360Loc",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(1),
            num_workers=1,
            shuffle=False,
            stages=LOADER_STAGES,
            # IterableDataset: the train split shuffles its sequences in __iter__.
            iterable=True,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_val"),
        scheduler="onecycle",
        setup_order="loaders_before_model",
        train_forward="module",
        num_processes=3,
        validation=None,
    ),
    # MP3D two-view, 512x1024, one process.
    "mp3d_double_512": dict(
        configs=[
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_512.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512.py",
        ],
        loader=dict(
            module="data.mp3d_dataloader_double_512",
            factory="load_MP3D_data",
            dataset_class="DatasetMP3D",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(32),
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),
        scheduler="warmup_cosine",
        setup_order="model_before_loaders",
        train_forward="plain",
        num_processes=1,
        validation="plain",
    ),
    # Kansas City (VIGOR), 160x320, one process.
    "kansas_double_160": dict(
        configs=[
            "configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_pixel.py",
            "configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_volume.py",
            "configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_all.py",
            "configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_pixel_unifuse.py",
        ],
        loader=dict(
            module="data.vigor_dataloader_double",
            factory="load_VIGOR_data",
            dataset_class="SatGrdDataset",
            dataset_kwargs=dict(
                train=dict(root_dir="/data/qiwei/nips25/", is_train=True),
                val=dict(root_dir="/data/qiwei/nips25/", is_train=False),
            ),
            call=_dataloader_call(32),
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),
        scheduler="warmup_cosine",
        setup_order="model_before_loaders",
        train_forward="plain",
        num_processes=1,
        validation="plain",
    ),
}

# ---------------------------------------------------------------------------
# Fine-tune row: fine-tunes of a trained checkpoint on one GPU through
# scripts/long_arm.sh. The mp3d_double_256 loaders on one process
# (batch_size_train per step), a OneCycle over cfg.onecycle_total_steps (runs stop
# at --max-steps with --save-final), no in-run validation (runs are evaluated with
# evaluate.py). long_s = the LS recipe (20,000 steps from the released stage 3;
# README, "LS fine-tune"), long_c0 = the same schedule with prune_invisible only (no lpips_input_range,
# sampling_align, ws_loss, and the default fused depth weight).
# ---------------------------------------------------------------------------
ENTRIES["mp3d_double_256_screen"] = dict(
    configs=[
        "configs/OmniScene/screen/stage3_screen.py",
        "configs/OmniScene/screen/long_c0.py",
        "configs/OmniScene/screen/long_s.py",
    ],
    loader=ENTRIES["mp3d_double_256"]["loader"],
    batch_size=ENTRIES["mp3d_double_256"]["batch_size"],
    scheduler="onecycle_steps",
    setup_order="loaders_before_model",
    train_forward="plain",
    num_processes=1,
    validation=None,
)

# ---------------------------------------------------------------------------
# Stage-4 rows (the fourth stage of the MP3D schedule: the stage-3 joint model
# trained further at 512x1024). Each combines two rows above:
#   loader, batch_size   the mp3d_double_512 row's: data.mp3d_dataloader_double_512
#                        (1024x512 panoramas at full size), the same dataset, DataLoader
#                        keywords, workers and stage seeds; train and val loaders both
#                        read batch_size_train
#   scheduler, setup_order, train_forward, validation
#                        the mp3d_double_256 row's: OneCycleLR with max_lr = cfg.lr and
#                        total_steps = len(train_dataloader) * max_epochs + 100, loaders
#                        before the model, .module.forward (--switch ddp_forward=true
#                        runs the wrapped DDP call), validation every val_freq through .module
#   num_processes        3 or 4, one row per launch size
#   stage4               loader_row / recipe_row: the two rows above; resolution: the
#                        image size [H, W] the row's loader yields, which the config
#                        must build the model for
# The config is all_256 with resolution = [512, 1024] (camera_args, perceptual
# resolution and pixel_gs.image_height follow it; no parameter shape depends on
# it). Init is weights-only: --resume-from <stage-3 checkpoint> --transfer
# stage3_to_stage4_512.
# ---------------------------------------------------------------------------
_MP3D_512 = ENTRIES["mp3d_double_512"]


def _stage4_row(num_processes):
    return dict(
        configs=["configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py"],
        loader=_MP3D_512["loader"],
        batch_size=_MP3D_512["batch_size"],
        scheduler="onecycle",
        setup_order="loaders_before_model",
        train_forward="module",
        num_processes=num_processes,
        validation="module",
        stage4=dict(loader_row="mp3d_double_512", recipe_row="mp3d_double_256", resolution=[512, 1024]),
    )


STAGE4_ENTRIES = {
    "mp3d_double_512_ddp3": _stage4_row(3),
    "mp3d_double_512_ddp4": _stage4_row(4),
}

# ---------------------------------------------------------------------------
# Resume rule (tools/resume.py): the names a weight transfer may leave missing
# (in the model, not in the checkpoint) or extra (in the checkpoint, not in the
# model). "prefix.*" matches every name starting with "prefix."; anything else
# is an exact name (tests/test_resume_rule.py checks these lists).
# ---------------------------------------------------------------------------
TRANSFERS = {
    # Default: checkpoint and model have identical names, dtypes and shapes.
    "exact": dict(allowed_missing=[], allowed_extra=[]),
    # pixel_256 checkpoint -> volume model (stage 1 -> 2): the pixel model's depth
    # network is not part of the volume model (333 tensors).
    "stage1_to_stage2": dict(allowed_missing=[], allowed_extra=["pixel_gs.mono_depth.*"]),
    # double-view pixel checkpoint -> single-view pixel model: same 333 tensors.
    "double_pixel_to_single_pixel": dict(allowed_missing=[], allowed_extra=["pixel_gs.mono_depth.*"]),
    # volume_256 -> all_256 (stage 2 -> 3): identical key and shape sets.
    "stage2_to_stage3": dict(allowed_missing=[], allowed_extra=[]),
    # all_256 (stage 3) -> the same architecture built at 512x1024 (stage 4, rows
    # mp3d_double_512_ddp3/4): no parameter shape depends on the image resolution,
    # so the key and shape sets are identical.
    "stage3_to_stage4_512": dict(allowed_missing=[], allowed_extra=[]),
    # MP3D all_256 -> 360Loc fine-tune (row loc360_all_256).
    "mp3d_all_256_to_loc360_pan2": dict(allowed_missing=[], allowed_extra=[]),
}

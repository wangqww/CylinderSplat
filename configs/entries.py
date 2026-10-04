"""Training entries for train.py (plan table T), the Phase-2 screen rows, the stage-4
rows, and the weight-transfer lists of the resume rule.

Pure data, no imports: train.py and the tests load this file by path. Every
table-T value is copied from the live code at f7b20b9 (legacy/ keeps the old
trainers byte-for-byte, so their line numbers still hold); tests/test_entries_table.py
parses those sources and fails when a value here drifts from them. The screen
rows (SCREEN_ENTRIES) reuse a table-T row's loader and change only what plan §7
names. The stage-4 rows (STAGE4_ENTRIES) combine the 512x1024 loader of one table-T
row with the training recipe of another. Nothing here reads cfg.dataset_name: the
row chosen with --entry decides the loader.
"""

# ---------------------------------------------------------------------------
# DataLoader(...) of the six load_*() factories, keyword -> source expression,
# verbatim (ast.unparse). Only the num_workers literal differs between loaders.
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


# The factories' `if stage == ...` chain: per-stage generator seed and
# persistent_workers ("other" = the final else). Their local `shuffle` is
# computed but never passed on; the DataLoader always gets shuffle=False.
LOADER_STAGES = {
    "train": dict(seed=1234, persistent_workers=True),
    "val": dict(seed=3456, persistent_workers=True),
    "test": dict(seed=2345, persistent_workers=False),
    "other": dict(seed=6789, persistent_workers=True),
}

# ---------------------------------------------------------------------------
# Live scheduler assignments of the legacy trainers, in order, verbatim.
# ---------------------------------------------------------------------------
SCHEDULERS = {
    # PanSplat OneCycle of the *_256 trainers (train_mp3d_cylinder_double_256.py:131-142).
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
    # Linear warm-up then cosine of the 160 / 512 trainers (train_mp3d_cylinder_double.py:117-124).
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
# Phase-2 screen (plan §7 "Screen recipe"; not a legacy assignment): the OneCycleLR
# above with every keyword kept (max_lr = cfg.lr) except total_steps = cfg.screen_steps + 100.
# train.py sets cfg.screen_steps from --screen-steps or the row's screen['steps'].
SCHEDULERS["onecycle_screen"] = [
    dict(SCHEDULERS["onecycle"][0],
         kwargs=dict(SCHEDULERS["onecycle"][0]["kwargs"], total_steps="cfg.screen_steps + 100")),
]

# Arguments of the training forward and of validation_step (identical in all six trainers).
TRAIN_FORWARD_ARGS = (["batch", "'train'"], {"iter": "global_iter", "iter_end": "cfg.max_train_steps"})
VALIDATION_STEP_ARGS = (["batch_val", "val_batch_save_dir"], {})

# Setup orders (what runs between Accelerator() and accelerator.prepare()):
#   loaders_before_model: set_seed, loaders, init_trackers, config dump + log, model,
#                         optimizer, scheduler, resume     (the *_256 trainers, :70-188)
#   model_before_loaders: init_trackers, set_seed, config dump + log, model, optimizer,
#                         scheduler, loaders, resume       (the 160 / 512 trainers, :69-173)

# ---------------------------------------------------------------------------
# Table T. Fields:
#   legacy_script    the trainer this row reproduces
#   configs          configs the author trained with this trainer (legacy/run2.sh); informative
#   loader           module / factory / dataset class, the dataset kwargs the factory passes per
#                    stage, the factory's verbatim DataLoader(...) keywords, num_workers (C3),
#                    shuffle, per-stage seeds, and whether the dataset is an IterableDataset
#   batch_size       dataset_params key each loader's batch size is read from
#   scheduler        key into SCHEDULERS
#   setup_order      see above
#   train_forward    'module' = my_model.module.forward (bypasses DDP, no gradient sync),
#                    'plain'  = my_model.forward
#   num_processes    the only process count train.py accepts for the row
#   validation       'module' / 'plain' = every val_freq via my_model[.module].validation_step,
#                    None = no validation loop (the val loader is still built and prepared)
#   legacy_gpu_pin   the CUDA_VISIBLE_DEVICES the legacy script sets at import (None = not set).
#                    train.py never sets it: the GPUs come from the launch config (INV-4).
# ---------------------------------------------------------------------------
ENTRIES = {
    "mp3d_double_256": dict(
        legacy_script="legacy/train_mp3d_cylinder_double_256.py",
        configs=[
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py",
        ],
        loader=dict(
            module="data.mp3d_dataloader_double_256",
            factory="load_MP3D_data",
            dataset_class="DatasetMP3D",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(32),  # loader :322-330
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),  # :77-78
        scheduler="onecycle",  # :131-142
        setup_order="loaders_before_model",
        train_forward="module",  # :214
        num_processes=3,
        validation="module",  # :241-248
        legacy_gpu_pin=None,
    ),
    "mp3d_single_256": dict(
        legacy_script="legacy/train_mp3d_cylinder_single_256.py",
        configs=["configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py"],
        loader=dict(
            module="data.mp3d_dataloader_single_256",
            factory="load_MP3D_data",
            dataset_class="DatasetMP3D",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(32),  # loader :322-330
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),
        scheduler="onecycle",  # :131-142
        setup_order="loaders_before_model",
        train_forward="module",  # :214
        num_processes=3,
        validation="module",  # :241-248
        legacy_gpu_pin=None,
    ),
    "loc360_all_256": dict(
        # The *_512 script name is historical: its loader reads 256x512 panoramas.
        legacy_script="legacy/train_360Loc_cylinder_double_all_512.py",
        configs=["configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py"],
        loader=dict(
            module="data.loc360_dataloader_double_all_512",
            factory="load_360Loc_data",
            dataset_class="Dataset360Loc",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(1),  # loader :372-380
            num_workers=1,
            shuffle=False,
            stages=LOADER_STAGES,
            # IterableDataset: the train split shuffles its sequences in __iter__ (:154-155),
            # so the shuffle_train switch is refused for this row.
            iterable=True,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_val"),  # :77-78
        scheduler="onecycle",  # :131-142
        setup_order="loaders_before_model",
        train_forward="module",  # :215
        num_processes=3,
        validation=None,
        legacy_gpu_pin=None,
    ),
    "mp3d_double_512": dict(
        legacy_script="legacy/train_mp3d_cylinder_double_512.py",
        configs=[
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_512.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512.py",
        ],
        loader=dict(
            module="data.mp3d_dataloader_double_512",
            factory="load_MP3D_data",
            dataset_class="DatasetMP3D",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(32),  # loader :315-323
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),  # :144-145
        scheduler="warmup_cosine",  # :116-123
        setup_order="model_before_loaders",
        train_forward="plain",  # :185
        num_processes=1,
        validation="plain",  # :207-213
        legacy_gpu_pin=None,
    ),
    "mp3d_double_160": dict(
        legacy_script="legacy/train_mp3d_cylinder_double.py",
        configs=[
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py",
        ],
        loader=dict(
            module="data.mp3d_dataloader_double",
            factory="load_MP3D_data",
            dataset_class="DatasetMP3D",
            dataset_kwargs=dict(train=dict(stage="train"), val=dict(stage="val")),
            call=_dataloader_call(32),  # loader :304-312
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),  # :145-146
        scheduler="warmup_cosine",  # :117-124
        setup_order="model_before_loaders",
        train_forward="plain",  # :198
        num_processes=1,  # legacy/run2.sh:42,91 (plain python; the script pins CUDA_VISIBLE_DEVICES=0)
        validation="plain",  # :226-232
        legacy_gpu_pin="0",  # os.environ at :2-3
    ),
    "kansas_double_160": dict(
        legacy_script="legacy/train_vigor_cylinder_double.py",
        configs=[
            "configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_pixel.py",
            "configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_volume.py",
            "configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_all.py",
        ],
        loader=dict(
            module="data.vigor_dataloader_double",
            factory="load_VIGOR_data",
            dataset_class="SatGrdDataset",
            dataset_kwargs=dict(
                train=dict(root_dir="/data/qiwei/nips25/", is_train=True),
                val=dict(root_dir="/data/qiwei/nips25/", is_train=False),
            ),
            call=_dataloader_call(32),  # loader :375-383
            num_workers=32,
            shuffle=False,
            stages=LOADER_STAGES,
            iterable=False,
        ),
        batch_size=dict(train="batch_size_train", val="batch_size_train"),  # :145-146
        scheduler="warmup_cosine",  # :117-124
        setup_order="model_before_loaders",
        train_forward="plain",  # :198
        num_processes=1,  # legacy/run2.sh:229-246 (plain python)
        validation="plain",  # :226-232
        legacy_gpu_pin="0",  # os.environ at :2-3
    ),
}

# ---------------------------------------------------------------------------
# Phase-2 screen rows (plan §7 "Screen recipe"). Not part of table T: they
# reproduce no legacy trainer, so the tests check them against their base row.
# Each shares its base row's loader dict (dataset, DataLoader keywords, workers,
# stage seeds), batch sizes and setup order, and differs only in:
#   num_processes   1 (C0 and the one-switch arms: one process over the full
#                   training loader, no shard) or 2 (the D1 pair)
#   train_forward   'plain' with one process (Accelerate does not wrap the model,
#                   so there is no .module); 'module' (legacy) for the D1 pair,
#                   which the ddp_forward switch turns into the wrapped DDP call
#   validation      the base row's loop (every val_freq, or none), called the
#                   same way as the training forward
#   scheduler       'onecycle_screen' (SCHEDULERS): max_lr = cfg.lr,
#                   total_steps = cfg.screen_steps + 100
#   screen          base:  the table-T row it is derived from
#                   steps: default of --screen-steps; the run stops after that
#                          many steps whatever cfg.max_epochs says, then saves
#                          checkpoint-<steps>
#                   seed:  default of --seed (plan §7: 42; the C0 repeat uses 43);
#                          replaces both the module-level SEED of torch.manual_seed
#                          and cfg.seed of set_seed(cfg.seed + local_process_index)
#   legacy_script   None (no legacy trainer)
# Init is weights-only via --resume-from / --transfer, as for every row
# (optimizer and scheduler start fresh).
# ---------------------------------------------------------------------------
_MP3D_256 = ENTRIES["mp3d_double_256"]
_MP3D_SINGLE_256 = ENTRIES["mp3d_single_256"]
_LOC360_256 = ENTRIES["loc360_all_256"]

SCREEN_ENTRIES = {
    # C0 and the one-switch MP3D arms (lr 2e-4); the D2b pair uses the volume_256 config.
    "screen_mp3d_all_256": dict(
        legacy_script=None,
        configs=[
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py",
            "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py",
        ],
        loader=_MP3D_256["loader"],
        batch_size=_MP3D_256["batch_size"],
        scheduler="onecycle_screen",
        setup_order=_MP3D_256["setup_order"],
        train_forward="plain",
        num_processes=1,
        validation="plain",
        legacy_gpu_pin=None,
        screen=dict(base="mp3d_double_256", steps=6000, seed=42),
    ),
    # The MP3D D3 pair (lr 2e-4): the single-view MP3D loader (one context view per sample) and
    # all_256 built with num_frames=1, without and with --switch v1_identity_pose=true. The
    # plan's Pan2 v=1 arm on 360Loc has no row: it needs a single-view 360Loc loader, which
    # does not exist yet.
    "screen_mp3d_single_256": dict(
        legacy_script=None,
        configs=["configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256_single.py"],
        loader=_MP3D_SINGLE_256["loader"],
        batch_size=_MP3D_SINGLE_256["batch_size"],
        scheduler="onecycle_screen",
        setup_order=_MP3D_SINGLE_256["setup_order"],
        train_forward="plain",
        num_processes=1,
        validation="plain",
        legacy_gpu_pin=None,
        screen=dict(base="mp3d_single_256", steps=6000, seed=42),
    ),
    # The 360Loc Pan2 arms (D4; lr 1e-4); IterableDataset, so shuffle_train stays refused.
    "screen_loc360_all_256": dict(
        legacy_script=None,
        configs=["configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py"],
        loader=_LOC360_256["loader"],
        batch_size=_LOC360_256["batch_size"],
        scheduler="onecycle_screen",
        setup_order=_LOC360_256["setup_order"],
        train_forward="plain",
        num_processes=1,
        validation=None,
        legacy_gpu_pin=None,
        screen=dict(base="loc360_all_256", steps=6000, seed=42),
    ),
    # The D1 pair: 2 processes, same per-rank batch and sample order; legacy .module.forward
    # by default, wrapped DDP with --switch ddp_forward=true.
    "screen_d1_pair_256": dict(
        legacy_script=None,
        configs=["configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py"],
        loader=_MP3D_256["loader"],
        batch_size=_MP3D_256["batch_size"],
        scheduler="onecycle_screen",
        setup_order=_MP3D_256["setup_order"],
        train_forward="module",
        num_processes=2,
        validation="module",
        legacy_gpu_pin=None,
        screen=dict(base="mp3d_double_256", steps=6000, seed=42),
    ),
}

# ---------------------------------------------------------------------------
# Stage-4 rows (the fourth stage of the MP3D schedule: the stage-3 joint model
# trained further at 512x1024). Not part of table T: they reproduce no legacy
# trainer (legacy/train_mp3d_cylinder_double_512.py trained the different
# all_512 architecture on one process), so the tests check them against the two
# rows they are built from:
#   loader, batch_size   the mp3d_double_512 row's: data.mp3d_dataloader_double_512
#                        (1024x512 panoramas at full size), the same dataset, DataLoader
#                        keywords, workers and stage seeds; train and val loaders both
#                        read batch_size_train
#   scheduler, setup_order, train_forward, validation
#                        the mp3d_double_256 row's: OneCycleLR with max_lr = cfg.lr and
#                        total_steps = len(train_dataloader) * max_epochs + 100, loaders
#                        before the model, legacy .module.forward (--switch
#                        ddp_forward=true runs the wrapped DDP call, D1), validation
#                        every val_freq through .module
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
        legacy_script=None,
        # all_256 built at resolution = [512, 1024]; the stage-4 config itself is kept with the run scripts.
        configs=["configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py"],
        loader=_MP3D_512["loader"],
        batch_size=_MP3D_512["batch_size"],
        scheduler="onecycle",
        setup_order="loaders_before_model",
        train_forward="module",
        num_processes=num_processes,
        validation="module",
        legacy_gpu_pin=None,
        stage4=dict(loader_row="mp3d_double_512", recipe_row="mp3d_double_256", resolution=[512, 1024]),
    )


STAGE4_ENTRIES = {
    "mp3d_double_512_ddp3": _stage4_row(3),
    "mp3d_double_512_ddp4": _stage4_row(4),
}

# ---------------------------------------------------------------------------
# Resume rule (plan A1): the names a weight transfer may leave missing (in the
# model, not in the checkpoint) or extra (in the checkpoint, not in the model).
# "prefix.*" matches every name starting with "prefix."; anything else is an
# exact name. Frozen from a header dry-run of the real checkpoints on
# 2026-09-30; tests/fixtures/allowed_keys.json is the frozen copy these must equal.
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
    # MP3D all_256 -> 360Loc Pan2 (row loc360_all_256).
    "mp3d_all_256_to_loc360_pan2": dict(allowed_missing=[], allowed_extra=[]),
    # D3 arm: 2-view all_256 -> OmniGaussianCylinderAll built with num_frames=1
    # (omni_gs_160x320_mp3d_cylinder_all_256_single.py, row screen_mp3d_single_256).
    "d3_single_view": dict(allowed_missing=[], allowed_extra=[]),
    # Pan2 v=1 arm: Pan2 checkpoint -> Pan2 with PixelGaussian360Loc(num_frames=1).
    "pan2_single_view": dict(allowed_missing=[], allowed_extra=[]),
    # D7 arms (rgb_retrieval='visibility_softmax'): allowed-missing is exactly the seven
    # new volume_gs.gs_decoder.gaussian_to_color_vis.* parameters of the
    # visibility-softmax colour head (frozen from the implemented module, 2026-09-30;
    # kept equal to tests/fixtures/allowed_keys.json).
    "d7_visibility_softmax": dict(allowed_missing=[
        "volume_gs.gs_decoder.gaussian_to_color_vis.encoder.0.weight",
        "volume_gs.gs_decoder.gaussian_to_color_vis.encoder.0.bias",
        "volume_gs.gs_decoder.gaussian_to_color_vis.head.0.weight",
        "volume_gs.gs_decoder.gaussian_to_color_vis.head.0.bias",
        "volume_gs.gs_decoder.gaussian_to_color_vis.head.2.weight",
        "volume_gs.gs_decoder.gaussian_to_color_vis.head.2.bias",
        "volume_gs.gs_decoder.gaussian_to_color_vis.log_beta",
    ], allowed_extra=[]),
}

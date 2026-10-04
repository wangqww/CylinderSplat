# legacy/

This directory holds the scripts of the released code (commit `f7b20b9`). They used to sit at the
repository root and are now kept here **byte for byte**: every file equals
`git show f7b20b9:<file name>`, including `run2.sh` and `.vscode/launch.json`, and their
`CUDA_VISIBLE_DEVICES` lines are untouched. They serve two purposes:

- **Reference.** They show what the released numbers were produced with. Their line numbers are cited in
  `configs/entries.py` and `configs/eval_entries.py`.
- **Equivalence checks.** `tests/test_entries_table.py` parses the live `DataLoader(...)`,
  scheduler, forward and validation calls of these trainers and compares them with
  `configs/entries.py` and `train.py`. Running a legacy script next to its new equivalent is the
  end-to-end check: the same numbers from a released checkpoint, and the same first training steps.

Do not edit these files. New work uses `train.py` and `evaluate.py` (see the [README](../README.md)).

## Running a legacy script

The scripts import `data`, `model`, `builder` and `tools` from the repository, and older models
write debug PNGs into the **current working directory**. Run them from a scratch directory,
with the repository on `PYTHONPATH`:

```bash
REPO=/path/to/CylinderSplat
RUN=/data/qiwei/cylindersplat_dev/runs/<run-id>       # a fresh directory under the dev root
mkdir -p $RUN/cwd && cd $RUN/cwd
PYTHONPATH=$REPO python $REPO/legacy/<script>.py --py-config $REPO/configs/OmniScene/<config>.py ...
```

- The legacy scripts have **no write guard**. Keep every path they write outside the protected trees
  (the author's `program/cylinderSplat`, any `workdirs` tree, the datasets `pano_grf`, `360Loc`, `Kansas`
  and `/data/dataset/VIGOR`, and `/data/qiwei/cylindersplat_repro`; the full list is in the README's write-guard
  section):
  - Trainers write into `--work-dir`. Trainers with a validation loop also write images to
    `cfg.output_dir/cfg.exp_name/validation/`. The configs set `output_dir =
    "/data/qiwei/nips25/workdirs"`, which is protected, so use an override config (below).
  - Evaluation scripts read `<--output-dir>/<--load-from>/model.safetensors` **and write** their log,
    dumped config and PNG/PLY files into `--output-dir`. Never pass the original work directory.
    Point `--output-dir` at the fresh run directory and link the checkpoint there.
- With this repository on `PYTHONPATH`, a legacy script runs the **current** model code. With every
  switch at its default, that code is equivalent to `f7b20b9`, but the four stage models no longer write
  debug PNGs. For the exact released code, use a checkout of `f7b20b9` (`main`) instead.
- `model/losses.py` now finds the LPIPS weights (`taming/modules/autoencoder/lpips/vgg.pth`) from the
  repository root, with no `taming` import, and raises `FileNotFoundError` if the file is missing. At
  `f7b20b9` it imported `taming.util` from the `taming-transformers` package (in that commit's
  requirements, not in the new ones) and read the weights relative to the working directory. So a
  checkout of `f7b20b9` needs `pip install taming-transformers`, and a link to `taming` in the scratch directory
  (`ln -s $REPO/taming $RUN/cwd/taming`), or LPIPS tries to download the weights. The ablation models
  (UniFuse / Depth Anywhere) still read `checkpoints/...` relative to the working directory in both versions.
- Most scripts set `CUDA_VISIBLE_DEVICES` when imported (column *GPU pin* below), which overrides
  the caller's choice. The new entry points never do this.
- Multi-process trainers need the 3-process launch: `PYTHONPATH=$REPO accelerate launch --config-file
  $REPO/configs/accelerate/accel_3proc.yaml $REPO/legacy/<trainer>.py ...`. The committed
  `accelerate_config.yaml` pins GPUs 0,2,3 and turns debug on.

Example: legacy evaluation of the released stage-3 checkpoint (REF-T1), which writes only under `$RUN`:

```bash
RUN=/data/qiwei/cylindersplat_dev/runs/legacy_eval_all256_48000
mkdir -p $RUN/cwd
ln -s /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/checkpoint-48000 $RUN/checkpoint-48000
cd $RUN/cwd
PYTHONPATH=$REPO python $REPO/legacy/evaluate_mp3d_double_256.py \
    --py-config $REPO/configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
    --output-dir $RUN --load-from checkpoint-48000        # pins GPU 1 at import
```

The `*_256` configs set `eval_args = dict(save_vis=True, save_ply=True)`, so this writes a
PNG and a PLY per sample to `$RUN/48000/`. Keep `save_vis`: the legacy script computes its
per-scene lines only in that branch.

### Override configs

A legacy trainer with a new equivalent takes the output root and the resume checkpoint from its
config. It ignores its own `--resume-from` flag. To change these values, write a small config that
inherits the original. Keep it out of the top level of `--work-dir` (for example in `$RUN/cfg/`),
because the trainer dumps its config to `<work-dir>/<same file name>`:

```python
# $RUN/cfg/pixel_256.py
_base_ = ['/path/to/CylinderSplat/configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py']
output_dir = '/data/qiwei/cylindersplat_dev/runs/legacy_mp3d_pixel_256'   # validation images go under it
# resume_from = '<previous stage>/checkpoint-36000/model.safetensors'     # a file, loaded by name+shape filter
```

## Old → new command map

General form:

```bash
# legacy trainer                                   # new
[accelerate launch ...] train_X.py \               accelerate launch --config-file configs/accelerate/accel_<N>proc.yaml \
    --py-config C --work-dir W                         train.py --entry <row> --py-config C --run-id <id> \
                                                       [--resume-from CKPT --transfer <name>]
# legacy evaluation                                # new
python evaluate_X.py --py-config C \               CUDA_VISIBLE_DEVICES=<gpu> python evaluate.py --dataset <row> \
    --output-dir D --load-from checkpoint-N            --py-config C --ckpt D/checkpoint-N --run-id <id>   # or --out-dir
```

### Training scripts

| Legacy script | Loader (`data/`) | GPU pin | New equivalent |
|---|---|---|---|
| `train_mp3d_cylinder_double_256.py` | `mp3d_dataloader_double_256` | – | `train.py --entry mp3d_double_256` (3 processes) |
| `train_mp3d_cylinder_single_256.py` | `mp3d_dataloader_single_256` | – | `train.py --entry mp3d_single_256` (3 processes) |
| `train_360Loc_cylinder_double_all_512.py` | `loc360_dataloader_double_all_512` (256×512) | – | `train.py --entry loc360_all_256` (3 processes) |
| `train_mp3d_cylinder_double_512.py` | `mp3d_dataloader_double_512` | – | `train.py --entry mp3d_double_512` (1 process) |
| `train_mp3d_cylinder_double.py` | `mp3d_dataloader_double` (160×320) | 0 | `train.py --entry mp3d_double_160` (1 process) |
| `train_vigor_cylinder_double.py` | `vigor_dataloader_double` (Kansas City) | 0 | `train.py --entry kansas_double_160` (1 process) |
| `train.py` | `dataloader` (nuScenes, Omni-Scene) | – | no new equivalent |
| `train_mp3d_cylinder.py` | `mp3d_dataloader` | 2 | no new equivalent |
| `train_mp3d_cylinder_double_random.py` | `mp3d_dataloader_double_random` | 0 | no new equivalent |
| `train_mp3d_cylinder_double_val.py` | `mp3d_dataloader_double` (validation loop only) | 2 | no new equivalent |
| `train_mp3d_cylinder_trible.py` | `mp3d_dataloader_trible` (three inputs) | 0 | no new equivalent |
| `train_360Loc.py` | `loc360_dataloader` | 0 | no new equivalent |
| `train_360Loc_cylinder_double.py` | `loc360_dataloader_double` | 0 | no new equivalent |
| `train_360Loc_cylinder_double_all.py` | `loc360_dataloader_double_all` | 3 | no new equivalent |
| `train_360Loc_cylinder_double_random.py` | `loc360_dataloader_double_random` | 3 | no new equivalent |
| `train_vigor.py` | `vigor_dataloader` | 3 | no new equivalent |

### Evaluation scripts

| Legacy script | Loader (`data/`) | GPU pin | New equivalent |
|---|---|---|---|
| `evaluate_mp3d_double_256.py` | `mp3d_dataloader_double_256` | 1 | `evaluate.py --dataset mp3d_double_256` |
| `evaluate_mp3d_single_256.py` | `mp3d_dataloader_single_256` | 1 | `evaluate.py --dataset mp3d_single_256` |
| `evaluate_mp3d_double_512.py` | `mp3d_dataloader_double_512` | 1 | `evaluate.py --dataset mp3d_double_512` |
| `evaluate_360Loc_double_256.py` | `loc360_dataloader_double_all_512` | 1 | `evaluate.py --dataset loc360_double_256` |
| `evaluate_VIGOR.py` | `vigor_dataloader_double` | 1 | `evaluate.py --dataset vigor_double` |
| `evaluate.py` | `dataloader` (nuScenes, Omni-Scene) | – | no new equivalent |
| `evaluate_mp3d.py` | `mp3d_dataloader` | 2 | no new equivalent |
| `evaluate_mp3d_double.py` | `mp3d_dataloader_double` (160×320) | 3 | no new equivalent |
| `evaluate_mp3d_trible.py` | `mp3d_dataloader_trible` | 1 | no new equivalent |
| `evaluate_360Loc.py` | `loc360_dataloader` | 0 | no new equivalent |
| `evaluate_360Loc_double.py` | `loc360_dataloader_double` | 3 | no new equivalent |

### Other files

| File | What it is |
|---|---|
| `run2.sh` | the author's command history (see below) |
| `.vscode/launch.json` | the author's VS Code debug launch entries |
| `demo.py`, `demo_vigor.py` | `forward_demo` runs (GPU pins 0 and 3), no new equivalent |
| `pano2point.py` | equirectangular pixel → 3D point helper |
| `ply2xml.py` | point cloud → XML scene file for figure rendering |
| `test_post.py` | plots a list of camera positions |

## run2.sh

`run2.sh` is a **command history, not a runnable script**. Executing it would start every
experiment in sequence. It also contains lines that cannot work as written:

- `evaluate_360Loc_double_all.py` (lines 208, 218, 223) never existed in the repository;
  `evaluate_360Loc_double_256.py` is the 360Loc evaluator.
- `omni_gs_160x320_VIGOR_cylinder_volume_decare.py` and `..._volume_spherical.py` (the commands at
  lines 242 and 246) are not in `configs/OmniScene/`, and both commands reuse the work directory of line 229.
- The UniFuse / Depth Anywhere ablation models (`*_unifuse.py`, `*_pixel_depthanywhere.py`
  configs) call `PixelGaussian` without its `extrinsics_in` argument and fail with a `TypeError`,
  at `f7b20b9` as well as now.
- Lines 47–64 start the `*_256` trainers with plain `python`. Those trainers call
  `my_model.module.forward`, which exists only under a multi-process (DDP) launch. Lines 67–88 are
  the `accelerate launch` forms that were used.

Every `--work-dir` and `--output-dir` in `run2.sh` points into a `workdirs` tree, which is protected. Replace
each `--work-dir` with a fresh directory under the dev root `/data/qiwei/cylindersplat_dev/runs/<run-id>`.
Read checkpoints in place with `evaluate.py --ckpt <old dir>/checkpoint-N`:

| `run2.sh` lines | Script, config | Original `--work-dir` | Run directory (under the dev root) | New command |
|---|---|---|---|---|
| 3, 6, 9 | `train_mp3d_cylinder_double_random.py`, pixel / volume / all (160) | `…/omni_gs_160x320_mp3d_cylinder_double_{pixel,volume,all}_random` | `runs/mp3d_double_{pixel,volume,all}_random` | none (legacy only) |
| 42, 103, 115 | `train_mp3d_cylinder_double.py`, pixel / volume / all (160) | `…/omni_gs_160x320_mp3d_cylinder_double_{pixel,volume,all}` | `runs/mp3d_double_{pixel,volume,all}_160` | `--entry mp3d_double_160` |
| 91, 95, 99, 107, 111 | `train_mp3d_cylinder_double.py`, volume_density / volume_unifuse / all_unifuse / pixel_depthanywhere / pixel_unifuse | `…/omni_gs_160x320_mp3d_cylinder_double_<variant>` | `runs/mp3d_double_<variant>_160` | `--entry mp3d_double_160` (the UniFuse / Depth Anywhere variants fail, see above) |
| 47, 72 | `train_mp3d_cylinder_double_256.py`, `pixel_256` | `…/omni_gs_160x320_mp3d_cylinder_double_pixel_256` | `runs/mp3d_pixel_256` | stage 1 below |
| 52, 67 | `train_mp3d_cylinder_double_256.py`, `volume_256` | `…/omni_gs_160x320_mp3d_cylinder_double_volume_256` | `runs/mp3d_volume_256` | stage 2 below |
| 57, 81 | `train_mp3d_cylinder_double_256.py`, `all_256` | `…/omni_gs_160x320_mp3d_cylinder_double_all_256` (line 81: `…_all_256_new`) | `runs/mp3d_all_256` | stage 3 below |
| 76 | `train_mp3d_cylinder_single_256.py`, `pixel_256_single` | `…/omni_gs_160x320_mp3d_cylinder_single_pixel_256` | `runs/mp3d_pixel_256_single` | single view below |
| 62, 86 | `train_360Loc_cylinder_double_all_512.py`, `360Loc_cylinder_all_256` | `…/omni_gs_160x320_360Loc_cylinder_double_all_256` | `runs/loc360_all_256` | 360Loc below |
| 119, 123 | `train_360Loc_cylinder_double_all.py`, `360Loc_cylinder_all` / `_pixel` | `…/omni_gs_160x320_360Loc_Cylinder_Double_{All2_depthanywhere,Pixel}` | `runs/loc360_double_{all,pixel}_160` | none (legacy only) |
| 229, 233, 238 | `train_vigor_cylinder_double.py`, VIGOR all / pixel / pixel_unifuse | `…/omni_gs_160x320_VIGOR_cylinder_double_{all,pixel,pixel_unifuse}` | `runs/kansas_double_{all,pixel,pixel_unifuse}_160` | `--entry kansas_double_160` |
| 242, 246 | `train_vigor_cylinder_double.py`, VIGOR volume_decare / volume_spherical | `…/omni_gs_160x320_VIGOR_cylinder_double_all` (same as line 229) | – | none: the configs do not exist |

The evaluation lines map row by row with the general form above. For example, lines 155–158 become
`evaluate.py --dataset mp3d_double_256 --py-config …/omni_gs_160x320_mp3d_cylinder_all_256.py
--ckpt /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/checkpoint-48000
--run-id eval_mp3d_all_256_48000`. Lines 140, 150 and 160 evaluate the two-view `pixel_256` /
`volume_256` / `all_256` configs with the single-view script. A model must be built with the view count it runs on,
so use a config with `num_frames=1` (such as `pixel_256_single`) for `--dataset mp3d_single_256`.

### The README training chain, mapped

In the new form (the [README](../README.md#training) commands), with `RUNS=/data/qiwei/cylindersplat_dev/runs`:

```bash
L3="accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py"
$L3 --entry mp3d_double_256 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py  --run-id mp3d_pixel_256
$L3 --entry mp3d_double_256 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py --run-id mp3d_volume_256 \
    --resume-from $RUNS/mp3d_pixel_256/checkpoint-36000 --transfer stage1_to_stage2
$L3 --entry mp3d_double_256 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py    --run-id mp3d_all_256 \
    --resume-from $RUNS/mp3d_volume_256/checkpoint-36000 --transfer stage2_to_stage3
$L3 --entry loc360_all_256  --py-config configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py  --run-id loc360_all_256 \
    --resume-from $RUNS/mp3d_all_256/checkpoint-36000 --transfer mp3d_all_256_to_loc360_pan2
$L3 --entry mp3d_single_256 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py --run-id mp3d_pixel_256_single \
    --resume-from $RUNS/mp3d_pixel_256/checkpoint-36000 --transfer double_pixel_to_single_pixel
```

In the legacy form (run2.sh lines 72, 67, 81, 86, 76), for equivalence runs. Each stage gets an
override config that sets `output_dir` to its run directory and `resume_from` to the previous stage's
`model.safetensors`:

```bash
REPO=/path/to/CylinderSplat; RUNS=/data/qiwei/cylindersplat_dev/runs
legacy_stage () {   # <run-id> <trainer> <config> [resume model.safetensors]
  local R=$RUNS/$1; mkdir -p $R/cwd $R/cfg
  { echo "_base_ = ['$REPO/configs/OmniScene/$3']"; echo "output_dir = '$R'"
    [ -n "$4" ] && echo "resume_from = '$4'"; } > $R/cfg/$3
  (cd $R/cwd && PYTHONPATH=$REPO accelerate launch --config-file $REPO/configs/accelerate/accel_3proc.yaml \
      $REPO/legacy/$2 --py-config $R/cfg/$3 --work-dir $R)
}
legacy_stage legacy_mp3d_pixel_256  train_mp3d_cylinder_double_256.py omni_gs_160x320_mp3d_cylinder_pixel_256.py
legacy_stage legacy_mp3d_volume_256 train_mp3d_cylinder_double_256.py omni_gs_160x320_mp3d_cylinder_volume_256.py \
    $RUNS/legacy_mp3d_pixel_256/checkpoint-36000/model.safetensors
legacy_stage legacy_mp3d_all_256    train_mp3d_cylinder_double_256.py omni_gs_160x320_mp3d_cylinder_all_256.py \
    $RUNS/legacy_mp3d_volume_256/checkpoint-36000/model.safetensors
legacy_stage legacy_loc360_all_256  train_360Loc_cylinder_double_all_512.py omni_gs_160x320_360Loc_cylinder_all_256.py \
    $RUNS/legacy_mp3d_all_256/checkpoint-36000/model.safetensors
legacy_stage legacy_mp3d_pixel_256_single train_mp3d_cylinder_single_256.py omni_gs_160x320_mp3d_cylinder_pixel_256_single.py \
    $RUNS/legacy_mp3d_pixel_256/checkpoint-36000/model.safetensors
```

The legacy trainers load `resume_from` with a silent name-and-shape filter. The new trainer
requires the named transfer to match exactly.

## How the legacy scripts behave

The README's "Paper vs code" section refers here. These behaviours are what the new entry points
reproduce by default. Where a switch changes one of them, it is named.

- **No gradient sync.** The `*_256` trainers call `my_model.module.forward` under a 3-process
  Accelerate launch. The DDP wrapper is bypassed, so the processes never average gradients. Each
  process trains on its own third of the (unshuffled) data, and rank 0 saves the checkpoint (switch `ddp_forward`).
  The 160 / 512 trainers run on one process.
- **Fixed data order.** The map-style loaders build `DataLoader(..., shuffle=False)` with per-stage generator
  seeds (train 1234, val 3456, test 2345) and 32 workers. The 360Loc loader uses 1 worker, and its
  `IterableDataset` shuffles the sequences itself in the train split (switch `shuffle_train`).
- **Silent resume.** `cfg.resume_from` is loaded by keeping every tensor whose name and shape match and
  silently dropping the rest. Only weights are loaded. The six trainers with a new equivalent ignore
  their `--resume-from` flag.
- **Validation.** Validation images go to `cfg.output_dir/cfg.exp_name/validation/` without `torch.no_grad()`.
  The 360Loc trainer has no validation loop.
- **Debug images.** At `f7b20b9` the models write `render_*.png` debug images into the working directory
  on every forward, and read the LPIPS weights relative to it.
- **Evaluation targets include the inputs.** The two-view MP3D scripts score targets [0, 1, 2] with inputs
  [0, 2], and the single-view script targets [0, 1, 2] with input [1]. 360Loc scores all four frames of a
  window with inputs at its ends.
- **Per-scene lines need `save_vis`.** Per-scene lines are computed only in the `save_vis` branch. The Total
  line of the `*_256` evaluators lacks the `wspsnr` label, so from `ssim` on each label shows the previous
  metric's value (WS-PSNR under `ssim`, and so on) and the `depthsim` value is dropped. The 512 script prints WS-PSNR as `psnr`, the 360Loc script as
  `ws_psnr`.
- **Unaligned depth metrics.** The `*_256` evaluators compute a per-view median scale alignment of the depth and then discard
  it. The reported AbsRel/RMSE/δ are unaligned.
- **Missing checkpoints.** Without `--load-from` (and no `load_from` in the config), an evaluator prints "Can't find
  checkpoint" and evaluates **randomly initialised weights**. An empty scene raises `ZeroDivisionError`. The 360Loc evaluator loads with `accelerator.load_state(...,
  strict=False)`, which also restores the RNG states saved with the checkpoint.

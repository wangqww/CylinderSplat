# Changelog: `dev` branch

Changes on top of the released code (`main`, commit `f7b20b9`). The rule for all of them is that,
**with every switch at its default, training and evaluation behave exactly as the released code**:
same numbers from the released checkpoints, same trainer behaviour per table-T row. Everything that
changes results is behind an opt-in switch (workstream D). The released scripts are kept
byte for byte in `legacy/` ([legacy/README.md](legacy/README.md)).

## A. Entry points, evaluation, safety

- **A0 `legacy/`.** The 32 root Python scripts, `run2.sh` and `.vscode/launch.json` moved into `legacy/`
  byte for byte, with an old → new command map in `legacy/README.md`. This gives one place for the
  released scripts, kept as the reference for equivalence checks.
- **A1 `train.py` + `configs/entries.py`.** One trainer driven by a table of six rows (`--entry`),
  one per legacy trainer: `mp3d_double_256`, `mp3d_single_256`, `loc360_all_256`,
  `mp3d_double_512`, `mp3d_double_160`, `kansas_double_160`. Each row copies its trainer's
  `DataLoader` keywords, scheduler, forward call, validation, setup order and process count.
  The trainer never reads `cfg.dataset_name`. Why: the six near-duplicate scripts drifted apart, and a row makes
  the differences explicit and testable.
- **A1 Phase-2 screen rows (`SCREEN_ENTRIES`).** Four more rows for the switch screen:
  `screen_mp3d_all_256`, `screen_mp3d_single_256`, `screen_loc360_all_256` (1 process each) and
  `screen_d1_pair_256` (2 processes). **They reproduce no legacy trainer.** Each reuses a table-T
  row's loader, batch size and setup order, runs OneCycleLR with `total_steps` = screen steps + 100,
  stops after `--screen-steps` steps (default 6000) and saves `checkpoint-<steps>`. `--seed`
  (default 42) replaces the seeds of `torch.manual_seed` and `set_seed`; the loader seeds stay.
  Both flags are refused by the table-T and stage-4 rows. `screen_mp3d_single_256` trains the new config
  `omni_gs_160x320_mp3d_cylinder_all_256_single.py` (the joint model built with `num_frames=1`) for
  the D3 arms. Why: the Phase-2 screen needs short, one-process runs that no legacy trainer provides.
- **A1 stage-4 rows (`STAGE4_ENTRIES`).** `mp3d_double_512_ddp3` and `mp3d_double_512_ddp4` (3 and 4
  processes) train stage 4 of the MP3D schedule: the stage-3 joint model (`all_256` architecture,
  config with `resolution = [512, 1024]`) at 512×1024. **They reproduce no legacy trainer.** Each takes
  the loader and batch sizes of `mp3d_double_512` and the scheduler (OneCycleLR over
  len(train loader) × `max_epochs` + 100 steps), setup order, `.module` forward and validation of
  `mp3d_double_256`. The new transfer `stage3_to_stage4_512` allows no missing or extra name: no
  parameter shape depends on the resolution. Why: the legacy 512 trainer trained the different
  `all_512` architecture on one process, so the stage-3 weights could not be continued at 512×1024.
- **A1 process count.** A row accepts only its own process count (3 for the table-T `*_256` rows, 1 for
  the other table-T rows, 1 or 2 for the screen rows, 3 or 4 for the stage-4 rows). Launch configs are in
  `configs/accelerate/accel_{1,2,3}proc.yaml`. Why: the `*_256` trainers' behaviour depends on it.
- **A1 resume rule (`tools/resume.py`, `--resume-from`, `--transfer`, `--no-resume`).** Resuming loads
  weights only, from a checkpoint directory or `model.safetensors`. It fails on a missing file, zero matches,
  or any missing or extra name, dtype or shape, except the frozen lists of a named transfer
  (`configs/entries.py` `TRANSFERS` = `tests/fixtures/allowed_keys.json`). Why: the legacy
  name-and-shape filter silently dropped whatever did not fit.
- **A1 extras.** `--max-steps` (short equivalence runs) and `--profile-steps` (C4). The switches
  are written to `switches.json` and the dumped config.
- **A2 `evaluate.py` + `configs/eval_entries.py`.** One evaluator with five rows, one per legacy
  evaluator: `mp3d_double_256`, `mp3d_single_256`, `mp3d_double_512`, `loc360_double_256`,
  `vigor_double`. Per-scene metrics are collected independently of `--save-vis`, with the legacy
  arithmetic. Why: the legacy scripts computed per-scene lines only while writing images.
- **A2 derived rows.** `mp3d_double_256_val` is `mp3d_double_256` on the MP3D validation split (model
  selection). `mp3d_double_512_full` and `mp3d_double_512_full_val` are `mp3d_double_256` and
  `mp3d_double_256_val` on the 512×1024 loader (stage 4), with `legacy_script` None: the legacy 512
  script computes a reduced metric set, which the `mp3d_double_512` row keeps reproducing. Why: stage 4
  needs the full metric set and a validation split at 512×1024.
- **A2 labels.** WS-PSNR is labelled `wspsnr` in every line (the 512 script printed `psnr`, the 360Loc
  script `ws_psnr`), and the MP3D-256 Total line gets the missing `wspsnr` placeholder. Why: the
  legacy Total line showed each value under the wrong label.
- **A2 flags.**
  - `--novel-only`: scores only the targets that are not inputs.
  - `--check-ref <name>` (REF-T1..T4): exact comparison with the frozen reference in `repro/`: every
    per-scene line, or for the totals-only REF-T3 the Total line without its wall-time prefix. It adds the
    reference's `provenance.allowed_extra` to `--allow-extra`: REF-T4-pixel allows `pixel_gs.mono_depth.*`,
    the 333 tensors its checkpoint holds that the pixel model does not build; the other references allow none
    (REF-T2's checkpoint holds no mono_depth tensors).
  - `--save-vis` / `--save-ply`: override `cfg.eval_args`; off by default. `--save-ply` is refused for
    `loc360_double_256` (per-view pixel Gaussians; legacy never wrote 360Loc PLYs).
  - `--align-depth`: the legacy median alignment as extra lines; the legacy code computed it and discarded it.
  - `--fast-ssim`: an extra GPU SSIM column.
  - `--strict-load` (default) / `--allow-partial`.
  - `--allow-extra PATTERN` (repeatable): skip checkpoint tensors the model does not build whose names
    match (an exact name or `prefix.*`; a bare `*` is refused). Missing names, shape mismatches and other
    extras still fail; the patterns are recorded in `metrics.json` under `checkpoint.allowed_extra`. Why:
    `--allow-partial` accepted any mismatch, and REF-T4-pixel could not pass the strict load.
  - `metrics.json`: a machine-readable record of the run.
- **A3 paths to config / CLI.** Every output goes under a run directory (`--work-dir` / `--out-dir` /
  `--run-id` under `$CYLINDERSPLAT_RUNS_ROOT`), and the checkpoint is read in place. The PanSplat backbone
  checkpoint of `OmniGaussianCylinderPixel` is now `model.backbone_ckpt`. It stays required, and a
  missing file is an error. The LPIPS weights are found from the repository root (`model/losses.py`).
- **LPIPS weights without taming-transformers.** `model/losses.py` no longer imports `taming.util`. A
  local `get_ckpt_path` returns the shipped `taming/modules/autoencoder/lpips/vgg.pth` and raises
  `FileNotFoundError` if it is missing, where taming's version would have downloaded it. Loading the
  shipped file is unchanged. Why: the new requirements do not list `taming-transformers`, and without it
  `import model` failed on a fresh install.
- **Write guard (`tools/write_guard.py`).** At start-up every write root, including symlink targets, is
  checked against the protected trees: the author's code, work and data trees (MP3D `pano_grf`,
  `360Loc`, `Kansas` and `/data/dataset/VIGOR`), any `workdirs` path, and `CYLINDERSPLAT_PROTECTED`.
  The process runs with its cwd in `<run>/cwd`, so relative writes stay in the run; a scratch dir that
  already contains a symlink is refused (the models write fixed file names there). `train.py` refuses a
  work dir that equals, contains or lies inside the resume checkpoint's directory, and `evaluate.py` the
  same for its out dir. At write time an existing output that is a symlink or a hard-linked file (log,
  dumped config, `switches.json`, `metrics.json`, PNG/PLY) is refused, so a reused run dir cannot write into
  another file. Why: the legacy scripts wrote next to checkpoints and into the cwd.
- **`repro/`.** `reference_metrics.json` freezes REF-T1..T4 of the released code (REF-T1, REF-T2, REF-T3,
  REF-T4-pixel, REF-T4-volume); see `repro/README.md` for the checkpoints and commands.

## B. Repository hygiene

- **B1.** Removed build artefacts and stray files: the stale `pano_gaussian` `_C*.so`, both
  `*.egg-info`, `simple-knn/build/`, `pano_gaussian/cuda_rasterizer/backward copy.cu`,
  `pano_gaussian/error_log.txt` and `._.DS_Store`. `.gitignore` now covers build output, `*.so`/`*.o`,
  `._*` and `runs/`. Why: the committed `.so` did not match the sources.
- **B2.** `test_post.py`, `pano2point.py` and `ply2xml.py` moved to `legacy/`, and the notebooks to `notebooks/`.
  `pretrained/dino_resnet50_pretrain.pth` is kept. Only the older non-cylinder configs reference it (the
  nuScenes, cube and plain `omni_gs_160x320*` Omni-Scene-style configs); no config of a `train.py` row does.
- **B3.** glm is a submodule (`pano_gaussian/third_party/glm`, g-truc/glm @ `8ebe4b5e`). The
  top-level gitlinks `mmcv`, `pytorch3d` and `diff-gaussian-rasterization` are removed. mmcv 2.1.0 and
  pytorch3d 0.7.8 are installed as documented in the README. Why: the extension did not build from a
  fresh clone (glm was empty), and the three gitlinks had no `.gitmodules` entry, so they could not
  be checked out.
- **B4.** `requirements.txt` is pinned from the environment that produced the released results.
  `docker/Dockerfile` and `docker/pip_requirements.txt` (identical to `requirements.txt`) are aligned
  with it. The source builds (mmcv, pytorch3d, `pano_gaussian`, `simple-knn`) run with
  `--no-build-isolation` after a build-tools step (`setuptools<70`, `wheel`, `ninja`), and the file's
  `--find-links` line gets the torch-scatter wheel from the PyG index. `taming-transformers` is not a
  requirement (see "LPIPS weights without taming-transformers" in A). UniK3D is not part of this environment: it needs Python ≥ 3.11 and torch ≥ 2.4, so
  the README installs it in a separate one. Why: under build isolation (uv, or pip ≥ 23.1 without
  `wheel`) the builds cannot see torch, and installing UniK3D here would upgrade torch and numpy.
- **B5 lazy imports.**
  - The nuScenes devkit (`data/dataloader.py`, `data/transforms/loading.py`) and open3d
    (`vis_feat.save_point_cloud`) are imported only where they are used.
  - `diff_gaussian_rasterization` / `diff_surfel_rasterization` in `model/twodgaussian.py` are imported on first use.
  - `torch_scatter` in `OmniGaussianCylinderAll` is imported inside the function that uses it.
  - The unused `simple_knn` import is removed from `volume_gs_decoder_conf.py`.
  - The four stage models no longer import `vis_feat`.
  - `pano_gaussian` stays the eager runtime rasteriser.

  Why: `import model` should not need packages that the released models never use.
- **B6 `tools/prepare_unik3d_depth.py`.** A port of the author's UniK3D generators with unchanged
  inference settings. It writes a mirrored tree under `--out-root`, and refuses the dataset root and protected
  trees. Before UniK3D is loaded, every output file is resolved through symlinks and checked against the
  protected trees, and any output path that goes through a symlink is refused, so a dataset "copy" made of
  links cannot write into the original. Each file is written to a new temp file in its directory and then
  swapped in with `os.replace`, so an existing output that is a hard link to an original is replaced by a new
  file and the original keeps its contents. It lists the same views as the loaders (`.DS_Store` entries are
  skipped). It runs in its own UniK3D environment (README "Depth prior"). Why: the depth prior's
  generator was not in the repository.
- **B7.** `README.md` rewritten (install, data, depth prior, four-stage schedule, evaluation protocol,
  switches, run directories, paper vs code), plus `legacy/README.md` and this file.

## C. Speedups that keep the outputs

- **C1 removed work that nothing reads.** Removed renders and debug PNG writes whose results no loss
  (and, in `val`, no validation image) reads. Why: every step did this work for nothing and wrote files
  into the cwd.
  - `OmniGaussianCylinderAll`: the pixel-only render (train only; kept in `val` for the validation images),
    the BEV render and five PNG writes.
  - `...360LocPan2`: the pixel / volume renders and blends that no loss reads. The volume render
    is kept if any volume loss weight is non-zero, and everything is kept in `val`. The BEV render and PNGs are removed.
  - `OmniGaussianCylinderVolume`: the pixel render, the BEV render and the PNG block.
  - `OmniGaussianCylinderPixel`: the PNG writes and the BEV render that only fed them.
  - The debug PNGs of `forward_test` (All, Volume).
  - The ablation classes (UniFuse, Depth Anywhere, Cartesian, spherical) are untouched.
- **C2 evaluation.** WS-PSNR weights are cached per (device, H, W, dtype) instead of being rebuilt per batch.
  SSIM stays skimage on CPU, and `--fast-ssim` is an optional extra column.
- **C3 loaders.** `num_workers` comes from the row (32, or 1 for 360Loc, as the loaders
  hard-code it). The 360Loc config's `num_workers` field was changed from 32 to 1 to match.
- **C4 timing.** `--profile-steps N` times N train steps with `torch.cuda.synchronize` and writes
  `profile_steps_rank<r>.json`.

## D. Opt-in switches (default = released behaviour)

The switches live in `tools/switches.py`. Set them with `switches = dict(...)` in a config or `--switch name=value`.
Unknown names, and model switches on a class that does not declare them, are refused.

- **D1 `ddp_forward`.** Forwards through the DDP wrapper, so the `*_256` rows all-reduce gradients (with
  `find_unused_parameters=True`). Every rank also waits at a barrier after the main process's save/validation
  of each step, because the next DDP forward is a collective (without it the other ranks would block in NCCL
  during validation and could hit the 1800 s timeout). Why: the legacy `.module.forward` trains three
  unsynchronised replicas.
- **D1b `shuffle_train`.** A new seeded order every epoch for map-style loaders. It is refused for 360Loc. Why:
  the loaders never shuffle.
- **D2a `lpips_eval`.** The LPIPS loss network stays in `eval()` (All, Pan2, Volume, Pixel). Why: `train()`
  re-enabled its dropout.
- **D2b `freeze_frozen_bn`.** The frozen backbone / pixel branch (and neck) of the stage-2 model stay in
  `eval()`. Why: their BatchNorm statistics drifted while "frozen".
- **D3 `v1_identity_pose`.** Single-view camera metas use the identity relative pose (All, Pan2).
  `PixelGaussian360Loc` gets a `num_frames` argument (default 2) and a v=1 cost-volume fallback copied
  from `PixelGaussian`. The new config `omni_gs_160x320_mp3d_cylinder_all_256_single.py` builds the joint
  model with `num_frames=1` for the MP3D D3 arms (row `screen_mp3d_single_256`, `--transfer d3_single_view`).
  Why: the V=1 path used the absolute `w2i`.
- **D4 `rotate_gaussians_to_world`.** The Gaussian quaternions (wxyz) are composed with the camera-to-world
  rotation once per Gaussian set (`model/utils/quaternion.compose_quaternion_c2w`), in the three pixel
  heads and the volume tails of All / Pan2 / Volume. Why: rotations stayed in the camera frame while
  positions moved to the world.
- **D5 `theta_periodic`.** θ is periodic in the cylindrical encoder and decoder: circular halo and wrap in
  plane attention, seam wrap in colour and depth sampling, rz pillars over [0, 2π)
  (`model/volume/theta_periodic.py`). Why: the panorama seam was a hard border.
- **D6 `cell_center_anchor`.** Anchors sit at cell centres with ±½-cell offsets and r ≥ 0. Why: lower-corner
  anchors with ±1-cell offsets let Gaussians leave their cell and r go negative.
- **D7 `rgb_retrieval="visibility_softmax"`.** A per-view encoder weighted by a softmax over visibility.
  The head itself accepts any number of views, but the model still runs at most 6, because
  `TPVFormerEncoderCylinder.cams_embeds` has `num_cams=6` rows (raising it changes that parameter's shape
  and breaks checkpoint resume). It adds 7 parameters `volume_gs.gs_decoder.gaussian_to_color_vis.*`,
  initialised from the concat head, and is resumed with `--transfer d7_visibility_softmax`. Why: the
  paper's Eq. 6 describes this weighting, and the concat head zero-pads a fixed 6-slot input.
- **D8 `prune_opacity=τ`.** Drops Gaussians with opacity < τ before the panorama rasterisation. Why: speed.
- **D9 `pixel_depth_sampling="nearest"`.** Nearest instead of bilinear reads of the depth prior in the
  pixel heads. Why: bilinear reads blend depths across object edges.
- **D10 `loc360_interleave`** (360Loc rows only). The train split yields one globally shuffled stream of
  (sequence, sample) pairs; each sequence still gives `times_per_scene` samples per epoch, so the epoch
  length and the OneCycle schedule are unchanged. Frames are decoded on first use and cached as uint8
  (bit-identical to `to_tensor`), so the persistent worker decodes each frame once. Why: the released
  stream yields all 1000 samples of one sequence before the next (~167 consecutive steps at 3 × batch 2),
  so the BatchNorm running statistics saved in a checkpoint come almost entirely from one sequence.
- **D11 `depth_valid_mask`** (Pan2). The depth losses skip pixels whose UniK3D prior depth lies outside
  the loader's (0.45, 50) m range. Why: the loader computes that mask but never returns it, so zero and
  out-of-range prior depths (about 6 % of the training pixels are exactly 0) were supervised.
- **`train.py --save-final`.** After the last step the main process saves the final weights as
  `checkpoint-<steps>` (screen rows always did). Why: the periodic saves stop up to `save_freq` steps
  before the end, and the 360Loc protocol (PanSplat's) tests the final weights.
- **Dev-release recipes.** `configs/OmniScene/release/` holds the configs that trained the published
  `dev` checkpoints (stage 1 and 2 at batch 4 per GPU and lr 4e-4, stage 3, the 360Loc fine-tune at lr 2e-4,
  10 epochs, seed 1111), each a `_base_` override of a released config; `omni_gs_160x320_mp3d_cylinder_all_512x1024.py`
  is the stage-4 config (all_256 at 512×1024, batch 1, lr 2e-4, 10 epochs). README: *Reproducing the dev release*.
- **Eval row `loc360_double_256_da`.** `loc360_double_256` with the Depth Anywhere pseudo-GT
  (`depthanywhere/*_depth_anywhere.png`, read like the 160×320 loader; `data/loc360_dataloader_da.py`) as
  the PCC reference, as in the paper. Why: the released 360Loc loader compares the rendered depth with the UniK3D prior that the model
  receives as input (PCC 0.99 for every checkpoint). Only PCC changes.

## Behaviour-preserving guarantees and their tests

Tests run on CPU from the repository root (`CUDA_VISIBLE_DEVICES= python -m pytest tests/`, env
`omniscene`). Tests marked `gpu` repeat a check through a CUDA op.

| Guarantee | Test |
|---|---|
| `legacy/` equals `f7b20b9` byte for byte | `cmp` against `git show f7b20b9:<file>` (the 32 scripts, `run2.sh` and `.vscode/launch.json`) |
| Table T equals the live legacy code: every `DataLoader(...)` keyword, the per-stage seeds, the scheduler calls, the forward and validation calls, the setup order; `train.build_dataloader` equals each legacy `load_*()` factory at run time | `tests/test_entries_table.py` |
| Screen rows equal their base table-T row except for the screen recipe (process count, forward, OneCycle length, seed); `--screen-steps` / `--seed` are refused on table-T rows | `tests/test_entries_table.py` |
| Resume rule: missing file, zero matches, extra / missing names, dtype and shape errors, tied weights; equals the legacy filter when it passes; transfer lists equal the frozen fixture | `tests/test_resume_rule.py`, `tests/fixtures/allowed_keys.json` |
| Every write root of a run (including symlinks) is refused inside protected trees; relative writes land in `<run>/cwd` | `tests/test_write_guard.py` |
| `evaluate.py` aggregation equals the verbatim legacy loops bitwise (lines, per-scene values, totals; WS-PSNR cache included) | `tests/test_eval_aggregation.py` |
| `repro/reference_metrics.json` holds REF-T1..T4 exactly as captured | `tests/test_reference_golden.py` |
| Stage models with switches off equal frozen `f7b20b9` copies: modules, parameters, train/eval state, camera metas (v=1, v=2, non-identity poses), losses, gradients, outputs, RNG | `tests/test_switches_identity_models.py` (copies in `tests/legacy_ref/models_ref.py`) |
| Pixel heads with switches off equal the frozen copies (edited statements and whole forward) | `tests/test_switches_identity_pixel.py` (`tests/legacy_ref/pixel_ref.py`) |
| Volume encoder / decoder / attention with D5–D7 off equal the frozen copies (construction RNG, outputs, gradients) | `tests/test_switches_identity_volume.py` (`tests/legacy_ref/volume_ref.py`; one `gpu` test) |
| C1: kept renders get bitwise the same Gaussians, loss and RNG are unchanged, validation images are byte-identical, no PNG in the cwd | `tests/test_render_removal.py` |
| D8 at τ = 0 hands the rasteriser exactly the legacy inputs; B5 lazy imports | `tests/test_render_prune.py` |
| Switches on do what they claim: D4 (identity / 90° cases, scales untouched), D9, D5–D7 (incl. the D7 init equivalence) | `tests/test_d4_composition.py`, `tests/test_pixel_switches_on.py`, `tests/test_volume_switches_on.py` |
| The release recipes differ from their base configs only in the documented values | `tests/test_release_configs.py` |
| D10 yields every (sequence, sample) pair once per epoch, mixed, each sample bit-identical to the released stream's; the DA loader changes only `outputs['depth']`; D11 bounds | `tests/test_loc360_loader.py` |

Status: the whole CPU suite passes in env `omniscene` on the GPU server (the tests marked `gpu` are
skipped without CUDA). Still to run on GPUs:

- REF-T1..T4 through `evaluate.py --check-ref` (exact at printed precision);
- per-row trainer equivalence at the row's process count (step-0 loss bitwise, steps 1–5 within 1e-4);
- `pano_gaussian` build parity;
- a 5-step smoke run per switch.

## Changes you will notice

**Training (`train.py`)**
- Validation images go to `<run>/validation/step-N/batch-M` under `torch.no_grad()`. The legacy trainers wrote
  them to `cfg.output_dir/cfg.exp_name/validation`, the author's work directory.
- Resuming is strict. Stage 2 needs `--transfer stage1_to_stage2`, and single view needs
  `--transfer double_pixel_to_single_pixel`. The configs' `resume_from` point to the author's checkpoints, so pass
  `--resume-from`. The `all_512` config's `resume_from` (a 160×320 checkpoint) is refused by design: use
  `--no-resume`. Stage 4 does not use `all_512`: it is `all_256` at 512×1024 on the rows
  `mp3d_double_512_ddp3/4` with `--transfer stage3_to_stage4_512`.
- A row refuses any other process count. The launch configs pin GPUs 1–3 of the author's machine.

**Models**
- The four stage models no longer write debug PNGs (`render_*.png`) into the cwd, in `forward` or
  `forward_test`. The ablation classes still do. In `train`, All returns `None` in the pixel-render slot of its
  output tuple, and Pan2 in its pixel slots (and in the volume slot when all volume weights are 0).
- `OmniGaussianCylinderPixel` refuses to build if `model.backbone_ckpt` does not exist. This also applies to evaluating a pixel
  model.

**Evaluation (`evaluate.py`)**
- `--ckpt` (required) replaces `--output-dir` + `--load-from`, and output goes to a separate run directory.
  Loading is strict, and a missing checkpoint exits non-zero. The legacy scripts evaluated random weights
  when no checkpoint was given. Extra checkpoint tensors are accepted only when named with `--allow-extra`
  (or by the `allowed_extra` of a `--check-ref` reference).
- `evaluate.py` runs on one process. The GPU comes from `CUDA_VISIBLE_DEVICES`, where the legacy scripts pinned one when imported.
  An empty scene logs `no samples` instead of raising `ZeroDivisionError`.
- `--save-vis` / `--save-ply` are off unless given. The `*_256` configs had both on.
- The labels are corrected (`wspsnr` everywhere, MP3D-256 Total line). The VIGOR Total line is unchanged
  and still omits `wspsnr`, which is in `metrics.json`.
- The `loc360_double_256` row keeps `accelerator.load_state` (it restores the saved RNG states, as
  legacy did), so it needs a checkpoint directory.
- `evaluate.py` ignores the training-only switches (`ddp_forward`, `shuffle_train`, `lpips_eval`,
  `freeze_frozen_bn`) and logs that it did.

**Switches**
- D1b uses a seeded `RandomSampler` that Accelerate shards and keeps in sync across processes, not a
  `DistributedSampler`, which would be sharded twice under Accelerate.
- D1 also turns on `find_unused_parameters=True`.
- D5 also wraps the decoder's depth-prior read and adds a one-cell circular halo to the value maps. It
  keeps the legacy half-pixel convention of the colour sampling (divide by w−1, `align_corners=False`),
  so the two sides of the seam still sit about one pixel apart.
- D7 adds a learnable temperature `gaussian_to_color_vis.log_beta` (init 0) besides the three layers.
  It keeps the old `gaussian_to_color` head frozen and unused, for checkpoint compatibility. The new weights are
  filled by a `load_state_dict` pre-hook.
- D8 prunes only the panorama render (not the BEV debug render), and τ must be in [0, 1) (checked when
  the switch is parsed).
- D4 with `PixelGaussian512` needs `forward(..., extrinsics_in=...)` by keyword. Both call sites pass it.
- At v=1, `PixelGaussian360Loc` uses `PixelGaussian`'s single-view fallback, the feature
  self-correlation. Legacy raised at `torch.stack([])` there. The head must be built with `num_frames=1`;
  with `num_frames=2` a v=1 forward raises `ValueError`. The block runs only at v=1, so v ≥ 2 is unchanged
  from `f7b20b9` (`tests/test_switches_identity_pixel.py`).

## Known gaps

- Stage 4 (joint 512×1024 from the stage-3 weights) keeps the `all_256` architecture and changes only the
  resolution (rows `mp3d_double_512_ddp3/4`, transfer `stage3_to_stage4_512`). The repository's `all_512`
  config builds a different architecture (359 missing names, 320 unused names, 26 shape mismatches
  against an `all_256` checkpoint); no transfer into it is defined. No stage-4 config ships in
  `configs/` yet, and no stage-4 result has been reproduced.
- The Pan2 single-view (v=1) arm of D3 on 360Loc has no screen row: it needs a single-view 360Loc
  loader, which does not exist yet. The head itself runs at v=1 (see Switches above).
- The UniFuse / Depth Anywhere ablation models call `PixelGaussian` without `extrinsics_in` and fail,
  as at `f7b20b9`.

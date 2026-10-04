# repro/

Reference numbers of the released code (commit f7b20b9) that the `dev` code must reproduce.

- `reference_metrics.json`: the frozen reference evaluations. Each reference holds the lines exactly as
  the legacy script printed them, every column parsed, and its provenance (legacy script, config,
  checkpoint, protocol, log).
- `sweep_table.md`: the checkpoint sweep behind the choice of REF-T1..T4. It lists every evaluated
  checkpoint of the saved runs, with WS-PSNR / SSIM / LPIPS next to the paper's Table 1 / 2 / 3 rows, and
  states the findings, including which paper numbers no saved checkpoint reproduces.

| reference | evaluate.py `--dataset` | config (`configs/OmniScene/`) | checkpoint | compared |
|---|---|---|---|---|
| REF-T1 | `mp3d_double_256` | `omni_gs_160x320_mp3d_cylinder_all_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/checkpoint-48000` | 7 per-scene lines |
| REF-T2 | `mp3d_single_256` | `omni_gs_160x320_mp3d_cylinder_pixel_256_single.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_single_pixel_256/checkpoint-33000` | 7 per-scene lines |
| REF-T3 | `loc360_double_256` | `omni_gs_160x320_360Loc_cylinder_all_256.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256/checkpoint-24000` | the Total line |
| REF-T4-pixel | `mp3d_double_256` | `omni_gs_160x320_mp3d_cylinder_pixel_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256/checkpoint-36000` | 7 per-scene lines |
| REF-T4-volume | `mp3d_double_256` | `omni_gs_160x320_mp3d_cylinder_volume_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256/checkpoint-36000` | 7 per-scene lines |

REF-T1 is the checkpoint closest to paper Table 1. REF-T2 is the single-view checkpoint closest to
Table 2, although no saved checkpoint reproduces Table 2. REF-T3 is the best 360Loc checkpoint. REF-T4
holds the stage-1 (pixel) and stage-2 (volume) checkpoints that the next stage's config resumes from.
The MP3D references score all three frames (context [0, 2], or [1] for single view). `wspsnr` is the
paper's PSNR column.

## Re-running a reference with evaluate.py

Run from the repository root in env `omniscene`, one process, on one free GPU among 1-3. The checkpoint
is read in place and never written. The output goes under `/data/qiwei/cylindersplat_repro2`, never next
to the checkpoint: an `--out-dir` inside the checkpoint directory, or one that contains it (its run
directory or any ancestor), is refused before anything is written. Keep the defaults: pass no `--switch`
and no `--novel-only`. The flags `--save-vis`, `--save-ply`, `--align-depth` and `--fast-ssim` only write
files or add extra lines, and they leave the compared lines unchanged. `--save-ply` is refused for
`loc360_double_256` (REF-T3): that model returns per-view pixel Gaussians, not one set per sample, and
the legacy script wrote no PLY files.

```bash
# REF-T1: stage 3 (joint), two views
CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset mp3d_double_256 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
    --ckpt /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/checkpoint-48000 \
    --out-dir /data/qiwei/cylindersplat_repro2/check_ref_t1 \
    --check-ref REF-T1

# REF-T2: single view (the config builds the model with pixel_gs.num_frames=1)
CUDA_VISIBLE_DEVICES=2 python evaluate.py --dataset mp3d_single_256 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py \
    --ckpt /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_single_pixel_256/checkpoint-33000 \
    --out-dir /data/qiwei/cylindersplat_repro2/check_ref_t2 \
    --check-ref REF-T2

# REF-T3: 360Loc, 'val' split, Total line only (loaded with accelerator.load_state, as the legacy script did)
CUDA_VISIBLE_DEVICES=3 python evaluate.py --dataset loc360_double_256 \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py \
    --ckpt /data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256/checkpoint-24000 \
    --out-dir /data/qiwei/cylindersplat_repro2/check_ref_t3 \
    --check-ref REF-T3

# REF-T4-pixel: stage 1 (pixel branch); --check-ref skips the reference's allowed extra
# pixel_gs.mono_depth.* tensors (333 in this checkpoint, not built by the model)
CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset mp3d_double_256 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py \
    --ckpt /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256/checkpoint-36000 \
    --out-dir /data/qiwei/cylindersplat_repro2/check_ref_t4_pixel \
    --check-ref REF-T4-pixel

# REF-T4-volume: stage 2 (volume branch)
CUDA_VISIBLE_DEVICES=2 python evaluate.py --dataset mp3d_double_256 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py \
    --ckpt /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256/checkpoint-36000 \
    --out-dir /data/qiwei/cylindersplat_repro2/check_ref_t4_volume \
    --check-ref REF-T4-volume
```

What `--check-ref NAME` does:

1. Before anything else runs, it checks that `--dataset`, `--novel-only`, the config file name and the
   checkpoint iteration (`checkpoint-N`) match the reference's provenance. On any difference it prints a
   warning for each one and exits non-zero, so a wrong pairing cannot pass. A reference name that is not
   frozen in `reference_metrics.json` exits the same way.
   It also adds the reference's `provenance.allowed_extra` patterns to `--allow-extra` (below), so the
   commands above need no load flag. REF-T4-pixel lists `pixel_gs.mono_depth.*`: its checkpoint holds 333
   such tensors that the pixel model does not build, and the legacy script dropped them silently. The
   other references list none (REF-T2's checkpoint holds no mono_depth tensors: 580 tensors, header read
   on 4090 2026-09-30).
2. After the evaluation:
   - Per-scene references (REF-T1, REF-T2, REF-T4-*): each scene's line, stripped, must equal the frozen
     `printed` string character for character.
   - Totals-only reference (REF-T3): the Total line after its `Finish evluation (N s). ` prefix must
     equal the frozen `evaluate_py_line`. That string is the legacy line (`printed`) with its one label
     `ws_psnr` printed as `wspsnr`, which is how evaluate.py prints the 360Loc Total line
     (`configs/eval_entries.py`).
3. It logs `match` or `MISMATCH` (expected vs got) per line, stores the result in
   `<out-dir>/metrics.json` under `reference_check`, and exits non-zero on any mismatch.

There is no tolerance: comparisons are made at printed precision. If a last-digit mismatch appears,
re-run the legacy script on the same GPU to separate nondeterminism from a code change, and report the
result (plan V2). The sweep found evaluation deterministic: REF-T1's checkpoint gave identical lines in
every run.

Notes:

- `OmniGaussianCylinderPixel` (REF-T2 and REF-T4-pixel) loads the PanSplat backbone checkpoint named by
  `model.backbone_ckpt` when the model is built. The trained weights then overwrite it, but the file must
  exist.
- The load is strict by default. If it reports skipped or uncovered tensors, inspect the names first.
  `--allow-extra PATTERN` (repeatable; an exact name or `prefix.*`) skips the checkpoint tensors the model
  does not build whose names match. Missing names, other shapes and every other extra name still exit.
  The patterns are recorded in `metrics.json` under `checkpoint.allowed_extra`, and the skipped count
  under `checkpoint.load.unused_allowed`. `--allow-partial` gives the legacy scripts' filtered load.

## How the references were captured

Each legacy script (`legacy/evaluate_*.py`, byte-identical to f7b20b9) ran on 4090 unchanged except for
its hard-coded `CUDA_VISIBLE_DEVICES` line. It used a copy of the repo config with `save_ply=False`,
which does not change the metrics. REF-T1 was captured on 2026-09-29 and matches the author's four
historical logs of that checkpoint. REF-T2..T4 come from the checkpoint sweep of 2026-09-29/30. The
exact log file of every reference is in its `provenance.source`.

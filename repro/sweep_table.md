# Checkpoint sweep of the released code (2026-09-29/30)

Every saved checkpoint of the runs below was evaluated with the legacy evaluation scripts of the released code
(commit `f7b20b9`) to choose the reference evaluations REF-T1..T4 in `reference_metrics.json` and to compare the
saved models with the paper. REF-T1..T4 are marked in bold in the tables.

## How the numbers were made

- Host `4090`, env `omniscene` (torch 2.1.0+cu118, mmcv 2.1.0, accelerate 1.5.2), one GPU per job.
- Each legacy script (`legacy/evaluate_*.py`) ran unchanged except for its hard-coded `CUDA_VISIBLE_DEVICES`
  line. The config is a copy of the repo config with `save_ply=False`, which does not change the metrics
  (`save_vis=True` as in the repo; the MP3D scripts compute the per-scene lines inside that branch).
- Evaluation protocol of the code: MP3D two-view scripts use context views [0, 2] and score all three frames
  [0, 1, 2], the two inputs included. The single-view script uses context [1] and scores all three. 360Loc uses
  context [l, l+3] and scores the four frames [l .. l+3].
- Run tags: `H` = `/home/qiwei/nips25/workdirs`, `D` = `/data/qiwei/nips25/workdirs`. These are two different run trees whose run
  directories have the same names.
- Sources: `.claude-implement/work/sweep/sweep_results.json` (per-scene values parsed from the logs; the 360Loc
  jobs appear there as `failed` only because they print no per-scene lines) and `loc360_results.json` (the 360Loc
  Total lines), both in the project directory outside the repo. The logs are under
  `/data/qiwei/cylindersplat_repro/logs` and `/data/qiwei/cylindersplat_repro2/logs` on 4090.
- Columns (scene key -> paper column, baseline = 2 x distance): `m3d_1.0` = M3D 2.0 m, `m3d_0.75` = M3D 1.5 m,
  `m3d_0.5` = M3D 1.0 m, `replica_0.5` = Replica 1.0 m, `residential_0.15` = Residential ~0.3 m. Each cell is
  **WS-PSNR / SSIM / LPIPS** at the printed precision. The paper rows use the paper's precision.
- "mean abs. WS-PSNR diff." is the mean over the five columns of |WS-PSNR - paper WS-PSNR|. It is measured against
  Table 1 for two-view runs and Table 2 for single-view runs, and it is the "closest to the paper" measure used
  below.

## Findings

1. **REF-T1 (T1_H_all_256 checkpoint-48000) is the checkpoint closest to paper Table 1**, at a mean abs. WS-PSNR
   diff. of 0.097 dB. The next closest are T1_H 51000 (0.132), 57000 (0.138), 66000 (0.141) and T1_D 36000
   (0.150). Against Table 1, M3D 1.5 m (25.912 vs 25.91) and Replica (30.290 vs 30.29) match. The other WS-PSNR
   gaps are M3D 2.0 m 23.653 vs 23.76 (-0.107), Residential 28.172 vs 28.25 (-0.078) and M3D 1.0 m 28.590 vs 28.89
   (-0.300, with SSIM 0.9195 vs .937). M3D 1.0 m is the largest gap. SSIM and LPIPS are within 0.005 and 0.008
   elsewhere, except Residential SSIM, which is 0.8668 against the paper's .817 (higher).
2. **No saved checkpoint reproduces Table 2 (single view).** REF-T2 (T2_D_single_pixel_256 checkpoint-33000) is
   the closest, at 0.554 dB. Its WS-PSNR is within 0.23 dB at M3D, but Replica is +0.61 and Residential +1.72. At
   all three M3D distances, every checkpoint of that run is below the paper on SSIM (best 0.7646 / 0.8057 / 0.8594
   against .822 / .854 / .915) and above it on LPIPS (best 0.2297 / 0.1808 / 0.1162 against .175 / .136 / .089).
   The paper (App. B) says the single-view model is a joint-stage fine-tune of the two-view model, and no such run
   was saved: the saved single-view run is pixel-only, and its config resumes from the stage-1 pixel checkpoint.
   The 2-view joint model fed one view (T2_H_all_256_v1probe) gives 13.9-15.1 dB at M3D. That is consistent with
   the absolute-pose handling of V=1 in `OmniGaussianCylinderAll` (switch D3) and is not a Table-2 model.
3. **360Loc (Table 3):** the best checkpoint is REF-T3 (T3_D_360Loc_all_256 checkpoint-24000), at WS-PSNR 28.205
   against 28.35, SSIM 0.8708 against .896 and LPIPS 0.1134 against .095. The script's `pcc` (0.9897) is computed
   against the loader's depth-prior maps. The paper's PCC (.884) uses another reference depth, so the two numbers
   are not comparable. The 360Loc config resumes from checkpoint-36000 of the H joint run, not from REF-T1's
   checkpoint-48000.
4. **Two runs are not the paper models.** `D/omni_gs_160x320_mp3d_cylinder_double_all_256_new` scores 9.5-14.5
   dB at M3D from checkpoint-6000 on (6-10 dB on Replica). `D/omni_gs_160x320_mp3d_cylinder_double_pixel_256`
   scores 11.9-12.9 dB at M3D (7.5-8.1 dB on Replica). Those numbers come from evaluating with the current
   configs, so the two runs were presumably trained with a different configuration. Their training configs were
   not checked. Neither run is used here. `D/omni_gs_160x320_mp3d_cylinder_double_volume_256` is also 2.5-6.3 dB
   below its H counterpart at M3D, at the same checkpoints.
5. **REF-T4** holds the stage checkpoints that feed the next stage, according to the configs' `resume_from`:
   `volume_256` resumes from `H/.../double_pixel_256/checkpoint-36000` (REF-T4-pixel), and `all_256` resumes from
   `H/.../double_volume_256/checkpoint-36000` (REF-T4-volume). For context, paper Table 4 (M3D 2.0 m) lists "only
   pixel" at 23.21 / .817 / .179 and "only cylindrical volume" at 22.17 / .782 / .210. At that distance
   REF-T4-pixel gives 23.224 / 0.8309 / 0.1852 and REF-T4-volume gives 21.428 / 0.7042 / 0.3401. The paper does
   not say that these rows come from the stage checkpoints.
6. **Evaluation is deterministic.** The sweep evaluated REF-T1's checkpoint again (T1_H_all_256 checkpoint-48000)
   and got all seven per-scene lines identical, in every column, to the REF-T1 capture made earlier on
   2026-09-29. That capture already matched the author's four historical logs of this checkpoint.
7. The first sweep ran the single-view jobs with the two-view configs, and those 17 jobs failed: T2_D with
   `pixel_256.py` and T2_H_all_256_as_single. The T2_D jobs were re-run with `pixel_256_single.py` in
   `/data/qiwei/cylindersplat_repro2`. Those are the T2_D numbers here.

## References frozen from this sweep

| reference | run tag | checkpoint | evaluate.py row | why |
|---|---|---|---|---|
| REF-T1 | T1_H_all_256 | 48000 | `mp3d_double_256` | closest to Table 1 (captured before the sweep, confirmed by it) |
| REF-T2 | T2_D_single_pixel_256 | 33000 | `mp3d_single_256` | closest to Table 2 (Table 2 itself is not reproduced) |
| REF-T3 | T3_D_360Loc_all_256 | 24000 | `loc360_double_256` | best 360Loc WS-PSNR; Total line only |
| REF-T4-pixel | T4_H_pixel_256 | 36000 | `mp3d_double_256` | stage-1 checkpoint that stage 2 resumes from |
| REF-T4-volume | T4_H_volume_256 | 36000 | `mp3d_double_256` | stage-2 checkpoint that stage 3 resumes from |

## Run tags

| tag | legacy script | config (`configs/OmniScene/`) | run dir | checkpoints | note |
|---|---|---|---|---|---|
| T1_H_all_256 | `evaluate_mp3d_double_256.py` | `omni_gs_160x320_mp3d_cylinder_all_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256` | 3000-69000 (23) | stage 3 (joint) 256x512. **REF-T1 = checkpoint-48000** |
| T1_D_all_256 | `evaluate_mp3d_double_256.py` | `omni_gs_160x320_mp3d_cylinder_all_256.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256` | 3000-45000 (15) | another run with the same name and config; best checkpoint-36000 (0.150 dB from Table 1) |
| T1_D_all_256_new | `evaluate_mp3d_double_256.py` | `omni_gs_160x320_mp3d_cylinder_all_256.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256_new` | 3000-51000 (17) | 9.5-14.5 dB at M3D from checkpoint-6000 on; not the paper model |
| T2_D_single_pixel_256 | `evaluate_mp3d_single_256.py` | `omni_gs_160x320_mp3d_cylinder_pixel_256_single.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_single_pixel_256` | 3000-48000 (16) | single view, pixel branch only. **REF-T2 = checkpoint-33000** |
| T2_H_all_256_v1probe | `evaluate_mp3d_single_256.py` | probe config (not in the repo): `omni_gs_160x320_mp3d_cylinder_all_256.py` with `pixel_gs.num_frames=1` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256` | 48000 | the 2-view joint model fed one view: 13.9-15.1 dB at M3D |
| T2_H_all_256_as_single | `evaluate_mp3d_single_256.py` | `omni_gs_160x320_mp3d_cylinder_all_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256` | 48000 | failed: launched with the two-view config (the model must be built with `num_frames=1`); replaced by the v1probe row |
| T3_D_360Loc_all_256 | `evaluate_360Loc_double_256.py` | `omni_gs_160x320_360Loc_cylinder_all_256.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256` | 3000-27000 (9) | 360Loc fine-tune. **REF-T3 = checkpoint-24000** |
| T3_H_360Loc_all_256 | `evaluate_360Loc_double_256.py` | `omni_gs_160x320_360Loc_cylinder_all_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256` | 10, 20, 30, 100, 200, 3000 | early checkpoints only |
| T4_H_pixel_256 | `evaluate_mp3d_double_256.py` | `omni_gs_160x320_mp3d_cylinder_pixel_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256` | 27000, 36000, 42000, 48000 | stage 1 (pixel). **REF-T4-pixel = checkpoint-36000** |
| T4_D_pixel_256 | `evaluate_mp3d_double_256.py` | `omni_gs_160x320_mp3d_cylinder_pixel_256.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256` | 27000, 36000, 42000, 48000 | 11.9-12.9 dB at M3D; not the stage-1 model the configs chain from |
| T4_H_volume_256 | `evaluate_mp3d_double_256.py` | `omni_gs_160x320_mp3d_cylinder_volume_256.py` | `/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256` | 3000-36000 (12) | stage 2 (volume). **REF-T4-volume = checkpoint-36000** |
| T4_D_volume_256 | `evaluate_mp3d_double_256.py` | `omni_gs_160x320_mp3d_cylinder_volume_256.py` | `/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256` | 3000-30000 (10) | 16.5-19.7 dB at M3D, 2.5-6.3 dB below T4_H_volume_256 at the same checkpoints |

## Paper rows (Ours)

| table | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m |
|---|---|---|---|---|---|
| Table 1 (two views) | 23.76 / .835 / .175 | 25.91 / .873 / .128 | 28.89 / .937 / .081 | 30.29 / .959 / .057 | 28.25 / .817 / .156 |
| Table 2 (single view) | 23.75 / .822 / .175 | 25.13 / .854 / .136 | 27.01 / .915 / .089 | 26.14 / .887 / .103 | 27.87 / .843 / .154 |
| Table 4 only pixel (context for REF-T4-pixel) | 23.21 / .817 / .179 | | | | |
| Table 4 only cylindrical volume (context for REF-T4-volume) | 22.17 / .782 / .210 | | | | |

Table 3 (360Loc, two views, 1.4 m): WS-PSNR 28.35, SSIM .896, LPIPS .095, PCC .884.

## Per-run results

### T1_H_all_256

Stage 3 (joint), H run. Two views, 256x512.

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m | mean abs. WS-PSNR diff. vs Table 1 |
|---|---|---|---|---|---|---|
| paper Table 1 Ours | 23.76 / .835 / .175 | 25.91 / .873 / .128 | 28.89 / .937 / .081 | 30.29 / .959 / .057 | 28.25 / .817 / .156 | |
| 3000 | 23.585 / 0.8297 / 0.1848 | 24.994 / 0.8597 / 0.1507 | 27.374 / 0.9050 / 0.0986 | 28.704 / 0.9462 / 0.0847 | 28.446 / 0.8793 / 0.1655 | 0.878 |
| 6000 | 23.886 / 0.8418 / 0.1797 | 25.555 / 0.8683 / 0.1431 | 27.794 / 0.9089 / 0.0949 | 29.309 / 0.9498 / 0.0817 | 28.261 / 0.8670 / 0.1688 | 0.514 |
| 9000 | 23.517 / 0.8282 / 0.1928 | 25.102 / 0.8593 / 0.1574 | 27.666 / 0.9081 / 0.0975 | 28.917 / 0.9444 / 0.0834 | 28.204 / 0.8665 / 0.1691 | 0.739 |
| 12000 | 24.298 / 0.8497 / 0.1650 | 25.630 / 0.8722 / 0.1354 | 27.366 / 0.9022 / 0.0968 | 29.065 / 0.9474 / 0.0816 | 27.930 / 0.8645 / 0.1711 | 0.777 |
| 15000 | 23.454 / 0.8311 / 0.1871 | 25.071 / 0.8630 / 0.1447 | 27.344 / 0.9071 / 0.0951 | 29.055 / 0.9500 / 0.0761 | 28.081 / 0.8706 / 0.1617 | 0.819 |
| 18000 | 23.731 / 0.8321 / 0.1880 | 25.720 / 0.8705 / 0.1397 | 28.245 / 0.9151 / 0.0880 | 29.798 / 0.9538 / 0.0723 | 28.083 / 0.8685 / 0.1605 | 0.305 |
| 21000 | 23.862 / 0.8430 / 0.1746 | 25.771 / 0.8709 / 0.1366 | 28.093 / 0.9132 / 0.0890 | 29.690 / 0.9534 / 0.0692 | 27.963 / 0.8634 / 0.1641 | 0.385 |
| 24000 | 22.527 / 0.8085 / 0.2033 | 24.601 / 0.8429 / 0.1567 | 27.911 / 0.9080 / 0.0951 | 28.575 / 0.9390 / 0.0837 | 28.532 / 0.8739 / 0.1616 | 1.104 |
| 27000 | 23.479 / 0.8350 / 0.1826 | 25.287 / 0.8641 / 0.1430 | 28.068 / 0.9143 / 0.0871 | 29.564 / 0.9519 / 0.0700 | 28.117 / 0.8671 / 0.1621 | 0.517 |
| 30000 | 23.262 / 0.8129 / 0.2077 | 24.804 / 0.8454 / 0.1636 | 27.933 / 0.9082 / 0.0999 | 29.067 / 0.9445 / 0.0900 | 28.162 / 0.8678 / 0.1668 | 0.774 |
| 33000 | 22.973 / 0.8102 / 0.1948 | 24.655 / 0.8395 / 0.1599 | 27.749 / 0.9058 / 0.0948 | 28.814 / 0.9413 / 0.0851 | 28.614 / 0.8768 / 0.1585 | 1.005 |
| 36000 | 23.398 / 0.8285 / 0.1838 | 25.343 / 0.8641 / 0.1415 | 28.465 / 0.9188 / 0.0831 | 29.898 / 0.9538 / 0.0724 | 28.352 / 0.8706 / 0.1579 | 0.370 |
| 39000 | 23.680 / 0.8311 / 0.1845 | 25.575 / 0.8658 / 0.1394 | 28.343 / 0.9182 / 0.0836 | 29.901 / 0.9541 / 0.0692 | 28.049 / 0.8654 / 0.1613 | 0.310 |
| 42000 | 24.111 / 0.8419 / 0.1752 | 26.136 / 0.8763 / 0.1299 | 28.227 / 0.9076 / 0.0877 | 29.794 / 0.9523 / 0.0713 | 27.796 / 0.8600 / 0.1675 | 0.438 |
| 45000 | 23.738 / 0.8329 / 0.1804 | 25.574 / 0.8690 / 0.1380 | 28.433 / 0.9191 / 0.0830 | 29.813 / 0.9540 / 0.0695 | 27.920 / 0.8621 / 0.1619 | 0.324 |
| **48000 (REF-T1)** | **23.653 / 0.8346 / 0.1801** | **25.912 / 0.8773 / 0.1292** | **28.590 / 0.9195 / 0.0809** | **30.290 / 0.9577 / 0.0641** | **28.172 / 0.8668 / 0.1567** | 0.097 |
| 51000 | 23.694 / 0.8353 / 0.1811 | 25.978 / 0.8763 / 0.1323 | 28.561 / 0.9206 / 0.0801 | 30.321 / 0.9584 / 0.0633 | 28.083 / 0.8655 / 0.1575 | 0.132 |
| 54000 | 23.028 / 0.8109 / 0.1984 | 25.089 / 0.8497 / 0.1454 | 28.682 / 0.9204 / 0.0802 | 29.973 / 0.9526 / 0.0676 | 28.333 / 0.8691 / 0.1570 | 0.432 |
| 57000 | 23.600 / 0.8305 / 0.1858 | 25.921 / 0.8742 / 0.1313 | 28.571 / 0.9198 / 0.0801 | 30.301 / 0.9586 / 0.0604 | 28.062 / 0.8644 / 0.1606 | 0.138 |
| 60000 | 23.272 / 0.8115 / 0.1962 | 25.141 / 0.8538 / 0.1471 | 28.596 / 0.9206 / 0.0846 | 29.691 / 0.9509 / 0.0748 | 28.200 / 0.8680 / 0.1653 | 0.440 |
| 63000 | 23.041 / 0.8083 / 0.1987 | 25.035 / 0.8501 / 0.1474 | 28.579 / 0.9199 / 0.0818 | 29.904 / 0.9529 / 0.0689 | 28.306 / 0.8714 / 0.1560 | 0.469 |
| 66000 | 23.480 / 0.8230 / 0.1883 | 25.791 / 0.8715 / 0.1309 | 28.834 / 0.9250 / 0.0759 | 30.416 / 0.9589 / 0.0601 | 28.128 / 0.8672 / 0.1585 | 0.141 |
| 69000 | 23.256 / 0.8154 / 0.1982 | 25.692 / 0.8682 / 0.1351 | 28.777 / 0.9240 / 0.0778 | 30.175 / 0.9577 / 0.0612 | 28.087 / 0.8652 / 0.1581 | 0.223 |

### T1_D_all_256

Stage 3 (joint), D run (same run-dir name and config as T1_H, different run).

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m | mean abs. WS-PSNR diff. vs Table 1 |
|---|---|---|---|---|---|---|
| paper Table 1 Ours | 23.76 / .835 / .175 | 25.91 / .873 / .128 | 28.89 / .937 / .081 | 30.29 / .959 / .057 | 28.25 / .817 / .156 | |
| 3000 | 22.921 / 0.8115 / 0.2073 | 24.379 / 0.8423 / 0.1634 | 27.252 / 0.9002 / 0.1048 | 28.073 / 0.9414 / 0.0932 | 28.338 / 0.8798 / 0.1611 | 1.263 |
| 6000 | 23.473 / 0.8358 / 0.1796 | 25.029 / 0.8662 / 0.1417 | 27.605 / 0.9078 / 0.0908 | 28.827 / 0.9453 / 0.0808 | 28.068 / 0.8696 / 0.1639 | 0.820 |
| 9000 | 22.499 / 0.8160 / 0.2081 | 24.445 / 0.8540 / 0.1520 | 27.623 / 0.9081 / 0.0929 | 28.850 / 0.9481 / 0.0842 | 27.840 / 0.8631 / 0.1657 | 1.169 |
| 12000 | 24.201 / 0.8401 / 0.1799 | 25.451 / 0.8612 / 0.1448 | 26.917 / 0.8869 / 0.1110 | 28.406 / 0.9399 / 0.0933 | 27.895 / 0.8640 / 0.1880 | 1.022 |
| 15000 | 23.937 / 0.8351 / 0.1815 | 25.329 / 0.8614 / 0.1454 | 27.730 / 0.9087 / 0.0935 | 29.100 / 0.9484 / 0.0738 | 28.295 / 0.8746 / 0.1624 | 0.631 |
| 18000 | 23.637 / 0.8364 / 0.1793 | 25.453 / 0.8665 / 0.1418 | 28.041 / 0.9112 / 0.0905 | 29.052 / 0.9491 / 0.0791 | 28.137 / 0.8692 / 0.1608 | 0.556 |
| 21000 | 23.907 / 0.8447 / 0.1746 | 25.270 / 0.8673 / 0.1430 | 27.805 / 0.9085 / 0.0921 | 28.973 / 0.9515 / 0.0685 | 28.187 / 0.8698 / 0.1629 | 0.650 |
| 24000 | 23.114 / 0.8142 / 0.1917 | 24.536 / 0.8363 / 0.1629 | 27.809 / 0.9055 / 0.0938 | 29.109 / 0.9461 / 0.0784 | 28.402 / 0.8748 / 0.1611 | 0.887 |
| 27000 | 23.345 / 0.8247 / 0.1914 | 25.244 / 0.8599 / 0.1452 | 28.193 / 0.9143 / 0.0873 | 29.593 / 0.9523 / 0.0690 | 28.268 / 0.8709 / 0.1612 | 0.499 |
| 30000 | 23.389 / 0.8211 / 0.1941 | 25.048 / 0.8516 / 0.1536 | 28.012 / 0.9082 / 0.0991 | 29.032 / 0.9454 / 0.0799 | 28.369 / 0.8693 / 0.1675 | 0.698 |
| 33000 | 23.328 / 0.8131 / 0.2008 | 24.881 / 0.8458 / 0.1580 | 28.039 / 0.9106 / 0.0943 | 29.089 / 0.9480 / 0.0820 | 28.534 / 0.8760 / 0.1646 | 0.759 |
| 36000 | 23.778 / 0.8342 / 0.1790 | 25.677 / 0.8687 / 0.1361 | 28.595 / 0.9201 / 0.0808 | 30.090 / 0.9543 / 0.0669 | 28.246 / 0.8691 / 0.1580 | 0.150 |
| 39000 | 23.509 / 0.8279 / 0.1885 | 25.482 / 0.8630 / 0.1418 | 28.362 / 0.9177 / 0.0837 | 29.792 / 0.9540 / 0.0668 | 28.141 / 0.8653 / 0.1631 | 0.363 |
| 42000 | 24.373 / 0.8477 / 0.1692 | 26.227 / 0.8783 / 0.1257 | 28.212 / 0.9084 / 0.0865 | 29.803 / 0.9507 / 0.0703 | 27.845 / 0.8604 / 0.1700 | 0.500 |
| 45000 | 23.867 / 0.8381 / 0.1795 | 25.550 / 0.8714 / 0.1361 | 28.420 / 0.9175 / 0.0825 | 29.852 / 0.9549 / 0.0651 | 27.908 / 0.8624 / 0.1623 | 0.343 |

### T1_D_all_256_new

Stage 3 (joint), D `_new` run (the `run2.sh` accelerate launch of the all stage writes to this directory).

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m | mean abs. WS-PSNR diff. vs Table 1 |
|---|---|---|---|---|---|---|
| paper Table 1 Ours | 23.76 / .835 / .175 | 25.91 / .873 / .128 | 28.89 / .937 / .081 | 30.29 / .959 / .057 | 28.25 / .817 / .156 | |
| 3000 | 16.573 / 0.5422 / 0.5929 | 16.821 / 0.5639 / 0.5814 | 17.691 / 0.5752 / 0.5709 | 16.141 / 0.5493 / 0.5545 | 20.848 / 0.6871 / 0.5611 | 9.805 |
| 6000 | 9.938 / 0.3736 / 0.6076 | 9.598 / 0.3775 / 0.6076 | 9.512 / 0.3967 / 0.5997 | 6.053 / 0.3546 / 0.6138 | 9.924 / 0.5012 / 0.5809 | 18.415 |
| 9000 | 11.045 / 0.4293 / 0.6284 | 10.860 / 0.4424 / 0.6291 | 10.782 / 0.4536 / 0.6266 | 6.915 / 0.4448 / 0.6656 | 11.218 / 0.5581 / 0.5979 | 17.256 |
| 12000 | 10.912 / 0.4281 / 0.6121 | 10.740 / 0.4385 / 0.6152 | 10.760 / 0.4554 / 0.6132 | 6.809 / 0.4340 / 0.6148 | 11.189 / 0.5575 / 0.6094 | 17.338 |
| 15000 | 10.997 / 0.4019 / 0.6078 | 10.992 / 0.4186 / 0.6115 | 11.081 / 0.4402 / 0.6063 | 6.998 / 0.4297 / 0.6154 | 11.188 / 0.5417 / 0.5817 | 17.169 |
| 18000 | 12.596 / 0.4684 / 0.6217 | 12.645 / 0.4834 / 0.6253 | 12.746 / 0.4921 / 0.6205 | 8.218 / 0.4897 / 0.6642 | 12.385 / 0.5861 / 0.6076 | 15.702 |
| 21000 | 12.220 / 0.4686 / 0.6373 | 12.194 / 0.4835 / 0.6377 | 12.191 / 0.4907 / 0.6339 | 7.903 / 0.4770 / 0.6706 | 11.953 / 0.5765 / 0.6354 | 16.128 |
| 24000 | 11.384 / 0.4365 / 0.6501 | 11.345 / 0.4487 / 0.6547 | 11.340 / 0.4558 / 0.6530 | 7.343 / 0.4330 / 0.6804 | 11.376 / 0.5449 / 0.6877 | 16.862 |
| 27000 | 12.837 / 0.4743 / 0.6235 | 12.837 / 0.4853 / 0.6288 | 12.977 / 0.4930 / 0.6291 | 8.312 / 0.4883 / 0.6669 | 12.497 / 0.5854 / 0.6198 | 15.528 |
| 30000 | 12.049 / 0.4596 / 0.6767 | 12.015 / 0.4717 / 0.6818 | 12.029 / 0.4745 / 0.6831 | 7.843 / 0.4452 / 0.7281 | 11.721 / 0.5578 / 0.7036 | 16.289 |
| 33000 | 13.013 / 0.4719 / 0.6605 | 13.045 / 0.4864 / 0.6670 | 13.172 / 0.4936 / 0.6635 | 8.625 / 0.4876 / 0.7218 | 12.551 / 0.5860 / 0.6536 | 15.339 |
| 36000 | 13.106 / 0.4775 / 0.6657 | 13.130 / 0.4869 / 0.6684 | 13.247 / 0.4932 / 0.6638 | 8.834 / 0.4723 / 0.7193 | 12.707 / 0.5868 / 0.6568 | 15.215 |
| 39000 | 13.335 / 0.4855 / 0.6650 | 13.324 / 0.4957 / 0.6711 | 13.492 / 0.5034 / 0.6689 | 9.129 / 0.5083 / 0.7336 | 12.826 / 0.5892 / 0.6743 | 14.999 |
| 42000 | 13.568 / 0.4871 / 0.6901 | 13.700 / 0.4988 / 0.6956 | 13.831 / 0.5060 / 0.6940 | 9.297 / 0.5074 / 0.7483 | 12.708 / 0.5901 / 0.7067 | 14.799 |
| 45000 | 13.685 / 0.4873 / 0.7035 | 13.957 / 0.5035 / 0.7074 | 14.181 / 0.5114 / 0.7025 | 9.793 / 0.5180 / 0.7668 | 13.202 / 0.6038 / 0.7029 | 14.456 |
| 48000 | 14.006 / 0.4974 / 0.6958 | 14.247 / 0.5127 / 0.6997 | 14.486 / 0.5219 / 0.6977 | 10.230 / 0.5362 / 0.7777 | 13.428 / 0.6131 / 0.7075 | 14.141 |
| 51000 | 13.746 / 0.4905 / 0.7146 | 13.947 / 0.5044 / 0.7198 | 14.147 / 0.5134 / 0.7167 | 10.262 / 0.5277 / 0.7897 | 13.237 / 0.6123 / 0.7325 | 14.352 |

### T2_D_single_pixel_256

Single view (context = the middle frame), pixel branch only, `pixel_gs.num_frames=1`. Compared with paper Table 2.

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m | mean abs. WS-PSNR diff. vs Table 2 |
|---|---|---|---|---|---|---|
| paper Table 2 Ours | 23.75 / .822 / .175 | 25.13 / .854 / .136 | 27.01 / .915 / .089 | 26.14 / .887 / .103 | 27.87 / .843 / .154 | |
| 3000 | 22.387 / 0.7409 / 0.2574 | 23.509 / 0.7852 / 0.2153 | 24.925 / 0.8346 / 0.1576 | 24.206 / 0.8876 / 0.1494 | 27.553 / 0.8545 / 0.1746 | 1.464 |
| 6000 | 23.997 / 0.7571 / 0.2316 | 25.110 / 0.7970 / 0.1889 | 26.827 / 0.8492 / 0.1279 | 28.135 / 0.9255 / 0.1031 | 29.357 / 0.8717 / 0.1442 | 0.786 |
| 9000 | 23.611 / 0.7561 / 0.2297 | 24.628 / 0.7922 / 0.1915 | 26.065 / 0.8418 / 0.1329 | 26.348 / 0.9187 / 0.1097 | 29.285 / 0.8697 / 0.1410 | 0.642 |
| 12000 | 23.874 / 0.7396 / 0.2381 | 24.991 / 0.7792 / 0.1943 | 26.743 / 0.8318 / 0.1313 | 28.174 / 0.9219 / 0.1006 | 29.489 / 0.8657 / 0.1409 | 0.837 |
| 15000 | 24.071 / 0.7604 / 0.2377 | 25.124 / 0.7988 / 0.1926 | 26.677 / 0.8498 / 0.1305 | 27.836 / 0.9191 / 0.1054 | 29.520 / 0.8660 / 0.1455 | 0.801 |
| 18000 | 24.194 / 0.7646 / 0.2324 | 25.210 / 0.8037 / 0.1862 | 26.850 / 0.8536 / 0.1245 | 27.624 / 0.9226 / 0.1023 | 29.676 / 0.8642 / 0.1395 | 0.795 |
| 21000 | 24.162 / 0.7607 / 0.2317 | 25.296 / 0.8029 / 0.1853 | 27.020 / 0.8545 / 0.1223 | 27.778 / 0.9190 / 0.1057 | 29.708 / 0.8650 / 0.1398 | 0.813 |
| 24000 | 24.132 / 0.7597 / 0.2338 | 25.209 / 0.7997 / 0.1879 | 26.766 / 0.8502 / 0.1250 | 27.569 / 0.9136 / 0.1064 | 29.587 / 0.8625 / 0.1400 | 0.770 |
| 27000 | 24.040 / 0.7495 / 0.2343 | 25.041 / 0.7863 / 0.1884 | 26.704 / 0.8347 / 0.1278 | 28.713 / 0.9274 / 0.0884 | 29.920 / 0.8636 / 0.1358 | 1.062 |
| 30000 | 23.474 / 0.7359 / 0.2662 | 24.706 / 0.7840 / 0.2061 | 26.646 / 0.8442 / 0.1372 | 26.548 / 0.8852 / 0.1294 | 29.437 / 0.8507 / 0.1482 | 0.608 |
| **33000 (REF-T2)** | **23.910 / 0.7513 / 0.2450** | **25.075 / 0.7944 / 0.1948** | **26.782 / 0.8464 / 0.1299** | **26.746 / 0.8948 / 0.1190** | **29.593 / 0.8585 / 0.1425** | 0.554 |
| 36000 | 24.417 / 0.7633 / 0.2314 | 25.518 / 0.8028 / 0.1821 | 27.330 / 0.8548 / 0.1181 | 28.764 / 0.9277 / 0.0910 | 30.152 / 0.8608 / 0.1340 | 1.256 |
| 39000 | 24.475 / 0.7646 / 0.2315 | 25.545 / 0.8057 / 0.1808 | 27.386 / 0.8580 / 0.1166 | 28.519 / 0.9262 / 0.0923 | 29.964 / 0.8611 / 0.1360 | 1.198 |
| 42000 | 24.519 / 0.7558 / 0.2328 | 25.560 / 0.7940 / 0.1862 | 27.304 / 0.8457 / 0.1221 | 28.838 / 0.9234 / 0.0911 | 30.209 / 0.8599 / 0.1372 | 1.306 |
| 45000 | 24.508 / 0.7625 / 0.2339 | 25.610 / 0.8039 / 0.1832 | 27.467 / 0.8579 / 0.1174 | 29.119 / 0.9262 / 0.0898 | 30.326 / 0.8640 / 0.1356 | 1.426 |
| 48000 | 24.457 / 0.7638 / 0.2337 | 25.549 / 0.8051 / 0.1819 | 27.372 / 0.8594 / 0.1162 | 29.021 / 0.9260 / 0.0885 | 30.125 / 0.8598 / 0.1375 | 1.325 |

### T2_H_all_256_v1probe

The 2-view joint model (REF-T1's checkpoint) built with `pixel_gs.num_frames=1` and fed one view. Compared with paper Table 2.

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m | mean abs. WS-PSNR diff. vs Table 2 |
|---|---|---|---|---|---|---|
| paper Table 2 Ours | 23.75 / .822 / .175 | 25.13 / .854 / .136 | 27.01 / .915 / .089 | 26.14 / .887 / .103 | 27.87 / .843 / .154 | |
| 48000 | 13.856 / 0.4698 / 0.4687 | 14.485 / 0.4927 / 0.4491 | 15.138 / 0.5169 / 0.4141 | 15.455 / 0.5982 / 0.3886 | 17.706 / 0.6272 / 0.3749 | 10.652 |

### T4_H_pixel_256

Stage 1 (pixel branch), H run.

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m |
|---|---|---|---|---|---|
| 27000 | 23.375 / 0.8343 / 0.1892 | 24.905 / 0.8613 / 0.1486 | 27.866 / 0.9114 / 0.0879 | 29.253 / 0.9505 / 0.0740 | 27.816 / 0.8656 / 0.1592 |
| **36000 (REF-T4-pixel)** | **23.224 / 0.8309 / 0.1852** | **25.297 / 0.8680 / 0.1379** | **28.176 / 0.9162 / 0.0826** | **29.976 / 0.9558 / 0.0658** | **27.793 / 0.8621 / 0.1590** |
| 42000 | 23.238 / 0.8255 / 0.1895 | 25.462 / 0.8660 / 0.1372 | 28.158 / 0.9151 / 0.0818 | 30.045 / 0.9543 / 0.0647 | 27.601 / 0.8597 / 0.1623 |
| 48000 | 22.847 / 0.8195 / 0.2010 | 24.964 / 0.8586 / 0.1458 | 28.191 / 0.9173 / 0.0812 | 30.139 / 0.9559 / 0.0626 | 27.770 / 0.8616 / 0.1589 |

### T4_D_pixel_256

Stage 1 (pixel branch), D run.

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m |
|---|---|---|---|---|---|
| 27000 | 12.919 / 0.4903 / 0.6473 | 12.727 / 0.4971 / 0.6509 | 12.650 / 0.5007 / 0.6509 | 8.079 / 0.4894 / 0.6021 | 12.660 / 0.6082 / 0.6511 |
| 36000 | 12.500 / 0.4840 / 0.6558 | 12.320 / 0.4897 / 0.6548 | 12.219 / 0.4908 / 0.6538 | 7.876 / 0.4753 / 0.6110 | 12.479 / 0.6007 / 0.6482 |
| 42000 | 12.278 / 0.4704 / 0.6640 | 12.055 / 0.4736 / 0.6648 | 11.909 / 0.4698 / 0.6652 | 7.626 / 0.4414 / 0.6325 | 12.219 / 0.5837 / 0.6511 |
| 48000 | 12.219 / 0.4637 / 0.6568 | 12.033 / 0.4678 / 0.6569 | 11.893 / 0.4663 / 0.6569 | 7.525 / 0.4342 / 0.6317 | 12.056 / 0.5761 / 0.6469 |

### T4_H_volume_256

Stage 2 (volume branch), H run.

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m |
|---|---|---|---|---|---|
| 3000 | 20.319 / 0.6581 / 0.4228 | 21.351 / 0.6957 / 0.3958 | 22.688 / 0.7460 / 0.3430 | 23.003 / 0.8118 / 0.3109 | 26.367 / 0.8209 / 0.3738 |
| 6000 | 20.443 / 0.6715 / 0.3980 | 21.486 / 0.7142 / 0.3598 | 22.988 / 0.7647 / 0.3068 | 23.259 / 0.8147 / 0.2847 | 26.669 / 0.8329 / 0.3429 |
| 9000 | 20.460 / 0.6665 / 0.3953 | 21.590 / 0.7096 / 0.3637 | 23.234 / 0.7670 / 0.3078 | 23.127 / 0.8273 / 0.2829 | 26.537 / 0.8305 / 0.3378 |
| 12000 | 20.765 / 0.6763 / 0.3836 | 22.067 / 0.7253 / 0.3373 | 23.855 / 0.7829 / 0.2732 | 23.901 / 0.8365 / 0.2547 | 26.925 / 0.8365 / 0.3220 |
| 15000 | 20.653 / 0.6801 / 0.3842 | 22.001 / 0.7299 / 0.3362 | 23.783 / 0.7874 / 0.2735 | 23.949 / 0.8353 / 0.2517 | 26.503 / 0.8332 / 0.3241 |
| 18000 | 21.200 / 0.6932 / 0.3580 | 22.382 / 0.7403 / 0.3154 | 24.146 / 0.7958 / 0.2570 | 24.191 / 0.8479 / 0.2308 | 27.137 / 0.8405 / 0.3057 |
| 21000 | 21.077 / 0.6957 / 0.3620 | 22.580 / 0.7473 / 0.3062 | 24.465 / 0.8038 / 0.2466 | 24.337 / 0.8496 / 0.2225 | 27.247 / 0.8401 / 0.3106 |
| 24000 | 21.136 / 0.6909 / 0.3560 | 22.595 / 0.7431 / 0.3056 | 24.399 / 0.8027 / 0.2399 | 24.217 / 0.8400 / 0.2214 | 27.513 / 0.8442 / 0.2919 |
| 27000 | 21.180 / 0.6998 / 0.3463 | 22.553 / 0.7511 / 0.2964 | 24.522 / 0.8075 / 0.2333 | 24.137 / 0.8428 / 0.2180 | 27.507 / 0.8424 / 0.2951 |
| 30000 | 21.431 / 0.7059 / 0.3416 | 22.764 / 0.7537 / 0.2967 | 24.520 / 0.8045 / 0.2390 | 24.452 / 0.8565 / 0.2179 | 26.747 / 0.8309 / 0.3030 |
| 33000 | 21.100 / 0.6893 / 0.3542 | 22.556 / 0.7395 / 0.3042 | 24.735 / 0.8026 / 0.2370 | 24.818 / 0.8635 / 0.2211 | 27.380 / 0.8426 / 0.2868 |
| **36000 (REF-T4-volume)** | **21.428 / 0.7042 / 0.3401** | **22.904 / 0.7560 / 0.2886** | **24.832 / 0.8127 / 0.2260** | **24.248 / 0.8405 / 0.2174** | **27.360 / 0.8398 / 0.2930** |

### T4_D_volume_256

Stage 2 (volume branch), D run.

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica 1.0 m | Residential ~0.3 m |
|---|---|---|---|---|---|
| 3000 | 17.049 / 0.5675 / 0.5951 | 17.143 / 0.5848 / 0.6018 | 17.223 / 0.5981 / 0.6064 | 16.217 / 0.6281 / 0.5650 | 20.463 / 0.7294 / 0.6115 |
| 6000 | 17.844 / 0.5838 / 0.5534 | 18.086 / 0.6030 / 0.5510 | 18.182 / 0.6129 / 0.5573 | 16.375 / 0.6330 / 0.5319 | 21.075 / 0.7385 / 0.5681 |
| 9000 | 17.927 / 0.5896 / 0.5786 | 18.436 / 0.6113 / 0.5702 | 19.106 / 0.6301 / 0.5451 | 17.588 / 0.6566 / 0.5095 | 23.042 / 0.7567 / 0.5467 |
| 12000 | 17.017 / 0.5800 / 0.5991 | 17.497 / 0.5985 / 0.5964 | 18.116 / 0.6134 / 0.5842 | 16.560 / 0.6381 / 0.5379 | 21.902 / 0.7443 / 0.5710 |
| 15000 | 17.711 / 0.5893 / 0.5658 | 18.439 / 0.6152 / 0.5527 | 19.062 / 0.6332 / 0.5454 | 17.839 / 0.6576 / 0.4918 | 21.072 / 0.7420 / 0.5744 |
| 18000 | 18.051 / 0.6030 / 0.5659 | 18.931 / 0.6266 / 0.5477 | 19.718 / 0.6463 / 0.5314 | 18.319 / 0.6771 / 0.4790 | 22.970 / 0.7514 / 0.5525 |
| 21000 | 16.862 / 0.5746 / 0.6089 | 17.687 / 0.5994 / 0.5943 | 18.613 / 0.6179 / 0.5818 | 17.408 / 0.6482 / 0.5116 | 21.754 / 0.7419 / 0.5803 |
| 24000 | 17.187 / 0.5917 / 0.5911 | 18.133 / 0.6164 / 0.5721 | 19.453 / 0.6459 / 0.5330 | 18.383 / 0.6774 / 0.4695 | 23.173 / 0.7613 / 0.5187 |
| 27000 | 16.507 / 0.5747 / 0.6245 | 17.140 / 0.5942 / 0.6145 | 18.468 / 0.6166 / 0.5794 | 17.545 / 0.6594 / 0.5018 | 22.170 / 0.7480 / 0.5521 |
| 30000 | 16.736 / 0.5777 / 0.6108 | 17.205 / 0.5943 / 0.6049 | 18.219 / 0.6148 / 0.5737 | 17.483 / 0.6627 / 0.4979 | 22.302 / 0.7497 / 0.5593 |

### T3_D_360Loc_all_256

360Loc, D run (`val` split, Total line only).

| checkpoint | PSNR | WS-PSNR | SSIM | LPIPS | pcc (script) |
|---|---|---|---|---|---|
| paper Table 3 Ours | | 28.35 | .896 | .095 | (PCC .884, other reference) |
| 3000 | 24.956 | 26.074 | 0.8332 | 0.1496 | 0.9893 |
| 6000 | 24.903 | 26.115 | 0.8363 | 0.1481 | 0.9898 |
| 9000 | 26.567 | 27.631 | 0.8502 | 0.1242 | 0.9891 |
| 12000 | 26.604 | 27.637 | 0.8556 | 0.1179 | 0.9894 |
| 15000 | 26.724 | 27.717 | 0.8638 | 0.1198 | 0.9895 |
| 18000 | 26.621 | 27.638 | 0.8651 | 0.1209 | 0.9896 |
| 21000 | 26.990 | 27.948 | 0.8670 | 0.1145 | 0.9900 |
| **24000 (REF-T3)** | **27.244** | **28.205** | **0.8708** | **0.1134** | **0.9897** |
| 27000 | 27.229 | 28.184 | 0.8688 | 0.1117 | 0.9900 |

### T3_H_360Loc_all_256

360Loc, H run (early checkpoints only).

| checkpoint | PSNR | WS-PSNR | SSIM | LPIPS | pcc (script) |
|---|---|---|---|---|---|
| paper Table 3 Ours | | 28.35 | .896 | .095 | (PCC .884, other reference) |
| 10 | 21.254 | 21.828 | 0.7308 | 0.2472 | 0.9809 |
| 20 | 21.514 | 22.105 | 0.7348 | 0.2428 | 0.9812 |
| 30 | 21.662 | 22.240 | 0.7378 | 0.2403 | 0.9812 |
| 100 | 22.348 | 23.158 | 0.7543 | 0.2229 | 0.9656 |
| 200 | 24.051 | 24.950 | 0.7950 | 0.1827 | 0.9859 |
| 3000 | 25.078 | 26.193 | 0.8318 | 0.1512 | 0.9893 |

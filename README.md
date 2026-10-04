# CylinderSplat

<h3 align="center">[ICLR 2026] 3D Gaussian Splatting with Cylindrical Triplanes for Panoramic Novel View Synthesis</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2603.05882">Paper (arXiv 2603.05882)</a> ·
  <a href="https://github.com/wangqww/CylinderSplat">Code</a>
</p>

CylinderSplat is a 3D Gaussian Splatting framework for panoramic novel view synthesis.

---

## Overview

CylinderSplat is a feed-forward model that turns one or a few input panoramas into
3D Gaussians and renders new panoramas directly with an equirectangular rasteriser. It has two branches:

- the **pixel branch** predicts per-pixel Gaussians from a ResNet + multi-view attention backbone
  and a UniK3D depth prior;
- the **volume branch** builds a cylindrical triplane around each input camera, refines it with
  triplane attention and decodes Gaussians that fill regions the inputs do not see.

This version of the code has one training entry point (`train.py`) and one evaluation
entry point (`evaluate.py`). They replace the per-experiment scripts, which are kept
unchanged in [`legacy/`](legacy/README.md). **With every switch at its default, the
entry points compute what the released scripts computed.** The released checkpoints
evaluate to the same numbers, bit for bit, and training follows the legacy trainer of the
chosen row. Improvements are opt-in [switches](#opt-in-switches). Changes are listed in
[`CHANGELOG_dev.md`](CHANGELOG_dev.md).

**Checkpoints and reproduction.** The `dev` release retrains every stage with this code and publishes
the five models, their training configs and the derived depth files:
see [Reproducing the dev release](#reproducing-the-dev-release).

---

## Installation

The released results were produced with Python 3.10, CUDA 11.8 and PyTorch 2.1.0 on RTX 4090 GPUs. Building the CUDA
extensions needs the CUDA 11.8 toolkit (`nvcc`). Follow this order, which is the one in the
header of [`requirements.txt`](requirements.txt):

```bash
git clone --recursive https://github.com/wangqww/CylinderSplat.git
cd CylinderSplat
git submodule update --init --recursive      # glm, needed to build pano_gaussian

# inside a fresh Python 3.10 environment (conda, venv or uv)
# 0. build tools (torch 2.1 needs setuptools < 70 to compile extensions)
pip install "setuptools<70" wheel ninja

# 1. PyTorch 2.1.0 + CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 2. mmcv 2.1.0 with CUDA ops, built from source
pip install psutil mmengine==0.10.7
git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git ../mmcv
(cd ../mmcv && MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install --no-build-isolation -e . -v)

# 3. pytorch3d 0.7.8, built from source
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

# 4. pinned Python dependencies (the file's --find-links line gets torch-scatter from the PyG wheel index)
pip install -r requirements.txt

# 5. local CUDA extensions: panorama Gaussian rasteriser and simple-knn
pip install --no-build-isolation ./pano_gaussian ./simple-knn
```

Steps 2, 3 and 5 compile against the torch installed in step 1, so they pass
`--no-build-isolation`. Without it, pip builds in an isolated environment that has no torch:
pytorch3d, `pano_gaussian` and `simple-knn` then fail with `No module named 'torch'`, and mmcv may
install without its CUDA ops (`No module named 'mmcv._ext'` at `import model`).

**With uv**, create the environment with `uv venv -p 3.10` and run every step with `uv pip install`
in place of `pip install`, with the same flags. uv builds in isolation unless it gets
`--no-build-isolation`, so steps 2, 3 and 5 need the flag there too.

The optional packages `open3d` (point-cloud dumps) and `pytest` (`tests/`) are listed at the end
of `requirements.txt`. UniK3D, which generates the depth prior, needs its own environment
(see *Depth prior* under [Data](#data)). `taming-transformers` is not needed: `model/losses.py`
reads the LPIPS weights shipped under `taming/` itself.

**Docker.** [`docker/Dockerfile`](docker/Dockerfile) runs the same steps on
`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`. The image has the dependencies and the two
extensions but not the code, so mount the repository:

```bash
git submodule update --init --recursive      # before the build: the image compiles pano_gaussian
docker build -f docker/Dockerfile -t cylindersplat .
#   optional: --build-arg TORCH_CUDA_ARCH_LIST="8.9" --build-arg HTTPS_PROXY=... --build-arg APT_MIRROR=...
docker run --gpus all --ipc=host -it -v "$PWD":/workspace -v /path/to/data:/path/to/data cylindersplat
```

### Weights

| What | Where it comes from | Needed for |
|---|---|---|
| CylinderSplat checkpoints of the `dev` release (5 models) | [OneDrive folder](https://1drv.ms/f/c/86d953bfc66eb903/IgCCry9vB7yVTbRvR5QKMIhPAWsS880HvFfSe8ABdaKQKZY), see [Reproducing the dev release](#reproducing-the-dev-release) | evaluation, fine-tuning |
| PanSplat checkpoint `pansplat_last.ckpt` (Lightning `.ckpt`, keys `encoder.backbone.*`; the author's PanSplat run at 256×512; SHA-256 `05b893817f9605d228db20221fbf7b983b35ef9f0473b255afd8409b58782cd1`) | [OneDrive](https://1drv.ms/u/c/86d953bfc66eb903/IQC8bZhj4FWPRJt9U5c0KrmlAR9-KBSQfIIXZWrYh8cOR7c?e=dPhQCt) (place it under `checkpoints/`) | every **`OmniGaussianCylinderPixel`** config (see below), i.e. stage 1 |
| LPIPS-VGG linear layers | shipped: `taming/modules/autoencoder/lpips/vgg.pth` | training loss |
| torchvision VGG16 ImageNet weights | downloaded to the torch hub cache on first use | LPIPS (training and evaluation) |
| UniK3D ViT-L (`lpiccinelli/unik3d-vitl`) | Hugging Face, via [UniK3D](https://github.com/lpiccinelli-eth/UniK3D) | depth-prior generation only |

The entry points take checkpoint paths on the command line (`evaluate.py --ckpt`,
`train.py --resume-from`). Each accepts a checkpoint directory that holds `model.safetensors`,
or the file itself. Only the ablation models (the UniFuse / Depth Anywhere variants) read
`checkpoints/...` by themselves, relative to the working directory.

The pixel-branch model (`OmniGaussianCylinderPixel`) initialises its backbone from the
PanSplat checkpoint when it is built, so this also happens during evaluation and when resuming. It
refuses to build if the file is missing. The default path is the author's machine. Set
`backbone_ckpt='/path/to/pansplat.ckpt'` inside `model = dict(...)` of every config whose model
type is `OmniGaussianCylinderPixel`. In `configs/OmniScene/` these are
`omni_gs_160x320_mp3d_cylinder_pixel_256.py`, `..._pixel_256_single.py`, `..._pixel_512.py`,
`..._pixel_256_mgs.py`, `omni_gs_160x320_mp3d_cylinder_pixel.py` (row `mp3d_double_160`) and
`omni_gs_160x320_VIGOR_cylinder_pixel.py` (row `kansas_double_160`).

---

## Data

The loaders read panoramas at 1024×512 and resize them to the model resolution (256×512
for the `*_256` loaders). Dataset roots are constants in the loaders. Point them at your
copies:

| Loader | Constant | Default (author's machine) |
|---|---|---|
| `data/mp3d_dataloader_{double_256,single_256,double_512,double}.py` | module-level `roots` | `/data/qiwei/nips25/pano_grf` |
| `data/loc360_dataloader_double_all_512.py` | `root` in `Dataset360Loc.__init__` | `/data/qiwei/nips25/360Loc` |
| `data/vigor_dataloader_double.py` (Kansas City) | `root_dir` in `load_VIGOR_data` (used by `evaluate.py`) **and** the `kansas_double_160` row of `configs/entries.py` (used by `train.py`) | `/data/qiwei/nips25/` |

**Matterport3D / Replica / Residential** are the PanoGRF renders (the same data as PanSplat). Each
sequence has three views:

```
<pano_grf>/png_render_{train,val}_1024x512_seq_len_3_m3d_dist_0.5/<scene>/<view>/
<pano_grf>/png_render_test_1024x512_seq_len_3_<name>_dist_<d>/<scene>/<view>/
        name_dist_d in: m3d_dist_{0.1,0.25,0.5,0.75,1.0}, residential_dist_0.15, replica_dist_0.5
<view>/ rgb.png             panorama
        rot.txt, tran.txt   camera rotation / translation
        depth.png           GT depth in mm (m3d sets only; train and test)
        depth_anywhere.png  Depth Anywhere depth, the PCC reference (not generated by this repo)
        depth_metric.npy    UniK3D metric distance   } the depth prior, generated with
        depth_conf.npy      UniK3D confidence        } tools/prepare_unik3d_depth.py
```

The views are sorted by name. The two-view loaders take views [0, 2] as inputs. The single-view
loader takes view [1].

**360Loc**: `train` uses concourse, hall and piatrium. `val` uses atrium, which is the test set of the paper.

```
<360Loc>/<location>/{mapping,query_360}/<sequence with "360" in its name>/
        camera_pose.json                         {frame file name: 4x4 pose}
        image/<frame>.jpg
        depth_metric/<frame>_depth.npy, <frame>_conf.npy     the UniK3D prior
```

**Kansas City** street-view panoramas (paper App. F; called `VIGOR` in the code) need
`Kansas/{train_list.txt,test_list_om.txt}` and, per road,
`Ground/`, `Satellite/`, `GrdInSat_dist_dir_month_new/` and `depth_metric/`. The depth tool
does not generate the Kansas City depth priors.

**Depth prior.** [`tools/prepare_unik3d_depth.py`](tools/prepare_unik3d_depth.py) is the
author's UniK3D generator: ViT-L, resolution level 9, 1024×512 spherical camera. By default
it leaves the dataset untouched and writes a mirrored tree under `--out-root`, which you merge
into your data copy. `--in-place` instead writes next to the images of a copy that is not
protected.

UniK3D needs Python ≥ 3.11, torch ≥ 2.4 and numpy ≥ 2, so it cannot go into the Python 3.10 /
torch 2.1 environment above (installing it there would upgrade torch and numpy and break the
extensions). Give it its own environment. The tool needs only UniK3D, torch, numpy and Pillow from
it, plus `tools/write_guard.py` from this repository, so run it from the repository root with that
environment's Python:

```bash
# a separate environment for UniK3D (with uv: uv venv -p 3.11 ~/venvs/unik3d, then uv pip install)
python3.11 -m venv ~/venvs/unik3d
git clone https://github.com/lpiccinelli-eth/UniK3D ../UniK3D
~/venvs/unik3d/bin/pip install -e ../UniK3D --extra-index-url https://download.pytorch.org/whl/cu121

# from the CylinderSplat root
U=~/venvs/unik3d/bin/python
$U tools/prepare_unik3d_depth.py --dataset mp3d   --data-root /path/pano_grf --out-root /path/pano_grf_depth
$U tools/prepare_unik3d_depth.py --dataset loc360 --data-root /path/360Loc   --out-root /path/360Loc_depth
rsync -a /path/pano_grf_depth/ /path/pano_grf/      # merge (only into your own copy)
```

Other options are `--stages` (MP3D splits, default `train val test`), `--batch-size`,
`--skip-existing` and `--unik3d` (hub id or local directory of the weights). The tool lists the
same views as the loaders, skipping `.DS_Store` entries. Before it loads UniK3D, it resolves every
output file through symlinks and stops if any file falls inside a protected tree
([write guard](#run-directories-and-the-write-guard)). It also refuses any output path that goes
through a symlink, because the write would land in the link's target. So `--in-place` needs a real
copy of the dataset, not a tree of links to the original. Files are written to a temp file and swapped in
with `os.replace`, so a hard-linked copy (`cp -al`) never changes the original's depth files.

---

## Reproducing the dev release

The `dev` branch retrained the MP3D schedule stage by stage with this code (September–October 2026),
chose every MP3D checkpoint on the validation split (`mp3d_double_256_val`, never on a test set), and
fine-tuned the stage-3 model on 360Loc with PanSplat's protocol. The five models, the configs that
trained them and the derived depth files are public, so the numbers below can be checked without
training and retrained with the commands at the end.

### Checkpoints

All files are in the OneDrive folder [`dev_release_20261004`](https://1drv.ms/f/c/86d953bfc66eb903/IgCCry9vB7yVTbRvR5QKMIhPAWsS880HvFfSe8ABdaKQKZY). Keep its layout:
`checkpoints/dev_release_20261004/<name>/model.safetensors`. Each `<name>` directory is what
`evaluate.py --ckpt` and `train.py --resume-from` take.

| `<name>` | model, resolution | training (config in `configs/OmniScene/`) | checkpoint | download |
|---|---|---|---|---|
| `mp3d_stage1_pixel_256x512` | pixel branch, 256×512 | stage 1, `release/stage1_pixel_256_b4.py`, 15 epochs | step 24000, best val | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQA6TUmSBMNRQp6WLuO2BCD3AaNtZGyFP_jYO_Rhlw0wEL4) |
| `mp3d_stage2_volume_256x512` | volume branch, 256×512 | stage 2 from stage 1, `release/stage2_volume_256_b4.py`, 15 epochs | step 24000, best val | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQBYPWqJO81rS7U3N0vzZfOaAZzo2wOmxVn4cQGfMgtt9LA) |
| `mp3d_stage3_joint_256x512` | joint model, 256×512 | stage 3 from stage 2, `release/stage3_all_256.py`, 25 epochs | step 78000, best val | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQDhstJnrudiQLWJ4RVIianaAbsqTMbhatRobwcAxVh2kqk) |
| `mp3d_stage4_joint_512x1024` | joint model, 512×1024 | stage 4 from stage 3, `omni_gs_160x320_mp3d_cylinder_all_512x1024.py`; stopped at step 44000 of 50000 | step 39000, best val (at 512×1024) | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQBpTaG90iA0S7v0ZOJXg2OkARgXpk5pPwCFZ-lo_9P1DPQ) |
| `loc360_finetune_256x512` | 360Loc model, 256×512 | fine-tuned from stage 3, `release/loc360_finetune_256.py`, 10 epochs | the final weights (step 21670) | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQAddPv-EkKyQKNV66dYBCueAd5z6pSQyS9rXZSSZbb8VEw) |

SHA-256 ([`SHA256SUMS`](https://1drv.ms/u/c/86d953bfc66eb903/IQCz_-zkkTOBRKfRmvHDpJptAY6_gGv4YIqOqmxWMxR7fPw)):

```
092df6c49d3006bd3c9fa9de540a5021478cd0a244b7b981867d131d280feb37  mp3d_stage1_pixel_256x512/model.safetensors
6ea29322cf57ee58f82cf80a2fa0eacf9e6c51933c2f1768bd147d81e729a96c  mp3d_stage2_volume_256x512/model.safetensors
f93174563f64ee7e02974445910938b1695c53575b77dd2abaea1d9aa4b243b8  mp3d_stage3_joint_256x512/model.safetensors
2b0890f95ca2938204a4974fc15bcab3e692d8b53f69dc16a6f3458718f5ae53  mp3d_stage4_joint_512x1024/model.safetensors
9dc15451fc5d5a68d4acb3f67e0264ddef88058517528194aa8b12a334084dbf  loc360_finetune_256x512/model.safetensors
```

### Data

1. **Images and poses.** Download them as described in PanSplat's
   [Data Preparation](https://github.com/chengzhag/PanSplat?tab=readme-ov-file#-data-preparation):
   *PanoGRF Data* (`pano_grf_lr.tar`: the Matterport3D, Replica and Residential renders) and
   *360Loc Data* (the official 360Loc release). The layouts this code reads are in [Data](#data); set the
   loader constants listed there to your two roots.
2. **Derived depth.** The loaders also read a UniK3D depth prior (`depth_metric.npy` / `depth_conf.npy`,
   and `depth_metric/<frame>_{depth,conf}.npy` for 360Loc) and the Depth Anywhere depth that serves as
   the PCC reference (`depth_anywhere.png`, and `depthanywhere/<frame>_depth_anywhere.png` for 360Loc).
   Neither comes with the downloads above. The `depth/` subfolder of the release holds the author's
   files; extract each archive inside its dataset root:

   | archive | contents | needed for | size |
   |---|---|---|---|
   | [`pano_grf_evalsets_depth.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQDm8ob5GRE4Q7QejEQELow6AX9HFExXjL5C1dJ0DRlFK9o) | test and val sets: `depth_anywhere.png`, `depth_metric.npy`, `depth_conf.npy` of every view | evaluation and checkpoint selection on MP3D / Replica / Residential | 2.2 GB |
   | [`pano_grf_train_depth_anywhere.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQCA1xVgPHK2TagBQ0C3dJxfAUNEBgwI1beo-L1sbKGveK0) | train set: `depth_anywhere.png` of every view | MP3D training (the loader reads it) | 3.4 GB |
   | [`360Loc_atrium_depth.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQDGQ7ri97nDRr31a7Km2z1_AdclbMEfpN1vBaNiLP_Kyn0) | atrium (the test scene): `depth_metric/` and `depthanywhere/` | 360Loc evaluation | 11.4 GB |
   | [`360Loc_train_depth_anywhere.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQD9he-4Yf_rRZMa8lV6LJmbAZNXm9I93JpHB5olfC-NyAc) | concourse, hall, piatrium: `depthanywhere/` | not read by training; for completeness | 0.5 GB |

   SHA-256 in [`SHA256SUMS_depth`](https://1drv.ms/u/c/86d953bfc66eb903/IQDT3HfkB015Rb0PRlu64kRbAcE6QFynTI6p21TSnE5hMOw). The UniK3D prior of the two **training** sets is not
   included (about 250 GB for MP3D and 27 GB for 360Loc); generate it with
   [`tools/prepare_unik3d_depth.py`](tools/prepare_unik3d_depth.py) ([Data](#data)), which is the
   generator that produced the released files.

   ```bash
   cd /path/to/pano_grf && tar -xf pano_grf_evalsets_depth.tar && tar -xf pano_grf_train_depth_anywhere.tar
   cd /path/to/360Loc   && tar -xf 360Loc_atrium_depth.tar
   ```
3. **PanSplat backbone** (stage 1 only): download `pansplat_last.ckpt` ([Weights](#weights)) and set
   `backbone_ckpt` in the pixel configs. The stage-1 model builds its backbone from it, so evaluating the
   stage-1 checkpoint needs it too.

### Evaluate the checkpoints

```bash
C=checkpoints/dev_release_20261004
O=configs/OmniScene
export CUDA_VISIBLE_DEVICES=0
python evaluate.py --dataset mp3d_double_256      --py-config $O/omni_gs_160x320_mp3d_cylinder_all_256.py      --ckpt $C/mp3d_stage3_joint_256x512  --out-dir runs/eval_stage3
python evaluate.py --dataset mp3d_double_512_full --py-config $O/omni_gs_160x320_mp3d_cylinder_all_512x1024.py --ckpt $C/mp3d_stage4_joint_512x1024 --out-dir runs/eval_stage4
python evaluate.py --dataset loc360_double_256_da --py-config $O/omni_gs_160x320_360Loc_cylinder_all_256.py    --ckpt $C/loc360_finetune_256x512    --out-dir runs/eval_loc360
python evaluate.py --dataset mp3d_double_256      --py-config $O/omni_gs_160x320_mp3d_cylinder_pixel_256.py    --ckpt $C/mp3d_stage1_pixel_256x512  --out-dir runs/eval_stage1
python evaluate.py --dataset mp3d_double_256      --py-config $O/omni_gs_160x320_mp3d_cylinder_volume_256.py   --ckpt $C/mp3d_stage2_volume_256x512 --out-dir runs/eval_stage2
```

These are the numbers the downloaded files gave with exactly these commands on an RTX 4090 (another GPU
may differ in the last digit). Validation: `mp3d_double_256_val` (512×1024: `mp3d_double_512_full_val`),
the split the checkpoints were chosen on.

**MP3D, Replica, Residential** (test sets; WS-PSNR per paper column; three-frame protocol, the inputs are
among the scored targets, see [Evaluation](#evaluation)):

| checkpoint | M3D 2.0 m | M3D 1.5 m | M3D 1.0 m | Replica | Residential | total WS-PSNR / SSIM / LPIPS | val WS-PSNR |
|---|---|---|---|---|---|---|---|
| stage 1 (pixel) | 22.387 | 24.969 | 28.767 | 30.601 | 27.925 | 27.481 / 0.9002 / 0.1027 | 26.797 |
| stage 2 (volume) | 21.415 | 22.980 | 25.325 | 25.599 | 27.733 | 24.826 / 0.8161 / 0.2206 | 23.222 |
| **stage 3 (joint, 256×512)** | 23.589 | 25.647 | 29.687 | 31.106 | 28.102 | 28.050 / 0.9044 / 0.1006 | 27.481 |
| **stage 4 (joint, 512×1024)** | 23.212 | 25.498 | 29.848 | 31.087 | 27.541 | 27.776 / 0.9030 / 0.1311 | 27.564 |

Stage 4 is scored at 512×1024 against 512×1024 ground truth, the other stages at 256×512, so its numbers
are not directly comparable with stage 3; the stage-3 checkpoint scored at 512×1024 (no stage-4
training) gives 26.707 total and 24.281 val. Training data have a 1.0 m baseline only; the wide
baselines (1.5 m, 2.0 m) gain least.

**360Loc** (the held-out atrium scene; PanSplat's protocol: 500 samples, inputs 1.4 m apart, the final
weights, no checkpoint selection on the test scene):

| frames scored | WS-PSNR | SSIM | LPIPS | PCC (vs Depth Anywhere) |
|---|---|---|---|---|
| all four (2 inputs + 2 interior), as PanSplat reports | 28.856 | 0.8851 | 0.1009 | 0.8630 |
| interior only (`--novel-only`) | 24.226 | 0.8032 | 0.1643 | 0.8634 |

PanSplat reports 28.14 / 0.860 / 0.127 (WS-PSNR / SSIM / LPIPS) on the same split. The row
`loc360_double_256` gives the same image metrics; its PCC is computed against the UniK3D prior that the
model receives as input and is close to 1 for every checkpoint.

### Train them again

The recipes are the configs in [`configs/OmniScene/release/`](configs/OmniScene/release) (each differs
from the released config only in the values its header lists) and the stage-4 config. Stages 1–3 and
360Loc run on 3 GPUs (24 GB each), stage 4 on 4 × 48 GB.

```bash
export CYLINDERSPLAT_RUNS_ROOT=/path/to/runs
RUNS=$CYLINDERSPLAT_RUNS_ROOT
R=configs/OmniScene/release
L3="accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py"

# stage 1: pixel branch (set backbone_ckpt to pansplat_last.ckpt in the pixel config first)
$L3 --entry mp3d_double_256 --py-config $R/stage1_pixel_256_b4.py --run-id dev_s1 --switch ddp_forward=true
# stage 2: volume branch, from the selected stage-1 checkpoint
$L3 --entry mp3d_double_256 --py-config $R/stage2_volume_256_b4.py --run-id dev_s2 --switch ddp_forward=true \
    --resume-from $RUNS/dev_s1/checkpoint-24000 --transfer stage1_to_stage2
# stage 3: joint, from the selected stage-2 checkpoint
$L3 --entry mp3d_double_256 --py-config $R/stage3_all_256.py --run-id dev_s3 --switch ddp_forward=true \
    --resume-from $RUNS/dev_s2/checkpoint-24000 --transfer stage2_to_stage3
# 360Loc: from the selected stage-3 checkpoint; the final weights are saved as checkpoint-<steps>
$L3 --entry loc360_all_256 --py-config $R/loc360_finetune_256.py --run-id dev_loc360 --save-final \
    --resume-from $RUNS/dev_s3/checkpoint-78000 --transfer mp3d_all_256_to_loc360_pan2 \
    --switch ddp_forward=true --switch loc360_interleave=true --switch depth_valid_mask=true
# stage 4: joint at 512x1024 on 4 GPUs, from the selected stage-3 checkpoint
accelerate launch --config-file configs/accelerate/accel_3proc.yaml --num_processes 4 --gpu_ids 0,1,2,3 train.py \
    --entry mp3d_double_512_ddp4 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py \
    --run-id dev_s4 --resume-from $RUNS/dev_s3/checkpoint-78000 --transfer stage3_to_stage4_512 --switch ddp_forward=true
```

- **Choosing checkpoints.** Evaluate every checkpoint of a stage on the validation row
  (`mp3d_double_256_val`; stage 4: `mp3d_double_512_full_val`) and continue from the best WS-PSNR. The
  step numbers above are those of the release; a rerun can peak at other steps. 360Loc has no
  validation split (its `val` stage is the test scene), so its final weights are the result.
- **Learning rates** were chosen by 5-epoch probes at 2e-4, 4e-4 and 8e-4 on the validation split: 4e-4 for
  stages 1 and 2, 2e-4 for stage 3. 360Loc and stage 4 use the stage-3 lr. Seeds: 0 (the config's) for
  stages 1–4, 1111 for 360Loc.
- **Switches.** `ddp_forward` synchronises gradients across the 3 (or 4) processes; the released
  trainers did not. `loc360_interleave` and `depth_valid_mask` fix two 360Loc training issues (see
  [Opt-in switches](#opt-in-switches)).
- **Time** (training only): stage 1 4 h 52 min, stage 2 7 h 25 min, stage 3 20 h on 3 × RTX 4090;
  360Loc 7 h 14 min on 3 × RTX 4090; stage 4 about 1.3 s per step on 4 × L40 (the release run was
  stopped after 44000 of its 50000 steps, about 16 h).

---

## Training

`train.py --entry <row>` selects a row of [`configs/entries.py`](configs/entries.py). Each row of
the table below (table T) reproduces one legacy trainer; the [screen rows](#phase-2-screen-rows)
reproduce none. The row sets the loader and its order, the scheduler, the forward call, validation
and the **process count**. A launch with a different number of processes exits with an error.

| `--entry` | legacy trainer | processes | launch config |
|---|---|---|---|
| `mp3d_double_256` | `train_mp3d_cylinder_double_256.py` | 3 | `configs/accelerate/accel_3proc.yaml` |
| `mp3d_single_256` | `train_mp3d_cylinder_single_256.py` | 3 | `accel_3proc.yaml` |
| `loc360_all_256` | `train_360Loc_cylinder_double_all_512.py` (reads 256×512) | 3 | `accel_3proc.yaml` |
| `mp3d_double_512` | `train_mp3d_cylinder_double_512.py` | 1 | `accel_1proc.yaml` |
| `mp3d_double_160` | `train_mp3d_cylinder_double.py` | 1 | `accel_1proc.yaml` |
| `kansas_double_160` | `train_vigor_cylinder_double.py` | 1 | `accel_1proc.yaml` |

The launch configs pin `gpu_ids` for the author's machine. Edit them or pass
`accelerate launch --gpu_ids ...`, and add `--main_process_port` when another multi-process
run is active.

Common options:

- `--run-id NAME` puts the run in `$CYLINDERSPLAT_RUNS_ROOT/NAME` ([run directories](#run-directories-and-the-write-guard)). `--work-dir DIR` sets the directory yourself.
- `--resume-from CKPT` starts from the weights in `CKPT` and overrides the config's `resume_from`. The optimizer and scheduler start at step 0. `--no-resume` ignores the config's `resume_from`.
- `--transfer NAME` lists the parameter names the checkpoint may miss or add. The default is `exact`. The load fails if any other name, dtype or shape differs.
- `--switch NAME=VALUE` sets an opt-in switch and can be repeated. `--max-steps N` stops after N steps. `--profile-steps N` times the first N steps. `--save-final` also saves the final weights as `checkpoint-<steps>`.
- `--screen-steps N` and `--seed S` exist only for the [screen rows](#phase-2-screen-rows). A table-T or stage-4 row refuses them and keeps its legacy seeding.

The configs' own `resume_from` values point to the author's checkpoints. Pass
`--resume-from` with your previous stage.

### MP3D two-view: the four-stage schedule

The author's schedule is stage 1, the pixel branch at 256×512, then stage 2, the volume branch at 256×512
(pixel branch frozen), then stage 3, both branches jointly at 256×512, then stage 4, joint at 512×1024
initialised from the stage-3 weights. Stages 1–3:

```bash
export CYLINDERSPLAT_RUNS_ROOT=/path/to/runs
RUNS=$CYLINDERSPLAT_RUNS_ROOT
L3="accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py --entry mp3d_double_256"

# stage 1: pixel branch (set backbone_ckpt in the config first)
$L3 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py --run-id mp3d_pixel_256

# stage 2: volume branch, from the stage-1 weights
$L3 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py --run-id mp3d_volume_256 \
    --resume-from $RUNS/mp3d_pixel_256/checkpoint-36000 --transfer stage1_to_stage2

# stage 3: joint, from the stage-2 weights
$L3 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py --run-id mp3d_all_256 \
    --resume-from $RUNS/mp3d_volume_256/checkpoint-36000 --transfer stage2_to_stage3
```

A checkpoint is written every `save_freq` (3000) steps, and `latest` links to the newest one. The
author's configs chain `checkpoint-36000` of the previous stage (15, 15 and 25 epochs; batch 2 per process).
`stage1_to_stage2` allows the 333 `pixel_gs.mono_depth.*` tensors of the pixel checkpoint,
which the volume model does not have. `stage2_to_stage3` requires identical names.

**Stage 4** (joint, 512×1024, from the stage-3 weights) is the author's recipe for the final
resolution. It keeps the `all_256` model and changes only the image size: a config that is
`all_256` with `resolution = [512, 1024]` (camera, perceptual-loss resolution and
`pixel_gs.image_height` follow it; no parameter shape depends on it). The rows
`mp3d_double_512_ddp3` and `mp3d_double_512_ddp4` (3 or 4 processes) train it on the 512×1024 loader
of `mp3d_double_512` with the recipe of `mp3d_double_256` (OneCycle, loaders first, `.module`
forward, validation). `stage3_to_stage4_512` requires identical names. These rows reproduce no
legacy trainer. The config is
[`omni_gs_160x320_mp3d_cylinder_all_512x1024.py`](configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py)
(batch 1, lr 2e-4, 10 epochs); the `dev` release trained it on 4 × L40 (batch 1 does not fit a 24 GB
GPU for training) and publishes a checkpoint ([Reproducing the dev release](#reproducing-the-dev-release)):

```bash
accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py --entry mp3d_double_512_ddp3 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py --run-id mp3d_all_512x1024 \
    --resume-from $RUNS/mp3d_all_256/checkpoint-48000 --transfer stage3_to_stage4_512 --switch ddp_forward=true
```

For 4 processes, pass `--num_processes 4 --gpu_ids <four GPUs>` to `accelerate launch`. Evaluate a stage-4
checkpoint with `evaluate.py --dataset mp3d_double_512_full` (test) or `mp3d_double_512_full_val`
(validation split). These rows are `mp3d_double_256` and `mp3d_double_256_val` on the 512×1024 loader,
with the full metric set; the `mp3d_double_512` row keeps the reduced metric set of the legacy 512 script.

The repository's `all_512` config builds a different architecture from `all_256`: a strict load of an
`all_256` checkpoint reports 359 missing names, 320 unused names and 26 shape mismatches, and no
transfer into it is defined. The `mp3d_double_512` row exists to reproduce the legacy 512 trainer. With
the `all_512` config, pass `--no-resume`: its `resume_from` (a 160×320 checkpoint) is refused by the
strict rule.

### 360Loc fine-tune

This starts from the MP3D joint model. The 360Loc model (`OmniGaussianCylinderVolume360LocPan2`) blends
the per-view reconstructions by inverse camera distance. The command below is the released recipe; the
`dev` release used `configs/OmniScene/release/loc360_finetune_256.py` with three switches
([Reproducing the dev release](#reproducing-the-dev-release)):

```bash
accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py --entry loc360_all_256 \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py --run-id loc360_all_256 \
    --resume-from $RUNS/mp3d_all_256/checkpoint-36000 --transfer mp3d_all_256_to_loc360_pan2
```

### Single view

The shipped single-view config is the pixel model built with `num_frames=1`. It is fine-tuned
from the two-view pixel checkpoint:

```bash
accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py --entry mp3d_single_256 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py --run-id mp3d_pixel_256_single \
    --resume-from $RUNS/mp3d_pixel_256/checkpoint-36000 --transfer double_pixel_to_single_pixel
```

A model must be built with the number of input views it runs on. Checkpoints load across
view counts because no weight shape depends on it.
`omni_gs_160x320_mp3d_cylinder_all_256_single.py` is the joint model (`OmniGaussianCylinderAll`)
built with `num_frames=1`. It has no legacy trainer and is used by the D3 screen arm below.

### Phase-2 screen rows

These rows run the short fine-tunes that decide which switches go into a full retrain. **They
reproduce no legacy trainer**, so a screen run is not a reproduction of any released result. Each
reuses a table-T row's loader (dataset, `DataLoader` keywords, workers, stage seeds), batch size
and setup order:

| `--entry` | loader of row | processes | launch config | config(s) | used for |
|---|---|---|---|---|---|
| `screen_mp3d_all_256` | `mp3d_double_256` | 1 | `accel_1proc.yaml` | `..._mp3d_cylinder_all_256.py` (`..._volume_256.py` for the `freeze_frozen_bn` pair) | the control and the one-switch MP3D arms |
| `screen_mp3d_single_256` | `mp3d_single_256` | 1 | `accel_1proc.yaml` | `..._mp3d_cylinder_all_256_single.py` | the D3 (`v1_identity_pose`) single-view arms |
| `screen_loc360_all_256` | `loc360_all_256` | 1 | `accel_1proc.yaml` | `..._360Loc_cylinder_all_256.py` | the 360Loc arms (`rotate_gaussians_to_world`) |
| `screen_d1_pair_256` | `mp3d_double_256` | 2 | `accel_2proc.yaml` | `..._mp3d_cylinder_all_256.py` | the `ddp_forward` pair |

A screen run uses the OneCycleLR schedule of the `*_256` rows with `max_lr` = the config's `lr` and
`total_steps` = screen steps + 100. It stops after `--screen-steps` steps (default 6000), whatever
`max_epochs` says, and saves `checkpoint-<steps>`. `--seed` (default 42) replaces both the seed of
`torch.manual_seed` and `cfg.seed` of `set_seed(seed + rank)`; the loader seeds stay. The
one-process rows call the model directly and train on the full loader, not a rank shard.
`screen_d1_pair_256` calls `.module.forward` as the legacy `*_256` trainers do, and the wrapped DDP
model with `--switch ddp_forward=true`. Initialise with `--resume-from` and a `--transfer`
(for example `d3_single_view` for the D3 arm from the two-view joint checkpoint). The single-view
Pan2 arm on 360Loc has no row yet, because there is no single-view 360Loc loader.

```bash
accelerate launch --config-file configs/accelerate/accel_1proc.yaml train.py --entry screen_mp3d_all_256 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py --run-id screen_c0_s43 \
    --resume-from $RUNS/mp3d_all_256/checkpoint-48000 --seed 43
```

---

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset mp3d_double_256 \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
    --ckpt $RUNS/mp3d_all_256/checkpoint-48000 --run-id eval_mp3d_all_256
```

`evaluate.py` runs on one process and uses the GPU chosen with `CUDA_VISIBLE_DEVICES` (on the author's
4090 server use GPUs 1-3; GPU 0 only when it is idle). It reads the
checkpoint in place and writes a log, the dumped config and `metrics.json` to `--out-dir` (or
`$CYLINDERSPLAT_RUNS_ROOT/<run-id>`). The checkpoint must match the model exactly. A missing
checkpoint or a partial load is an error. `--allow-extra PATTERN` (repeatable; an exact name or
`prefix.*`, as in the `--transfer` lists; a bare `*` is refused) skips checkpoint tensors that the
model does not build and whose names match. Missing names, shape mismatches and every other extra
name still stop the run, and the patterns are recorded in `metrics.json` under
`checkpoint.allowed_extra`. `--allow-partial` (same as `--no-strict-load`) evaluates
anyway with the legacy name-and-shape filter. A checkpoint with no matching tensor always fails.

| `--dataset` | legacy script | inputs | scored targets | `--novel-only` |
|---|---|---|---|---|
| `mp3d_double_256` | `evaluate_mp3d_double_256.py` | [0, 2] | [0, 1, 2] | [1] |
| `mp3d_single_256` | `evaluate_mp3d_single_256.py` | [1] | [0, 1, 2] | [0, 2] |
| `mp3d_double_512` | `evaluate_mp3d_double_512.py` | [0, 2] | [0, 1, 2] | [1] |
| `loc360_double_256` | `evaluate_360Loc_double_256.py` | [0, 3] | [0, 1, 2, 3] | [1, 2] |
| `loc360_double_256_da` | – (PCC vs Depth Anywhere, as in the paper) | [0, 3] | [0, 1, 2, 3] | [1, 2] |
| `vigor_double` | `evaluate_VIGOR.py` (Kansas City) | [0, 2] | [0, 1, 2] | [1] |

**Protocol.** The released numbers use the three-frame protocol. All frames of a sequence are
targets, **including the input views**. Each sample is the mean over its targets, and each scene
is the mean over its samples. `--novel-only` scores only the targets that are not inputs,
view by view, before averaging. The 360Loc row prints totals only (its batches carry no scene
key) and needs a checkpoint directory, because it restores the checkpoint with
`accelerator.load_state` as the legacy script did.

**Metrics.** `wspsnr` is the PSNR column of the paper (WS-PSNR, latitude-weighted). `psnr` is
plain PSNR of clipped images. `ssim` is computed with skimage, `lpips` with LPIPS-VGG, and `pcc` is the correlation of the
rendered depth with `depth_anywhere.png`. `abs silog rmse delta1-3` compare the rendered depth with the GT
`depth.png`, without scale alignment (m3d sets only; Replica and Residential print 0.0000).
`depthsim` measures seam continuity (left vs right edge of the rendered depth).

| scene key | paper column | | scene key | paper column |
|---|---|---|---|---|
| `m3d_1.0` | M3D 2.0 m | | `replica_0.5` | Replica 1.0 m |
| `m3d_0.75` | M3D 1.5 m | | `residential_0.15` | Residential ~0.3 m |
| `m3d_0.5` | M3D 1.0 m | | `m3d_0.1`, `m3d_0.25` | not in the paper tables |

Other flags:

- `--save-vis` / `--save-ply` write PNG / PLY per sample. They are off unless given, and they override `eval_args` in the config. `--save-ply` is refused for `loc360_double_256`: that model returns per-view pixel Gaussians, and the legacy script never wrote PLYs for 360Loc.
- `--align-depth` adds lines with median-scale-aligned depth metrics.
- `--fast-ssim` adds a GPU SSIM column. The reported `ssim` stays skimage.
- `--check-ref NAME` (`REF-T1`, `REF-T2`, `REF-T3`, `REF-T4-pixel`, `REF-T4-volume`) compares with the frozen reference in [`repro/`](repro/README.md): every per-scene line, or for the totals-only REF-T3 the Total line. It exits non-zero on any difference. It also adds the reference's `provenance.allowed_extra` patterns to `--allow-extra`. REF-T4-pixel allows `pixel_gs.mono_depth.*`, the 333 tensors its checkpoint holds and the pixel model does not build; the other references allow none.

---

## Opt-in switches

A switch is set in a config (`switches = dict(theta_periodic=True)`) or on the command line
(`--switch theta_periodic=true`, repeatable). The command line wins. The run records its
switches in the dumped config (and in `switches.json` / `metrics.json`). Unknown switches are rejected. A model switch set on a model
class that does not support it is also rejected before the model is built.

**Defaults reproduce the released checkpoints bit for bit. Any switch set to a non-default value changes
training and/or the outputs.** A checkpoint trained with a model switch must be evaluated with the
same switch. `evaluate.py` ignores the training-only switches (marked *train*) and logs that it did.

| Switch | Where | With the switch on | Default (released behaviour) |
|---|---|---|---|
| `ddp_forward` | *train* | forward through the DDP wrapper, so gradients are all-reduced (adds `find_unused_parameters=True`, and a barrier after the main-process save/validation of each step) | `false`: the `*_256` trainers call `model.module.forward`, so each process trains on its own data shard without gradient sync |
| `shuffle_train` | *train*, map-style loaders | a new seeded random order every epoch (a `RandomSampler`; Accelerate shards the batches across processes). Refused for `loc360_all_256`, whose dataset already shuffles | `false`: fixed order |
| `lpips_eval` | *train*, All / Pan2 / Volume / Pixel models | the LPIPS loss network stays in `eval()` (no dropout) | `false` |
| `freeze_frozen_bn` | *train*, stage-2 Volume model | the frozen backbone / pixel branch stay in `eval()`, so their BatchNorm statistics do not drift | `false` |
| `v1_identity_pose` | model, All / Pan2 | single-view camera metas use the identity relative pose | `false`: absolute `w2i` |
| `rotate_gaussians_to_world` | model, pixel heads + volume branch | Gaussian rotations (wxyz) are composed with the camera-to-world rotation, once per Gaussian set | `false`: rotations stay in the reference-camera frame |
| `theta_periodic` | model, volume encoder + decoder | θ is periodic: wrap-around plane attention and sampling, a seam-wrapping colour / depth read, and rz pillars over [0, 2π) | `false` |
| `cell_center_anchor` | model, volume decoder | anchors at cell centres, ±½-cell offsets, r ≥ 0 | `false`: lower-corner anchors, ±1-cell offsets |
| `rgb_retrieval` | model, volume decoder | `visibility_softmax`: a per-view encoder weighted by a softmax over visibility (Gaussian distance − prior depth). Adds 7 parameters, initialised from the current head. Resume with `--transfer d7_visibility_softmax` | `concat`: an MLP over the concatenated per-view windows |
| `prune_opacity` | renderer | drops Gaussians with opacity < τ (0 ≤ τ < 1) before the panorama rasterisation | `0.0` (off) |
| `pixel_depth_sampling` | model, pixel heads | `nearest` interpolation when the pixel branch reads the depth prior | `bilinear` |
| `loc360_interleave` | *train*, 360Loc rows | the train split yields all (sequence, sample) pairs of an epoch in one random order (same samples per sequence and epoch length); frames are cached as uint8 | `false`: one sequence's 1000 samples after another |
| `depth_valid_mask` | *train*, Pan2 model | the depth losses skip pixels whose prior depth is outside (0.45, 50) m | `false`: zero and out-of-range prior depths are supervised |

---

## Run directories and the write guard

Every file a run writes goes under one run directory: `--work-dir` / `--out-dir`, or
`$CYLINDERSPLAT_RUNS_ROOT/<run-id>`. The default root is `/data/qiwei/cylindersplat_dev/runs`,
on the author's machine, so set the variable on yours. `CYLINDERSPLAT_REPRO_ROOT` (default
`/data/qiwei/cylindersplat_repro2`) is the root that reproduction evaluations use. Pass it with
`--out-dir $CYLINDERSPLAT_REPRO_ROOT/<name>`. Run ids use letters, digits, `.`, `_` and `-`.

```
<run>/  <config>.py  <timestamp>.log  switches.json  logs/          config, log, trackers
        checkpoint-N/  latest -> checkpoint-N                     train.py
        validation/step-N/batch-M/                                train.py validation
        profile_steps_rank<r>.json                                train.py --profile-steps
        metrics.json  <iteration>/ (with --save-vis/--save-ply)     evaluate.py
        cwd/                                                      the process working directory
```

- The process moves into `<run>/cwd`, so any relative write (debug images of the older model
  classes) stays inside the run. `train.py` moves there before it builds the model, so the ablation
  models' relative `checkpoints/...` reads need a `checkpoints` link in `<run>/cwd`. `evaluate.py`
  builds the model from the repository root and moves there before the first forward. The LPIPS
  weights are always found from the repository root.
- At start-up the guard resolves every write root, following symlinks. It exits with an error if a root falls
  inside a protected tree. The protected trees are the author's paths on the GPU server
  (`PROTECTED_PREFIXES` in [`tools/write_guard.py`](tools/write_guard.py)), **any path with a
  `workdirs` component**, and the extra prefixes in `CYLINDERSPLAT_PROTECTED` (separated by `:`).
  The author's paths are:
  - the released code: `/home/qiwei/program/cylinderSplat` and `/data/qiwei/home_archive/program/cylinderSplat`;
  - work directories: `/home/qiwei/nips25/workdirs`, `/data/qiwei/nips25/workdirs`,
    `/home/qiwei/ICLR25/workdirs` and `/data/qiwei/ICLR25/workdirs`;
  - datasets: `/data/qiwei/nips25/pano_grf` (MP3D / Replica / Residential), `/data/qiwei/nips25/360Loc`,
    `/data/qiwei/nips25/Kansas` (Kansas City) and `/data/dataset/VIGOR`;
  - the reproduction sweep's output, `/data/qiwei/cylindersplat_repro`.

  On another machine, add your own dataset roots to `CYLINDERSPLAT_PROTECTED`.
- Checkpoints and data are read in place and never written. A resume checkpoint inside the new run's
  directory is refused, and `evaluate.py` refuses an `--out-dir` that is inside or contains the checkpoint directory.

---

## Paper vs code

Where the paper text and the code differ, the code is authoritative. It produced the released
checkpoints, and it is what the defaults reproduce.

- **Reconstruction loss.** The loss is L2 (`loss_args.recon_loss_type='l2'`), not the L1 of Eq. 7. The depth-prior term
  is confidence-weighted, and each config sets its weight (`loss_args.weight_*`). The joint model adds
  masked volume-branch terms.
- **Resolution and schedule.** The paper states 512×1024. The models were trained at 256×512 first
  (stages 1–3, OneCycle schedule, lr 2e-4, 3 processes of batch 2), and then at 512×1024 (stage 4).
- **Multi-GPU training.** The `*_256` trainers do not synchronise gradients, and the MP3D loaders do not shuffle
  (switches `ddp_forward`, `shuffle_train`).
- **Evaluation targets include the inputs** (three-frame protocol above). Use `--novel-only` for
  novel views only.
- **Depth metrics are unaligned.** AbsRel/RMSE/δ compare metric depth directly. The legacy script computed a median
  alignment and discarded it (`--align-depth` prints aligned values as extra lines).
- **Model details** that the switches address: Gaussian rotations stay in the camera frame (App. E says
  global; `rotate_gaussians_to_world`); θ is not periodic in the released code (App. H;
  `theta_periodic`); anchors sit at the cell's lower corner (`cell_center_anchor`); RGB retrieval
  is an MLP over concatenated views rather than the softmax of Eq. 6 (`rgb_retrieval`).
- **Pixel branch.** The pixel branch builds a one-candidate correlation at the prior depth and refines it with a U-Net. Its
  backbone is initialised from a PanSplat checkpoint.

[`legacy/README.md`](legacy/README.md) records how the legacy scripts behave.

---

## Repository layout

| Path | Contents |
|---|---|
| `train.py`, `configs/entries.py` | training entry point and its rows; weight-transfer lists (`TRANSFERS`) |
| `evaluate.py`, `configs/eval_entries.py` | evaluation entry point and its rows |
| `configs/OmniScene/` | model / data / loss configs (`*_256.py`: the released 256×512 models; `*_all_512x1024.py`: stage 4) |
| `configs/OmniScene/release/` | the training recipes of the `dev` release checkpoints |
| `configs/accelerate/` | 1-, 2- and 3-process launch configs |
| `tools/` | `switches.py`, `write_guard.py`, `resume.py`, `metrics.py`, `prepare_unik3d_depth.py` |
| `model/`, `data/`, `builder/` | models, loaders, registry |
| `pano_gaussian/`, `simple-knn/` | CUDA extensions (glm is a submodule) |
| `repro/` | frozen reference numbers of the released code |
| `legacy/` | the original scripts, byte-identical |
| `tests/` | CPU tests: `CUDA_VISIBLE_DEVICES= python -m pytest tests/` (tests marked `gpu` need CUDA) |

---

## Citation

```bibtex
@inproceedings{wang2026cylindersplat,
  title     = {CylinderSplat: 3D Gaussian Splatting with Cylindrical Triplanes for Panoramic Novel View Synthesis},
  author    = {Wang, Qiwei and Ze, Xianghui and Yu, Jingyi and Shi, Yujiao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## Acknowledgements

This code builds on Omni-Scene (the code base; MIT license in [`LICENSE`](LICENSE)), the 3D Gaussian
Splatting rasteriser of Inria GRAPHDECO (`pano_gaussian/`, see its `LICENSE.md`; `simple-knn/`),
PanSplat, TPVFormer, UniMatch, MVSplat, taming-transformers (LPIPS) and UniK3D. We thank their
authors.

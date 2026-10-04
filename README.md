# CylinderSplat

<h3 align="center">[ICLR 2026] 3D Gaussian Splatting with Cylindrical Triplanes for Panoramic Novel View Synthesis</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2603.05882">Paper (arXiv 2603.05882)</a> ·
  <a href="https://github.com/wangqww/CylinderSplat">Code</a>
</p>

CylinderSplat is a feed-forward model that turns one or a few input panoramas into 3D Gaussians and renders new
panoramas directly with an equirectangular rasteriser. It has two branches:

- the **pixel branch** predicts per-pixel Gaussians from a ResNet + multi-view attention backbone and a UniK3D
  depth prior;
- the **volume branch** builds a cylindrical triplane around each input camera, refines it with triplane attention
  and decodes Gaussians that fill regions the inputs do not see.

`train.py` and `evaluate.py` are the only entry points. Each run selects a row of
[`configs/entries.py`](configs/entries.py) (training) or [`configs/eval_entries.py`](configs/eval_entries.py)
(evaluation) and a model config from [`configs/OmniScene/`](configs/OmniScene).

---

## Installation

The models were trained with Python 3.10, CUDA 11.8 and PyTorch 2.1.0. Building the CUDA extensions needs the
CUDA 11.8 toolkit (`nvcc`). Follow this order, which is the one in the header of
[`requirements.txt`](requirements.txt):

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

# 4. pinned Python dependencies
pip install -r requirements.txt

# 5. local CUDA extensions: panorama Gaussian rasteriser and simple-knn
pip install --no-build-isolation ./pano_gaussian ./simple-knn
```

Steps 2, 3 and 5 compile against the torch installed in step 1, so they pass `--no-build-isolation`. Without it,
pip builds in an isolated environment that has no torch: pytorch3d, `pano_gaussian` and `simple-knn` then fail with
`No module named 'torch'`, and mmcv may install without its CUDA ops (`No module named 'mmcv._ext'`).
**With uv**, create the environment with `uv venv -p 3.10` and run every step with `uv pip install` and the same
flags. `pytest` (for `tests/`) is listed as optional at the end of `requirements.txt`.

**Docker.** [`docker/Dockerfile`](docker/Dockerfile) runs the same steps on
`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`. The image has the dependencies and the two extensions but not the
code, so mount the repository:

```bash
git submodule update --init --recursive      # before the build: the image compiles pano_gaussian
docker build -f docker/Dockerfile -t cylindersplat .
docker run --gpus all --ipc=host -it -v "$PWD":/workspace -v /path/to/data:/path/to/data cylindersplat
```

---

## Checkpoints

All files are in the OneDrive folder
[`dev_release_20261004`](https://1drv.ms/f/c/86d953bfc66eb903/IgCCry9vB7yVTbRvR5QKMIhPAWsS880HvFfSe8ABdaKQKZY).
Keep its layout, `checkpoints/dev_release_20261004/<name>/model.safetensors`; each `<name>` directory is what
`evaluate.py --ckpt` and `train.py --resume-from` take.

| `<name>` | model | how it was trained (configs in `configs/OmniScene/`) | checkpoint | download |
|---|---|---|---|---|
| `mp3d_stage1_pixel_256x512` | pixel branch, 256×512 | stage 1, `release/stage1_pixel_256_b4.py` | step 24000 (best on the validation split) | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQA6TUmSBMNRQp6WLuO2BCD3AaNtZGyFP_jYO_Rhlw0wEL4) |
| `mp3d_stage2_volume_256x512` | volume branch, 256×512 | stage 2 from stage 1, `release/stage2_volume_256_b4.py` | step 24000 (best on the validation split) | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQBYPWqJO81rS7U3N0vzZfOaAZzo2wOmxVn4cQGfMgtt9LA) |
| `mp3d_stage3_joint_256x512` | joint model, 256×512 | stage 3 from stage 2, `release/stage3_all_256.py` | step 78000 (best on the validation split) | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQDhstJnrudiQLWJ4RVIianaAbsqTMbhatRobwcAxVh2kqk) |
| `mp3d_stage4_joint_512x1024` | joint model, 512×1024 | stage 4 from stage 3, `omni_gs_160x320_mp3d_cylinder_all_512x1024.py` (run stopped at step 44000 of 50000) | step 21000 | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQBpTaG90iA0S7v0ZOJXg2OkARgXpk5pPwCFZ-lo_9P1DPQ) |
| `loc360_finetune_256x512` | 360Loc model, 256×512 | fine-tuned from stage 3, `release/loc360_finetune_256.py` | the final weights (step 21670) | [model.safetensors](https://1drv.ms/u/c/86d953bfc66eb903/IQAddPv-EkKyQKNV66dYBCueAd5z6pSQyS9rXZSSZbb8VEw) |

SHA-256 of every file: [`SHA256SUMS`](https://1drv.ms/u/c/86d953bfc66eb903/IQCz_-zkkTOBRKfRmvHDpJptAY6_gGv4YIqOqmxWMxR7fPw).

**Training stage 1** needs the PanSplat checkpoint
[`pansplat_last.ckpt`](https://1drv.ms/u/c/86d953bfc66eb903/IQC8bZhj4FWPRJt9U5c0KrmlAR9-KBSQfIIXZWrYh8cOR7c?e=dPhQCt)
(the author's PanSplat run at 256×512; SHA-256 `05b893817f9605d228db20221fbf7b983b35ef9f0473b255afd8409b58782cd1`):
the pixel model (`OmniGaussianCylinderPixel`) initialises its backbone from it when it is built, also for
evaluation. Set `backbone_ckpt='/path/to/pansplat_last.ckpt'` in `model = dict(...)` of the pixel configs
(`omni_gs_160x320_mp3d_cylinder_pixel*.py`, `omni_gs_160x320_VIGOR_cylinder_pixel*.py`).

---

## Data

1. **Images and poses.** Download them as described in PanSplat's
   [Data Preparation](https://github.com/chengzhag/PanSplat?tab=readme-ov-file#-data-preparation): *PanoGRF Data*
   (`pano_grf_lr.tar`: the Matterport3D, Replica and Residential renders) and *360Loc Data* (the official 360Loc
   release).
2. **Derived depth.** The loaders also read a UniK3D depth prior and the Depth Anywhere depth that serves as the
   PCC reference; neither comes with those downloads. The `depth/` subfolder of the checkpoint folder holds the
   author's files. Extract each archive inside its dataset root:

   | archive | contents | needed for |
   |---|---|---|
   | [`pano_grf_evalsets_depth.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQDm8ob5GRE4Q7QejEQELow6AX9HFExXjL5C1dJ0DRlFK9o) (2.2 GB) | test and val sets: `depth_anywhere.png`, `depth_metric.npy`, `depth_conf.npy` per view | evaluation, checkpoint selection |
   | [`pano_grf_train_depth_anywhere.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQCA1xVgPHK2TagBQ0C3dJxfAUNEBgwI1beo-L1sbKGveK0) (3.4 GB) | train set: `depth_anywhere.png` per view | MP3D training |
   | [`360Loc_atrium_depth.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQDGQ7ri97nDRr31a7Km2z1_AdclbMEfpN1vBaNiLP_Kyn0) (11.4 GB) | atrium (the test scene): `depth_metric/`, `depthanywhere/` | 360Loc evaluation |
   | [`360Loc_train_depth_anywhere.tar`](https://1drv.ms/u/c/86d953bfc66eb903/IQD9he-4Yf_rRZMa8lV6LJmbAZNXm9I93JpHB5olfC-NyAc) (0.5 GB) | concourse, hall, piatrium: `depthanywhere/` | not read by training (completeness) |

   SHA-256: [`SHA256SUMS_depth`](https://1drv.ms/u/c/86d953bfc66eb903/IQDT3HfkB015Rb0PRlu64kRbAcE6QFynTI6p21TSnE5hMOw).
   The UniK3D prior of the two training sets is not included (about 250 GB for MP3D and 27 GB for 360Loc);
   generate it with the depth-prior tool below.

   ```bash
   cd /path/to/pano_grf && tar -xf pano_grf_evalsets_depth.tar && tar -xf pano_grf_train_depth_anywhere.tar
   cd /path/to/360Loc   && tar -xf 360Loc_atrium_depth.tar
   ```
3. **Point the loaders at your copies.** The dataset roots are constants in the loaders:

   | loader | constant | default |
   |---|---|---|
   | `data/mp3d_dataloader_{double_256,single_256,double_512,double}.py` | module-level `roots` | `/data/qiwei/nips25/pano_grf` |
   | `data/loc360_dataloader_double_all_512.py` | `root` in `Dataset360Loc.__init__` | `/data/qiwei/nips25/360Loc` |
   | `data/vigor_dataloader_double.py` (Kansas City) | `root_dir` in `load_VIGOR_data` and in the `kansas_double_160` row of `configs/entries.py` | `/data/qiwei/nips25/` |

**Layouts.** Matterport3D / Replica / Residential (PanoGRF renders, three views per sequence; the two-view loaders
take views [0, 2] as inputs, the single-view loader view [1]):

```
<pano_grf>/png_render_{train,val}_1024x512_seq_len_3_m3d_dist_0.5/<scene>/<view>/
<pano_grf>/png_render_test_1024x512_seq_len_3_<name>_dist_<d>/<scene>/<view>/
        name_dist_d in: m3d_dist_{0.1,0.25,0.5,0.75,1.0}, residential_dist_0.15, replica_dist_0.5
<view>/ rgb.png             panorama
        rot.txt, tran.txt   camera rotation / translation
        depth.png           GT depth in mm (m3d sets)
        depth_anywhere.png  Depth Anywhere depth (the PCC reference)
        depth_metric.npy    UniK3D metric distance  } the depth prior
        depth_conf.npy      UniK3D confidence       }
```

360Loc (`train`: concourse, hall, piatrium; `val`: atrium, the test scene):

```
<360Loc>/<location>/{mapping,query_360}/<sequence with "360" in its name>/
        camera_pose.json                                   {frame file name: 4x4 camera-to-world pose}
        image/<frame>.jpg
        depth_metric/<frame>_depth.npy, <frame>_conf.npy   the UniK3D prior
        depthanywhere/<frame>_depth_anywhere.png           Depth Anywhere depth (the PCC reference)
```

Kansas City street-view panoramas (paper App. F; called `VIGOR` in the code) need
`Kansas/{train_list.txt,test_list_om.txt}` and, per road, `Ground/`, `Satellite/`, `GrdInSat_dist_dir_month_new/`
and `depth_metric/`.

**Depth prior.** [`tools/prepare_unik3d_depth.py`](tools/prepare_unik3d_depth.py) generates the UniK3D prior
(ViT-L, resolution level 9, 1024×512 spherical camera) that the released files were made with. It writes a mirrored
tree under `--out-root` that you merge into your data copy (`--in-place` writes next to the images instead). UniK3D
needs Python ≥ 3.11, torch ≥ 2.4 and numpy ≥ 2, so give it its own environment and run the tool from the
repository root with that environment's Python:

```bash
python3.11 -m venv ~/venvs/unik3d            # or: uv venv -p 3.11 ~/venvs/unik3d
git clone https://github.com/lpiccinelli-eth/UniK3D ../UniK3D
~/venvs/unik3d/bin/pip install -e ../UniK3D --extra-index-url https://download.pytorch.org/whl/cu121

U=~/venvs/unik3d/bin/python
$U tools/prepare_unik3d_depth.py --dataset mp3d   --data-root /path/pano_grf --out-root /path/pano_grf_depth
$U tools/prepare_unik3d_depth.py --dataset loc360 --data-root /path/360Loc   --out-root /path/360Loc_depth
rsync -a /path/pano_grf_depth/ /path/pano_grf/      # merge into your copy
```

Other options: `--stages` (MP3D splits, default `train val test`), `--batch-size`, `--skip-existing`, `--unik3d`
(hub id or local directory of the weights).

---

## Training

`train.py --entry <row>` selects a row of [`configs/entries.py`](configs/entries.py). The row fixes the loader, the
scheduler, the forward call, validation and the number of processes (a launch with another process count stops).

| `--entry` | data | processes | configs |
|---|---|---|---|
| `mp3d_double_256` | MP3D two-view, 256×512 | 3 | `omni_gs_160x320_mp3d_cylinder_{pixel,volume,all}_256.py`, `release/stage{1,2,3}_*.py` |
| `mp3d_double_512_ddp3`, `mp3d_double_512_ddp4` | MP3D two-view, 512×1024 (stage 4) | 3 / 4 | `omni_gs_160x320_mp3d_cylinder_all_512x1024.py` |
| `mp3d_single_256` | MP3D single view, 256×512 | 3 | `omni_gs_160x320_mp3d_cylinder_{pixel,all}_256_single.py` |
| `loc360_all_256` | 360Loc, 256×512 | 3 | `omni_gs_160x320_360Loc_cylinder_all_256.py`, `release/loc360_finetune_256.py` |
| `mp3d_double_512` | MP3D two-view, 512×1024, the `all_512` architecture | 1 | `omni_gs_160x320_mp3d_cylinder_{pixel,all}_512.py` |
| `mp3d_double_160` | MP3D two-view, 160×320 (ablations) | 1 | the other `omni_gs_160x320_mp3d_*.py` configs |
| `kansas_double_160` | Kansas City, 160×320 | 1 | `omni_gs_160x320_VIGOR_cylinder_*.py` |

Launch with the matching config in [`configs/accelerate/`](configs/accelerate) (`accel_3proc.yaml` or
`accel_1proc.yaml`; pass `--gpu_ids` and, for 4 processes, `--num_processes 4`). Options:

- `--run-id NAME` writes the run to `$CYLINDERSPLAT_RUNS_ROOT/NAME` (default `workdirs/NAME` in the repository),
  `--work-dir DIR` to `DIR`;
- `--resume-from CKPT` starts from the weights in `CKPT` (optimizer and scheduler start fresh); `--transfer NAME`
  names the parameters the checkpoint may miss or add (default `exact`; any other difference stops the load);
  `--no-resume` ignores the config's `resume_from`;
- `--switch NAME=VALUE` sets one of the switches below; `--save-final` also saves the final weights as
  `checkpoint-<steps>`; `--max-steps N` stops after N steps.

**Switches** (off by default; set in a config with `switches = dict(...)` or with `--switch`):

| switch | effect |
|---|---|
| `ddp_forward` | forward through the DDP wrapper, so gradients are synchronised across processes (adds `find_unused_parameters=True`); without it each process trains its own replica and only rank 0 is saved |
| `loc360_interleave` | 360Loc: the training stream mixes all sequences instead of one sequence after another |
| `depth_valid_mask` | 360Loc model: the depth loss skips prior depths outside (0.45, 50) m |

### MP3D two-view (stages 1–4)

Stage 1 trains the pixel branch, stage 2 the volume branch with the pixel branch frozen, stage 3 both jointly, all
at 256×512; stage 4 continues the joint model at 512×1024. These are the commands of the released checkpoints:

```bash
export CYLINDERSPLAT_RUNS_ROOT=/path/to/runs
RUNS=$CYLINDERSPLAT_RUNS_ROOT
R=configs/OmniScene/release
L3="accelerate launch --config-file configs/accelerate/accel_3proc.yaml train.py"

# stage 1: pixel branch (set backbone_ckpt first, see Checkpoints)
$L3 --entry mp3d_double_256 --py-config $R/stage1_pixel_256_b4.py --run-id s1 --switch ddp_forward=true
# stage 2: volume branch, from the selected stage-1 checkpoint
$L3 --entry mp3d_double_256 --py-config $R/stage2_volume_256_b4.py --run-id s2 --switch ddp_forward=true \
    --resume-from $RUNS/s1/checkpoint-24000 --transfer stage1_to_stage2
# stage 3: joint, from the selected stage-2 checkpoint
$L3 --entry mp3d_double_256 --py-config $R/stage3_all_256.py --run-id s3 --switch ddp_forward=true \
    --resume-from $RUNS/s2/checkpoint-24000 --transfer stage2_to_stage3
# stage 4: joint at 512x1024 on 4 GPUs (batch 1 needs about 26 GB per GPU)
accelerate launch --config-file configs/accelerate/accel_3proc.yaml --num_processes 4 --gpu_ids 0,1,2,3 train.py \
    --entry mp3d_double_512_ddp4 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512x1024.py \
    --run-id s4 --resume-from $RUNS/s3/checkpoint-78000 --transfer stage3_to_stage4_512 --switch ddp_forward=true
```

Choose the checkpoint of each stage by evaluating every saved checkpoint on the validation split
(`mp3d_double_256_val`; stage 4: `mp3d_double_512_full_val`) and continue from the best WS-PSNR; the step numbers
above are those of the released runs. The learning rates (4e-4 for stages 1–2, 2e-4 for stages 3–4) were chosen by
short probes on the validation split. On 3 × RTX 4090, stages 1, 2 and 3 take about 5, 7.5 and 20 hours.

### 360Loc fine-tune

From the selected stage-3 checkpoint; 360Loc has no validation split (its `val` stage is the test scene), so the
final weights are the result:

```bash
$L3 --entry loc360_all_256 --py-config $R/loc360_finetune_256.py --run-id loc360 --save-final \
    --resume-from $RUNS/s3/checkpoint-78000 --transfer mp3d_all_256_to_loc360_pan2 \
    --switch ddp_forward=true --switch loc360_interleave=true --switch depth_valid_mask=true
```

### Single view

The single-view configs build the pixel model (or the joint model) with `num_frames=1`, fine-tuned from the
two-view weights (no weight shape depends on the number of views):

```bash
$L3 --entry mp3d_single_256 --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py \
    --run-id single --resume-from $RUNS/s1/checkpoint-24000 --transfer double_pixel_to_single_pixel
```

### Ablations and Kansas City

The 160×320 rows train the ablation models (UniFuse / Depth Anywhere depth priors, Cartesian and spherical
triplanes, the density and cluster variants) and the Kansas City models on one process:

```bash
L1="accelerate launch --config-file configs/accelerate/accel_1proc.yaml train.py"
$L1 --entry mp3d_double_160   --py-config configs/OmniScene/omni_gs_160x320_mp3d_decare_volume.py --run-id decare
$L1 --entry kansas_double_160 --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_all.py  --run-id kansas
```

The UniFuse and Depth Anywhere models read their depth networks from `checkpoints/` relative to the working
directory.

---

## Evaluation

```bash
C=checkpoints/dev_release_20261004
O=configs/OmniScene
python evaluate.py --dataset mp3d_double_256      --py-config $O/omni_gs_160x320_mp3d_cylinder_all_256.py      --ckpt $C/mp3d_stage3_joint_256x512  --out-dir runs/eval_stage3
python evaluate.py --dataset mp3d_double_512_full --py-config $O/omni_gs_160x320_mp3d_cylinder_all_512x1024.py --ckpt $C/mp3d_stage4_joint_512x1024 --out-dir runs/eval_stage4
python evaluate.py --dataset loc360_double_256_da --py-config $O/omni_gs_160x320_360Loc_cylinder_all_256.py    --ckpt $C/loc360_finetune_256x512    --out-dir runs/eval_loc360
python evaluate.py --dataset mp3d_double_256      --py-config $O/omni_gs_160x320_mp3d_cylinder_pixel_256.py    --ckpt $C/mp3d_stage1_pixel_256x512  --out-dir runs/eval_stage1
python evaluate.py --dataset mp3d_double_256      --py-config $O/omni_gs_160x320_mp3d_cylinder_volume_256.py   --ckpt $C/mp3d_stage2_volume_256x512 --out-dir runs/eval_stage2
```

`evaluate.py` runs on one GPU (`CUDA_VISIBLE_DEVICES`) and writes a log, the dumped config and `metrics.json` to
`--out-dir`. The checkpoint must match the model exactly; `--allow-extra PATTERN` skips named extra tensors and
`--allow-partial` loads with a name-and-shape filter.

| `--dataset` | data | inputs | scored targets | `--novel-only` |
|---|---|---|---|---|
| `mp3d_double_256` / `mp3d_double_256_val` | MP3D, Replica, Residential test sets / MP3D validation split, 256×512 | [0, 2] | [0, 1, 2] | [1] |
| `mp3d_double_512_full` / `mp3d_double_512_full_val` | the same at 512×1024 | [0, 2] | [0, 1, 2] | [1] |
| `mp3d_double_512` | 512×1024, reduced metric set | [0, 2] | [0, 1, 2] | [1] |
| `mp3d_single_256` | single view, 256×512 | [1] | [0, 1, 2] | [0, 2] |
| `loc360_double_256_da` | 360Loc atrium, 256×512; PCC against Depth Anywhere | [0, 3] | [0, 1, 2, 3] | [1, 2] |
| `loc360_double_256` | the same, PCC against the UniK3D prior | [0, 3] | [0, 1, 2, 3] | [1, 2] |
| `vigor_double` | Kansas City | [0, 2] | [0, 1, 2] | [1] |

**Protocol.** All frames of a sequence are targets, including the input views; each sample is the mean over its
targets and each scene the mean over its samples. `--novel-only` scores only the views that are not inputs. The
360Loc rows follow PanSplat: 500 samples, inputs three frames (about 1.4 m) apart, totals only.

**Metrics.** `wspsnr` is the PSNR of the paper (latitude-weighted WS-PSNR); `psnr` is plain PSNR of clipped images;
`ssim` (skimage) and `lpips` (VGG); `pcc` is the correlation of the rendered depth with the Depth Anywhere depth;
`abs silog rmse delta1-3` compare the rendered depth with the GT `depth.png` without scale alignment (m3d sets only);
`depthsim` measures seam continuity. Scene keys and paper columns: `m3d_1.0` = M3D 2.0 m, `m3d_0.75` = 1.5 m,
`m3d_0.5` = 1.0 m, `replica_0.5` = Replica, `residential_0.15` = Residential.

Other flags: `--save-vis` / `--save-ply` (PNG / PLY per sample), `--align-depth` (adds median-aligned depth
metrics), `--fast-ssim` (adds a GPU SSIM column).

---

## Repository layout

| path | contents |
|---|---|
| `train.py`, `configs/entries.py` | training entry point, its rows and the weight-transfer lists |
| `evaluate.py`, `configs/eval_entries.py` | evaluation entry point and its rows |
| `configs/OmniScene/` | model configs; `release/` holds the recipes of the released checkpoints |
| `configs/accelerate/` | launch configs for 1, 2 and 3 processes |
| `model/`, `data/`, `builder/` | models, loaders, model registry |
| `tools/` | switches, checkpoint loading rules, metrics, the depth-prior tool |
| `pano_gaussian/`, `simple-knn/` | CUDA extensions (glm is a submodule) |
| `tests/` | CPU tests: `CUDA_VISIBLE_DEVICES= python -m pytest tests/` |

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

This code builds on Omni-Scene (the code base; MIT license in [`LICENSE`](LICENSE)), the 3D Gaussian Splatting
rasteriser of Inria GRAPHDECO (`pano_gaussian/`, see its `LICENSE.md`; `simple-knn/`), PanSplat, TPVFormer, UniMatch,
MVSplat, taming-transformers (LPIPS) and UniK3D. We thank their authors.

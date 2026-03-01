<h1 align=center font-weight:100> [CVPR2025] <strong><i>Omni-Scene</i></strong>⚡️: Omni-Gaussian Representation for Ego-Centric Sparse-View Scene Reconstruction</h1>

<p align="center">
    <a href="weidongxu@westlake.edu.cn">Dongxu Wei</a>
    &nbsp;·&nbsp;
    <a href="lizhiqi49@westlake.edu.cn">Zhiqi Li</a>
    &nbsp;·&nbsp;
    <a href="https://ethliup.github.io/">Peidong Liu*</a>
    <h3 align="center"><a href="https://arxiv.org/abs/2412.06273">Paper</a> | <a href="https://wswdx.github.io/omniscene">Project Page</a> </h3>
  </p>

This as an official implementation of our CVPR 2025 paper 
[**Omni-Scene**: Omni-Gaussian Representation for Ego-Centric Sparse-View Scene Reconstruction](https://wswdx.github.io/omniscene), based on the [mmdet3d](https://github.com/open-mmlab/mmdetection3d) framework and [accelerate](https://github.com/huggingface/accelerate) library.
In this repository, we provide our code, along with our pre-processed nuScenes dataset, which is specially reformulated for ego-centric scene reconstruction.

<div align="center">
  <img src="assets/figures/omniscene.jpg" alt=""  width="1100" />
</div>

## News
- 2025/2/27: Our paper is accepted by CVPR 2025!
- 2025/3/3: Code and data are now available. If you find this repository useful, please give us a star🌟!

## TODO
- [X] Upload code and data
- [X] Upload subset of our data for fast demo running
- [ ] Add tutorial for 3D generation
- [ ] Add code and documents for Waymo dataset

## Demo (3D Reconstruction)

Reconstruct scenes with 3D Gaussians given 6-view images from [nuScenes](https://www.nuscenes.org/) in a feed-forward manner:

https://github.com/user-attachments/assets/99732d4d-613b-43bf-9476-a6b42dfe5301

## Demo (3D Generation)

Generate scenes with 3D Gaussians by combining our method with 2D multi-view diffusion model [MagicDrive](https://github.com/cure-lab/MagicDrive):

https://github.com/user-attachments/assets/dba0bb77-31ae-47df-a1f5-4abe5b96c87d

## Get started

### 1. Installation

```bash
# (Optional) create a fresh conda env
conda create --name omniscene -y "python==3.10"
conda activate omniscene

# install dependencies
pip install --upgrade pip setuptools

## install pytorch (CUDA 11.8)
pip install "torch==2.1.0+cu118" "torchvision==0.16.0+cu118" torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
## install pytorch (CUDA 12.1)
pip install "torch==2.1.0+cu121" "torchvision==0.16.0+cu121" torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

## install 3DGS rasterizer (w/ depth)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install diff-gaussian-rasterization

## common dependencies
pip install -r requirements.txt

## compile mmcv-2.1.0 and install mmdet3d-1.4.0
pip install -U openmim
pip install mmengine
pip install ninja psutil
git clone https://github.com/open-mmlab/mmcv.git
git checkout v2.1.0
MAX_JOBS=16 MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9+PTX" pip install -e . -v
mim install 'mmdet>=3.0.0'
mim install 'mmdet3d==1.4.0'
```
You can refer to [MMLab documents](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) for details about mmcv and mmdet3d installation.

### 2. Prepare the dataset
We have provided a pre-processed dataset on [OneDrive](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EleccAyvf8RFimGfDqqThzsBNxA3SS0o86eHN4nSUd1TQg), and [BaiduYun](https://pan.baidu.com/s/1iM5RDj3_Q4jdb0a5hZo-CA) (extract code: qbck). You can download and extract the compressed data files as follows.
```bash
cat dataset_omniscene_part* > dataset_omniscene.tar
tar -xvf dataset_omniscene.tar
rm -rf *.tar
rm -rf *part*
mv dataset_omniscene {ROOT}/data/nuScenes
```
We also have provided a subset of the above full data on [GoogleDrive](https://drive.google.com/file/d/1P0CuGHBXMfCNDAlq2SbZgBM1TF8YGXKY/view?usp=sharing) for convenience in running demonstration and checking data formulation.

Put the extracted files under {ROOT}/data, and the data should be structured like this:
```bash
{ROOT}/data/nuScenes
├── samples_small
├── samples_dpt_small
├── samples_dptm_small
├── samples_param_small
├── sweeps_small
├── sweeps_dpt_small
├── sweeps_dptm_small
├── sweeps_param_small
├── interp_12Hz_trainval
```

### 3. Running demo

This command will generate and save 360 degree exploring videos for the reconstructed 3D scenes:

```bash
accelerate launch --config-file accelerate_config.yaml demo.py \
    --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
    --output-dir outputs/omni_gs_nusc_novelview_r50_224x400_vis \
    --load-from checkpoints/checkpoint-100000
```
where
- `--config-file accelerate_config.yaml` is the relative path of accelrate configuration file;
- `--py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py"` is the relative path of Omni-Scene's configuration file;
- `--output-dir` is the relative path of output directory. We save the rendered videos here.
- `--load-from` is the relative path of model weights that you want to use.

> Note1: >=1 A100 GPUs are required to run the demo of our full method.

> Note2: you can only download the subset of our full data for fast running demo.
>

### 4. Training

The training script is as follows. We have released our pre-trained weights [here](https://drive.google.com/drive/folders/1vgc8VjXhuo35KwFqbJiqdu5FEDg6AMRy?usp=sharing).

```bash
accelerate launch --config-file accelerate_config.yaml train.py \
    --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
    --work-dir workdirs/omni_gs_nusc_novelview_r50_224x400 \
    --resume-from path/to/checkpoints
```
where
- `--config-file accelerate_config.yaml` is the relative path of accelrate configuration file;
- `--py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py"` is the relative path of Omni-Scene's configuration file;
- `--work-dir` is the relative path of experiment work directory. We save logs, checkpoints, visualizations, plys here;
- (optional) `--resume-from` is the relative path of checkpoints that you want to resume from. You should delete this argument for training from scratch.

> Note: >=2 A100 GPUs are required to run the training of our full method.
>

### 5. Evaluation

The evaluation script is as follows.

```bash
accelerate launch --config-file accelerate_config.yaml evaluate.py \
    --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
    --output-dir outputs/omni_gs_nusc_novelview_r50_224x400 \
    --load-from checkpoints/checkpoint-100000
```
where
- `--config-file accelerate_config.yaml` is the relative path of accelrate configuration file;
- `--py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py"` is the relative path of Omni-Scene's configuration file;
- `--output-dir` is the relative path of output directory. We save the results and visualizations here.
- `--load-from` is the relative path of model weights that you want to evaluate.

> Note: >=1 A100 GPUs are required to run the evaluation of our full method.
>

## Citation

If you find this useful, please consider citing:

```bibtex
@inproceedings{wei2024omniscene,
    author = {Wei, Dongxu and Li, Zhiqi and Liu, Peidong},
    title = {Omni-scene: omni-gaussian representation for ego-centric sparse-view scene reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2025}
}
```

## Acknowledgments

Omni-Scene is based on [mmdetection3d]((https://github.com/open-mmlab/mmdetection3d)). It is also greatly inspired by the following outstanding contributions to the open-source community: [TPVFormer](https://github.com/wzzheng/TPVFormer), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [MVSplat](https://github.com/donydchen/mvsplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [Metric3D-v2](https://github.com/YvanYin/Metric3D) and [MagicDrive](https://github.com/cure-lab/MagicDrive).
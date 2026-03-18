# CylinderSplat

<h3 align="center">[ICLR 2026] 3D Gaussian Splatting with Cylindrical Triplanes for Panoramic Novel View Synthesis</h3>

CylinderSplat is a 3D Gaussian Splatting framework for panoramic novel view synthesis.

---

## Environment Setup

> It is recommended to create an isolated Python environment first (e.g., Conda or venv).

### 0) Download pretrained checkpoint

Please download the pretrained checkpoint from:

https://1drv.ms/u/c/86d953bfc66eb903/IQC8bZhj4FWPRJt9U5c0KrmlAR9-KBSQfIIXZWrYh8cOR7c?e=dPhQCt

Then move the downloaded file to the [checkpoints/](checkpoints/) directory.

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Install `pano_gaussian`

```bash
cd pano_gaussian
pip install .
cd ..
```

---

## Training (MP3D, Double Input, 256 Resolution)

Double-input training consists of three stages. Run them in order.

### Stage 1: Pixel Branch

```bash
accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256
```

### Stage 2: Volume Branch

```bash
accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256
```

### Stage 3: Joint Training (All)

```bash
accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256
```

---

## Project Structure (Key Paths)

- Training configs: [configs/OmniScene/](configs/OmniScene/)
- Data loaders: [data/](data/)
- Example training script collection: [run2.sh](run2.sh)
- Additional datasets (360Loc / VIGOR): use the corresponding `train_*.py` and `evaluate_*.py` scripts in the project root.

---

## Notes

- Ensure the checkpoint file is placed under [checkpoints/](checkpoints/) before training.
- If needed, adjust `--work-dir` to your local output path.
- For reproducibility, keep your environment and package versions consistent with [requirements.txt](requirements.txt).

---

## TODO

- [ ] Add dataset preparation instructions
- [ ] Add evaluation command examples
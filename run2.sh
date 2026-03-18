### random
# train
python train_mp3d_cylinder_double_random.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
     --work-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_random
python train_mp3d_cylinder_double_random.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random
python train_mp3d_cylinder_double_random.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_random


# eval
python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random \
    --load-from "checkpoint-36000"

python evaluate_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random \
    --load-from "checkpoint-36000"

python evaluate_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --output-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_random \
    --load-from "checkpoint-9000"

python evaluate_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_random \
    --load-from "checkpoint-21000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_random \
    --load-from "checkpoint-54000"

### double
# train
python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
     --work-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel

# 256*512 pixel train
python train_mp3d_cylinder_double_256.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py \
     --work-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256

# 256*512 volume train
python train_mp3d_cylinder_double_256.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py \
     --work-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256

# 256*512 all train
python train_mp3d_cylinder_double_256.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
     --work-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256

# 256*512 all train All
python train_360Loc_cylinder_double_all_512.py \
     --py-config configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py \
     --work-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256

# accelerate 256*512 volume train
accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256

# accelerate 256*512 pixel train
accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256

accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_single_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256_single.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_single_pixel_256

# accelerate 256*512 all train
accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256_new

# accelerate 256*512 all train 360Loc
accelerate launch --config-file accelerate_config.yaml train_360Loc_cylinder_double_all_512.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256


python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_density.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_density

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_unifuse.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_unifuse

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_unifuse.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_unifuse

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_depthanywhere.py \
     --work-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_depthanywhere

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_unifuse.py \
     --work-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_unifuse

python train_mp3d_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all

python train_360Loc_cylinder_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All2_depthanywhere"

python train_360Loc_cylinder_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_pixel.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Pixel"


# eval

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --output-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py \
    --output-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_256 \
    --load-from "checkpoint-36000"

python evaluate_mp3d_single_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_256.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_single_pixel_256 \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py \
    --output-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256 \
    --load-from "checkpoint-36000"

python evaluate_mp3d_single_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_256.py \
    --output-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_256 \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
    --output-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256 \
    --load-from "checkpoint-48000"

python evaluate_mp3d_single_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py \
    --output-dir /home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256 \
    --load-from "checkpoint-48000"


python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_unifuse.py \
    --output-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_unifuse \
    --load-from "checkpoint-36000"


python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume \
    --load-from "checkpoint-36000"         # 还没收敛，训练时间越长效果越好


python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_unifuse.py \
    --output-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_unifuse \
    --load-from "checkpoint-3000"         # 还没收敛，训练时间越长效果越好

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume_density.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_density \
    --load-from "checkpoint-3000"         # 还没收敛，训练时间越长效果越好

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_depthanywhere.py \
    --output-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_depthanywhere \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_unifuse.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_unifuse \
    --load-from "checkpoint-18000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_density.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_density \
    --load-from "checkpoint-12000"

python evaluate_mp3d_double_512.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_512.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all \
    --load-from "checkpoint-39000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All1" \
    --load-from "checkpoint-21000"

python evaluate_360Loc_double_256.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all_256.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_cylinder_double_all_256 \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All3" \
    --load-from "checkpoint-21000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_pixel.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Pixel" \
    --load-from "checkpoint-21000"

# VIGOR
python train_vigor_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_all.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_all

python train_vigor_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_pixel.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_pixel


python train_vigor_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_pixel_unifuse.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_pixel_unifuse

python train_vigor_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_volume_decare.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_all

python train_vigor_cylinder_double.py \
     --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_volume_spherical.py \
     --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_all

python evaluate_VIGOR.py \
    --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_all.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_all \
    --load-from "checkpoint-27000"

python evaluate_VIGOR.py \
    --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_pixel_unifuse.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_pixel_unifuse \
    --load-from "checkpoint-6000"
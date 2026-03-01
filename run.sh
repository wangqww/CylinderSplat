accelerate launch --config-file accelerate_config.yaml train_vigor.py \
    --py-config configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py \
    --work-dir workdirs/omni_gs_nusc_novelview_r50_224x400

### VIGOR

python train_vigor.py \
    --py-config configs/OmniScene/omni_gs_cube_160x320.py \
    --work-dir workdirs/omni_gs_cube_160x320

# vigor panorama
python train_vigor.py \
    --py-config configs/OmniScene/omni_gs_160x320.py \
    --work-dir workdirs/omni_gs_160x320
# vigor cube
python train_vigor.py \
    --py-config configs/OmniScene/omni_gs_cube_160x320.py \
    --work-dir workdirs/omni_gs_cube_160x320


### MP3D

python train_mp3d.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d.py \
    --work-dir workdirs/omni_gs_160x320_mp3d_new

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all1

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --work-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel

python train_mp3d_cylinder_double_random.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --work-dir /home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_random

python train_mp3d_cylinder_double_random.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume_random

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_decare_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_decare_double_volume

python train_mp3d_cylinder_double_random.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_random1

python train_mp3d_cylinder_trible.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_trible_pixel.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_trible_pixel

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_decare_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_decare_double_volume

python train_mp3d_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_volume.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_spherical_double_volume

accelerate launch --config-file accelerate_config.yaml train_mp3d_cylinder_double_512.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_512.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_512

### VIGOR

python train_vigor_cylinder_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_VIGOR_cylinder_pixel.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_VIGOR_cylinder_double_pixel


### 360Loc

python train_360Loc_cylinder_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All1"

python train_360Loc_cylinder_double_random.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All_random"

python train_360Loc_cylinder_double_pixel.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_pixel.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Pixel_new"

python train_360Loc_cylinder_double_volume.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume"


python train_360Loc_cylinder_double_volume_pan2.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan2.py" \
    --work-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2_fix"

python train_360Loc.py \
    --py-config configs/OmniScene/omni_gs_160x320_360Loc.py \
    --work-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Spherical

# Eval MP3D

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_double.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double" \
    --load-from "checkpoint-33000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_pan" \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel2 \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume1" \
    --load-from "checkpoint-18000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all1" \
    --load-from "checkpoint-36000"

python evaluate_mp3d_double_512.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_512.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_512" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py \
    --output-dir "/home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_decare_double_volume_pan" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_decare_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_decare_double_volume_random2 \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_spherical_volume.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_spherical_double_volume \
    --load-from "checkpoint-27000"

python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py" \
    --output-dir "/home/qiwei/ICLR25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel_random" \
    --load-from "checkpoint-27000"

python evaluate_mp3d_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_pixel1" \
    --load-from "checkpoint-36000"

python evaluate_mp3d_trible.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_trible_pixel.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_trible_pixel" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_volume.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_volume" \
    --load-from "checkpoint-27000"

python evaluate_mp3d_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_decare_volume.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_decare_double_volume" \
    --load-from "checkpoint-15000"


python evaluate_mp3d_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_spherical_volume.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_spherical_double_volume" \
    --load-from "checkpoint-15000"


python evaluate_mp3d.py \
    --py-config "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_pixel_volume.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_pixel" \
    --load-from "checkpoint-3000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all1" \
    --load-from "checkpoint-45000"

python evaluate_mp3d_double.py \
    --py-config configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all.py \
    --output-dir /data/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all1 \
    --load-from "checkpoint-9000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_all.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All1" \
    --load-from "checkpoint-21000"

python evaluate_360Loc_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan2.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double_all.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_volume_pan2.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_Volume_Pan2" \
    --load-from "checkpoint-3000"

python evaluate_360Loc_double.py \
    --py-config "configs/OmniScene/omni_gs_160x320_360Loc_cylinder_double_all.py" \
    --output-dir "/data/qiwei/nips25/workdirs/omni_gs_160x320_360Loc_Cylinder_Double_All_Volume" \
    --load-from "checkpoint-3000"
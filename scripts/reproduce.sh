#!/usr/bin/env bash
# Reproduce the evaluation of the five released checkpoints with one command:
#
#   bash scripts/reproduce.sh [--gpus 0,1,2] [--only NAME,...] [--data-dir DIR] [--out-dir DIR]
#
# 1. downloads the checkpoints, the PanSplat backbone and the evaluation data (about 20 GB) from the
#    Hugging Face dataset eacsai/CylinderSplat; set HF_ENDPOINT=https://hf-mirror.com to use the mirror;
# 2. checks every downloaded file against SHA256SUMS;
# 3. extracts the evaluation data into DIR/pano_grf and DIR/360Loc (about 20 GB more);
# 4. evaluates each checkpoint on one GPU, the GPUs of --gpus in parallel, and prints a summary next to
#    the numbers this code gives on an RTX 4090.
#
# Run it from the environment of the README "Installation" section. Re-running skips finished steps: a
# checkpoint whose OUT_DIR/<name>/metrics.json exists is not evaluated again.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HF_REPO=eacsai/CylinderSplat
DATA_DIR=$REPO/hf_data
OUT_DIR=$REPO/workdirs/reproduce
GPUS=${CUDA_VISIBLE_DEVICES:-0}
ONLY=
PYTHON=${PYTHON:-python}

# name | evaluate.py --dataset | config in configs/OmniScene (the 360Loc run is the longest; it goes first)
JOBS=(
  "loc360_finetune_256x512|loc360_double_256_da|omni_gs_160x320_360Loc_cylinder_all_256.py"
  "mp3d_stage4_joint_512x1024|mp3d_double_512_full|omni_gs_160x320_mp3d_cylinder_all_512x1024.py"
  "mp3d_stage3_joint_256x512|mp3d_double_256|omni_gs_160x320_mp3d_cylinder_all_256.py"
  "mp3d_stage2_volume_256x512|mp3d_double_256|omni_gs_160x320_mp3d_cylinder_volume_256.py"
  "mp3d_stage1_pixel_256x512|mp3d_double_256|omni_gs_160x320_mp3d_cylinder_pixel_256.py"
)

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case $1 in
    --gpus) GPUS=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --data-dir) DATA_DIR=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done
mkdir -p "$DATA_DIR" "$OUT_DIR"
DATA_DIR=$(cd "$DATA_DIR" && pwd)
OUT_DIR=$(cd "$OUT_DIR" && pwd)

# the selected jobs and the evaluation archives they need
SELECTED=()
NEED_MP3D=0
NEED_LOC360=0
for job in "${JOBS[@]}"; do
  name=${job%%|*}
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi
  SELECTED+=("$job")
  case $name in loc360_*) NEED_LOC360=1 ;; *) NEED_MP3D=1 ;; esac
done
if [ ${#SELECTED[@]} -eq 0 ]; then
  echo "--only $ONLY matches no checkpoint; names: $(printf '%s ' "${JOBS[@]%%|*}")" >&2
  exit 2
fi
ARCHIVES=()
[ $NEED_MP3D = 1 ] && ARCHIVES+=(eval/pano_grf_evalsets_images.tar eval/pano_grf_evalsets_depth.tar)
[ $NEED_LOC360 = 1 ] && ARCHIVES+=(eval/360Loc_atrium_images.tar eval/360Loc_atrium_depth.tar)
FILES=(checkpoints/pansplat_backbone/pansplat_last.ckpt "${ARCHIVES[@]}")
for job in "${SELECTED[@]}"; do FILES+=("checkpoints/${job%%|*}/model.safetensors"); done

echo "== 1/4 download from ${HF_ENDPOINT:-https://huggingface.co} ($HF_REPO) into $DATA_DIR"
"$PYTHON" - "$HF_REPO" "$DATA_DIR" SHA256SUMS "${FILES[@]}" <<'EOF'
import os
import sys
import time

# The default 10 s timeouts and single attempt are fragile on slow or flaky links (connections to the storage
# host can fail for minutes at a time); each retry resumes the partial file.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

repo, local_dir, *files = sys.argv[1:]
for name in files:
    print(f"   {name}", flush=True)
    for attempt in range(1, 21):
        try:
            hf_hub_download(repo, name, repo_type="dataset", local_dir=local_dir)
            break
        except (EntryNotFoundError, RepositoryNotFoundError):
            raise
        except Exception as e:
            if attempt == 20:
                raise
            wait = min(30 * attempt, 300)
            print(f"   attempt {attempt} failed ({type(e).__name__}); retrying in {wait} s", flush=True)
            time.sleep(wait)
EOF

echo "== 2/4 check SHA-256"
(
  cd "$DATA_DIR"
  for f in "${FILES[@]}"; do
    line=$(awk -v p="./$f" '$2 == p' SHA256SUMS)
    [ -n "$line" ] || { echo "$f is not listed in SHA256SUMS" >&2; exit 1; }
    stamp=".sha256_ok/$f"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$line" ] && [ ! "$f" -nt "$stamp" ]; then continue; fi
    echo "$line" | sha256sum -c --quiet || { echo "checksum mismatch: $f (delete it and run again)" >&2; exit 1; }
    mkdir -p "$(dirname "$stamp")" && echo "$line" > "$stamp"
  done
)

echo "== 3/4 extract"
extract() {  # archive, destination root
  local stamp=$2/.extracted_$(basename "$1")
  [ -f "$stamp" ] && return 0
  echo "   $1 -> $2"
  mkdir -p "$2" && tar -xf "$DATA_DIR/$1" -C "$2" && touch "$stamp"
}
for a in "${ARCHIVES[@]}"; do
  case $a in eval/pano_grf_*) extract "$a" "$DATA_DIR/pano_grf" ;; eval/360Loc_*) extract "$a" "$DATA_DIR/360Loc" ;; esac
done

echo "== 4/4 evaluate on GPU(s) $GPUS; logs and metrics.json in $OUT_DIR/<name>/"
export CYLINDERSPLAT_PANO_GRF=$DATA_DIR/pano_grf
export CYLINDERSPLAT_360LOC=$DATA_DIR/360Loc
export CYLINDERSPLAT_PANSPLAT_CKPT=$DATA_DIR/checkpoints/pansplat_backbone/pansplat_last.ckpt
cd "$REPO"
run_job() {  # job, gpu
  local name dataset config
  IFS='|' read -r name dataset config <<< "$1"
  if [ -f "$OUT_DIR/$name/metrics.json" ]; then echo "   $name: done before"; return 0; fi
  echo "   $name: started on GPU $2"
  mkdir -p "$OUT_DIR/$name"
  if CUDA_VISIBLE_DEVICES=$2 "$PYTHON" evaluate.py --dataset "$dataset" --py-config "configs/OmniScene/$config" \
      --ckpt "$DATA_DIR/checkpoints/$name" --out-dir "$OUT_DIR/$name" > "$OUT_DIR/$name/stdout.log" 2>&1; then
    echo "   $name: finished"
  else
    echo "   $name: FAILED, see $OUT_DIR/$name/stdout.log"
  fi
}
IFS=, read -ra GPU_LIST <<< "$GPUS"
for k in "${!GPU_LIST[@]}"; do
  (
    for ((j = k; j < ${#SELECTED[@]}; j += ${#GPU_LIST[@]})); do run_job "${SELECTED[j]}" "${GPU_LIST[k]}"; done
  ) &
done
wait

"$PYTHON" - "$OUT_DIR" "${SELECTED[@]%%|*}" <<'EOF'
import json, os, sys

# This code on an RTX 4090 (WS-PSNR, SSIM, LPIPS, PCC); repeated runs differ by about 1e-6 dB
# (atomic adds in the volume branch), other GPUs and library versions by slightly more.
EXPECTED = {
    "mp3d_stage1_pixel_256x512": {
        "m3d_1.0": (22.3872, 0.8145, 0.2074, 0.8215), "m3d_0.75": (24.9686, 0.8594, 0.1472, 0.8679),
        "m3d_0.5": (28.7669, 0.9224, 0.0759, 0.9194), "replica_0.5": (30.6009, 0.9595, 0.0571, 0.8563),
        "residential_0.15": (27.9248, 0.8664, 0.1540, 0.8075)},
    "mp3d_stage2_volume_256x512": {
        "m3d_1.0": (21.4154, 0.7102, 0.3369, 0.8043), "m3d_0.75": (22.9799, 0.7610, 0.2791, 0.8631),
        "m3d_0.5": (25.3254, 0.8258, 0.2056, 0.9068), "replica_0.5": (25.5992, 0.8641, 0.1777, 0.8929),
        "residential_0.15": (27.7328, 0.8470, 0.2642, 0.8112)},
    "mp3d_stage3_joint_256x512": {
        "m3d_1.0": (23.5893, 0.8251, 0.1992, 0.8462), "m3d_0.75": (25.6471, 0.8667, 0.1440, 0.8892),
        "m3d_0.5": (29.6867, 0.9328, 0.0707, 0.9214), "replica_0.5": (31.1060, 0.9622, 0.0565, 0.8573),
        "residential_0.15": (28.1016, 0.8667, 0.1548, 0.8163)},
    "mp3d_stage4_joint_512x1024": {
        "m3d_1.0": (23.3043, 0.8433, 0.2280, 0.8532), "m3d_0.75": (25.3597, 0.8669, 0.1830, 0.8861),
        "m3d_0.5": (29.6112, 0.9252, 0.1033, 0.9203), "replica_0.5": (30.4573, 0.9580, 0.0827, 0.8579),
        "residential_0.15": (27.7385, 0.8613, 0.2113, 0.8178)},
    "loc360_finetune_256x512": {"total": (28.8556, 0.8851, 0.1009, 0.8630)},
}
LABELS = {"m3d_1.0": "M3D 2.0 m", "m3d_0.75": "M3D 1.5 m", "m3d_0.5": "M3D 1.0 m", "replica_0.5": "Replica",
          "residential_0.15": "Residential", "total": "360Loc"}

out_dir, names = sys.argv[1], sys.argv[2:]
print(f"\n{'checkpoint':28s} {'test set':12s} {'WS-PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s} {'PCC':>7s}  {'expected WS-PSNR':>16s}")
missing = 0
for name in names:
    path = os.path.join(out_dir, name, "metrics.json")
    if not os.path.isfile(path):
        print(f"{name:28s} no metrics.json (see {os.path.join(out_dir, name, 'stdout.log')})")
        missing += 1
        continue
    record = json.load(open(path))
    for key, exp in EXPECTED[name].items():
        m = (record["total"] if key == "total" else record["scenes"][key])["metrics"]
        diff = m["wspsnr"] - exp[0]
        print(f"{name:28s} {LABELS[key]:12s} {m['wspsnr']:8.3f} {m['ssim']:7.4f} {m['lpips']:7.4f} {m['pcc']:7.4f}"
              f"  {exp[0]:8.3f} ({diff:+.3f})")
sys.exit(1 if missing else 0)
EOF

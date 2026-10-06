#!/usr/bin/env bash
# One fine-tune run on one GPU: configs/OmniScene/screen/<ARM>.py (ARM = long_*; row
# mp3d_double_256_screen) trained from a checkpoint, then every saved checkpoint evaluated on mp3d_double_256_val (all
# targets and --novel-only), one checkpoint chosen on val by tools/select_checkpoint.py (--fallback best by default),
# and only that checkpoint evaluated on mp3d_double_256 (all targets and --novel-only) and its rendered Gaussians
# counted on val (tools/count_rendered_gaussians.py).
#
#   CYLINDERSPLAT_S3=<dir of mp3d_stage3_joint_256x512> CYLINDERSPLAT_RUNS_ROOT=<runs root> \
#   CYLINDERSPLAT_RELEASED_VAL=<the released checkpoint's mp3d_double_256_val --novel-only metrics.json> \
#   [CYLINDERSPLAT_SELECT_REFS="<val --novel-only metrics.json> ..."] [CYLINDERSPLAT_SELECT_FALLBACK=best|none] \
#       bash scripts/long_arm.sh <ARM> <GPU> [--transfer NAME]
#
# The run goes to $CYLINDERSPLAT_RUNS_ROOT/<ARM>: checkpoint-<step>, eval_mp3d_double_256_val[_novel]_<step> for every
# step, selection.json, eval_mp3d_double_256[_novel]_<chosen step>, count_<chosen step>.json. Training restarts from
# step 0 (train.py loads weights only), so the run claims its directory with an exclusive mkdir; with the final
# checkpoint present, only missing evaluations run. The selection reference is the released record, or the per-metric
# best of it and every record in CYLINDERSPLAT_SELECT_REFS (space-separated, each passed as --reference; default empty =
# released only), with --fallback $CYLINDERSPLAT_SELECT_FALLBACK (default best). The last line printed is
# `selected <step> <reason>` (reason vtol or fallback, see select_checkpoint.py). The open-file limit is raised (the 32
# loader workers need more than 1024 descriptors). With CYLINDERSPLAT_ALLOWED_GPUS set (e.g. "1 2 3"), GPU must be one
# of them. Exit codes: 2 bad arguments or environment (checked before training); 3 test is read once: the run holds a
# test record of another step, or the new selection is null after test was read (selection.json is left unchanged); 4
# the run directory exists without the final checkpoint (a stopped or a concurrent run; move it aside by hand); 5 a
# CYLINDERSPLAT_EVAL_STEPS step has no checkpoint, or the selection named no saved step; 6 no step within V-tol with
# fallback none (the last line is `selected none none`; no test record and no count are written), and every later run
# of an arm whose selection.json records that null choice; any other code is the training's.
#
# Optional overrides (unset = the behaviour above): CYLINDERSPLAT_INIT (the checkpoint directory training starts from
# instead of CYLINDERSPLAT_S3, which is then not needed), CYLINDERSPLAT_STEPS (training steps, default 20000),
# CYLINDERSPLAT_EVAL_STEPS (space-separated steps: only these checkpoints are evaluated on val and offered to the
# selection; each must exist).
set -euo pipefail
usage() { echo "usage: long_arm.sh <ARM> <GPU> [--transfer NAME]" >&2; exit 2; }
need() { [ -n "${!1:-}" ] || { echo "set $1$2" >&2; exit 2; }; }
[ $# -ge 2 ] || usage
ARM=$1
GPU=$2
shift 2
TRANSFER=exact
case $# in
    0) ;;
    2) { [ "$1" = --transfer ] && [ -n "$2" ]; } || usage; TRANSFER=$2 ;;
    *) usage ;;
esac
[ -n "${CYLINDERSPLAT_INIT:-}" ] || need CYLINDERSPLAT_S3 " to the released stage-3 checkpoint directory"
INIT=${CYLINDERSPLAT_INIT:-${CYLINDERSPLAT_S3:-}}
need CYLINDERSPLAT_RUNS_ROOT ""
need CYLINDERSPLAT_RELEASED_VAL " to the released val --novel-only metrics.json"
PYTHON=${PYTHON:-python}
STEPS=${CYLINDERSPLAT_STEPS:-20000}
case $STEPS in '' | *[!0-9]* | 0*) echo "CYLINDERSPLAT_STEPS must be a positive integer: $STEPS" >&2; exit 2 ;; esac
read -r -d '' -a eval_steps <<< "${CYLINDERSPLAT_EVAL_STEPS:-}" || true
for step in ${eval_steps[@]+"${eval_steps[@]}"}; do
    case $step in *[!0-9]* | 0?*) echo "CYLINDERSPLAT_EVAL_STEPS: not a step: $step" >&2; exit 2 ;; esac
done
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=configs/OmniScene/screen/$ARM.py
RUN=$CYLINDERSPLAT_RUNS_ROOT/$ARM
cd "$REPO"
case $ARM in long_*) ;; *) echo "not a long config: $ARM" >&2; exit 2 ;; esac
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 2; }
if [ -n "${CYLINDERSPLAT_ALLOWED_GPUS:-}" ]; then
    case " $CYLINDERSPLAT_ALLOWED_GPUS " in *" $GPU "*) ;; *) echo "GPU $GPU not in $CYLINDERSPLAT_ALLOWED_GPUS" >&2; exit 2 ;; esac
fi
# a null selection is final: a re-run with another fallback or reference list must not choose a step and read test
if [ -f "$RUN/selection.json" ] && grep -q '"chosen": null' "$RUN/selection.json"; then
    echo "null choice recorded in $RUN/selection.json" >&2
    echo "selected none none"
    exit 6
fi
FALLBACK=${CYLINDERSPLAT_SELECT_FALLBACK:-best}
case $FALLBACK in
    best | none) ;;
    *) echo "CYLINDERSPLAT_SELECT_FALLBACK must be best or none: $FALLBACK" >&2; exit 2 ;;
esac
read -r -d '' -a select_refs <<< "${CYLINDERSPLAT_SELECT_REFS:-}" || true  # split on any whitespace, no globbing
reference_args=()
for ref in ${select_refs[@]+"${select_refs[@]}"}; do
    [ -f "$ref" ] || { echo "no selection reference $ref" >&2; exit 2; }
    reference_args+=(--reference "$ref")
done
hard=$(ulimit -Hn)
if [ "$hard" = unlimited ] || [ "$hard" -ge 65536 ]; then ulimit -n 65536; else ulimit -n "$hard"; fi

if [ ! -f "$RUN/checkpoint-$STEPS/model.safetensors" ]; then
    mkdir "$RUN" 2>/dev/null || { echo "$RUN exists without checkpoint-$STEPS: a stopped or running run" >&2; exit 4; }
    "$PYTHON" -m accelerate.commands.launch --config-file configs/accelerate/accel_1proc.yaml --gpu_ids "$GPU" \
        train.py --entry mp3d_double_256_screen --py-config "$CONFIG" --run-id "$ARM" \
        --resume-from "$INIT" --transfer "$TRANSFER" --max-steps "$STEPS" --save-final
fi

candidates=()
for step in ${eval_steps[@]+"${eval_steps[@]}"}; do
    [ -f "$RUN/checkpoint-$step/model.safetensors" ] || { echo "CYLINDERSPLAT_EVAL_STEPS: no checkpoint-$step" >&2; exit 5; }
done
for ckpt in "$RUN"/checkpoint-*; do
    step=${ckpt##*-}
    if [ ${#eval_steps[@]} -gt 0 ]; then
        case " ${eval_steps[*]} " in *" $step "*) ;; *) continue ;; esac
    fi
    for novel in "" --novel-only; do
        out=$RUN/eval_mp3d_double_256_val${novel:+_novel}_$step
        [ -f "$out/metrics.json" ] && continue
        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate.py --dataset mp3d_double_256_val --py-config "$CONFIG" \
            --ckpt "$ckpt" --out-dir "$out" $novel
    done
    candidates+=("$step=$RUN/eval_mp3d_double_256_val_novel_$step/metrics.json")
done

# the new selection replaces selection.json only after the read-test-once check
sel=$RUN/selection.json.new
read -r chosen reason < <("$PYTHON" tools/select_checkpoint.py --released "$CYLINDERSPLAT_RELEASED_VAL" \
    ${reference_args[@]+"${reference_args[@]}"} --fallback "$FALLBACK" --out "$sel" "${candidates[@]}")
test_records=()
for done_test in "$RUN"/eval_mp3d_double_256_[0-9]* "$RUN"/eval_mp3d_double_256_novel_*; do
    [ -e "$done_test" ] && test_records+=("$done_test")
done
if [ "$chosen" = none ] && [ "$reason" = none ]; then
    if [ ${#test_records[@]} -gt 0 ]; then
        rm -f "$sel"
        echo "test already read (${test_records[0]}): a null selection is refused" >&2
        exit 3
    fi
    mv "$sel" "$RUN/selection.json"
    echo "selected none none"  # no step within V-tol of the reference: test is not read, nothing is counted
    exit 6
fi
[ -f "$RUN/checkpoint-$chosen/model.safetensors" ] || { rm -f "$sel"; echo "selection failed: '$chosen'" >&2; exit 5; }
for done_test in ${test_records[@]+"${test_records[@]}"}; do
    case $done_test in
        "$RUN/eval_mp3d_double_256_$chosen" | "$RUN/eval_mp3d_double_256_novel_$chosen") ;;
        *) rm -f "$sel"; echo "test already read for another checkpoint: $done_test" >&2; exit 3 ;;
    esac
done
mv "$sel" "$RUN/selection.json"
for novel in "" --novel-only; do
    out=$RUN/eval_mp3d_double_256${novel:+_novel}_$chosen
    [ -f "$out/metrics.json" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate.py --dataset mp3d_double_256 --py-config "$CONFIG" \
        --ckpt "$RUN/checkpoint-$chosen" --out-dir "$out" $novel
done
count=$RUN/count_$chosen.json
[ -f "$count" ] || CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" tools/count_rendered_gaussians.py --py-config "$CONFIG" \
    --ckpt "$RUN/checkpoint-$chosen" --out "$count" --split val
echo "selected $chosen $reason"

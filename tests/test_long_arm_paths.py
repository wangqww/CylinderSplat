"""scripts/long_arm.sh with a stub PYTHON (no GPU, dataset or checkpoint): the selection references and fallback from
the environment, a null selection (exit 6, test not read), test read once (exit 3), a claimed run directory (exit 4), a
training failure, the argument checks (exit 2) and the opt-in overrides CYLINDERSPLAT_INIT / STEPS / EVAL_STEPS."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

ARM = "long_c0"  # an existing long config; the script refuses any other name

# Stands in for `python` in long_arm.sh: logs its argv (one JSON list per line) and emulates train.py (through the
# accelerate launcher), evaluate.py, tools/select_checkpoint.py and tools/count_rendered_gaussians.py.
STUB = r'''#!{python}
import json
import os
import sys

argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(argv) + "\n")


def opt(name):
    return argv[argv.index(name) + 1]


if argv[:2] == ["-m", "accelerate.commands.launch"]:
    run = os.path.join(os.environ["CYLINDERSPLAT_RUNS_ROOT"], opt("--run-id"))
    if os.environ["STUB_TRAIN"] == "crash":
        sys.exit(1)
    final = int(opt("--max-steps"))
    for step in sorted({min(5000, final), final}):
        os.makedirs(os.path.join(run, f"checkpoint-{step}"))
        open(os.path.join(run, f"checkpoint-{step}", "model.safetensors"), "w").close()
elif argv[0] == "evaluate.py":
    os.makedirs(opt("--out-dir"))
    with open(os.path.join(opt("--out-dir"), "metrics.json"), "w") as f:
        json.dump(dict(dataset=opt("--dataset"), novel_only="--novel-only" in argv), f)
elif argv[0] == "tools/select_checkpoint.py":
    chosen, reason = os.environ["STUB_SELECT"].split()
    with open(opt("--out"), "w") as f:  # the real tool's layout: json.dump(..., indent=2)
        json.dump(dict(chosen=None if chosen == "none" else chosen, reason=reason, stub=True), f, indent=2)
    print(os.environ["STUB_SELECT"])
elif argv[0] == "tools/count_rendered_gaussians.py":
    with open(opt("--out"), "w") as f:
        json.dump(dict(mean=1.0), f)
else:
    sys.exit(f"unexpected call {argv}")
'''


def long_arm(tmp_path, select="20000 vtol", train="ok", arm=ARM, drop_s3=False, args=(), **env_extra):
    """Run long_arm.sh ARM 0 [ARGS] with the stub; returns (completed process, logged calls, run directory)."""
    stub = tmp_path / "python"
    stub.write_text(STUB.replace("{python}", sys.executable))
    stub.chmod(0o755)
    released = tmp_path / "released_val.json"
    released.write_text("{}")
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    log = tmp_path / "calls.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CYLINDERSPLAT_")}
    env.update(PYTHON=str(stub), CYLINDERSPLAT_S3=str(tmp_path / "s3"), CYLINDERSPLAT_RUNS_ROOT=str(runs),
               CYLINDERSPLAT_RELEASED_VAL=str(released), STUB_LOG=str(log), STUB_SELECT=select, STUB_TRAIN=train,
               **env_extra)
    if drop_s3:
        del env["CYLINDERSPLAT_S3"]
    proc = subprocess.run([BASH, os.path.join(REPO_ROOT, "scripts", "long_arm.sh"), arm, "0", *args], env=env,
                          cwd=tmp_path, capture_output=True, text=True, timeout=120)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc, calls, runs / arm


def entries(run, pattern):
    return sorted(p.name for p in run.glob(pattern))


def select_call(calls):
    (call,) = [c for c in calls if c[0] == "tools/select_checkpoint.py"]
    return call


def opt(call, name):
    return call[call.index(name) + 1]


def test_the_default_environment_keeps_the_released_only_selection(tmp_path):
    proc, calls, run = long_arm(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "selected 20000 vtol"
    assert select_call(calls) == [
        "tools/select_checkpoint.py", "--released", str(tmp_path / "released_val.json"), "--fallback", "best",
        "--out", f"{run}/selection.json.new", f"20000={run}/eval_mp3d_double_256_val_novel_20000/metrics.json",
        f"5000={run}/eval_mp3d_double_256_val_novel_5000/metrics.json"]
    assert (run / "selection.json").exists() and not (run / "selection.json.new").exists()
    assert entries(run, "eval_mp3d_double_256_[0-9n]*") == ["eval_mp3d_double_256_20000",
                                                           "eval_mp3d_double_256_novel_20000"]
    assert entries(run, "count_*.json") == ["count_20000.json"]


def test_select_refs_pass_one_reference_each(tmp_path):
    refs = [tmp_path / "released_ref.json", tmp_path / "ls_val.json"]
    for ref in refs:
        ref.write_text("{}")
    proc, calls, _ = long_arm(tmp_path, CYLINDERSPLAT_SELECT_REFS=f" {refs[0]}  {refs[1]}\n",
                              CYLINDERSPLAT_SELECT_FALLBACK="none")
    assert proc.returncode == 0, proc.stderr
    call = select_call(calls)
    assert call[3:9] == ["--reference", str(refs[0]), "--reference", str(refs[1]), "--fallback", "none"]


@pytest.mark.parametrize("env, message", [
    (dict(CYLINDERSPLAT_SELECT_FALLBACK="first"), "best or none"),
    (dict(CYLINDERSPLAT_SELECT_REFS="/nonexistent/ls_val.json"), "no selection reference"),
])
def test_a_bad_selection_environment_is_refused_before_training(tmp_path, env, message):
    proc, calls, run = long_arm(tmp_path, **env)
    assert proc.returncode == 2 and message in proc.stderr
    assert calls == [] and not run.exists()


def test_a_null_selection_reads_no_test_and_counts_nothing(tmp_path):
    proc, calls, run = long_arm(tmp_path, select="none none", CYLINDERSPLAT_SELECT_FALLBACK="none")
    assert proc.returncode == 6, proc.stderr
    assert proc.stdout.splitlines()[-1] == "selected none none"
    assert opt(select_call(calls), "--fallback") == "none"
    assert len(entries(run, "eval_mp3d_double_256_val*")) == 4  # every saved step on val, all targets and novel
    assert entries(run, "eval_mp3d_double_256_[0-9n]*") == [] and entries(run, "count_*") == []
    assert {opt(c, "--dataset") for c in calls if c[0] == "evaluate.py"} == {"mp3d_double_256_val"}
    assert not any(c[0] == "tools/count_rendered_gaussians.py" for c in calls)


def test_a_training_failure_keeps_its_exit_code(tmp_path):
    proc, calls, run = long_arm(tmp_path, train="crash")
    assert proc.returncode == 1
    assert len(calls) == 1 and entries(run, "eval_*") == []


def test_a_null_selection_is_final(tmp_path):
    proc, calls, run = long_arm(tmp_path, select="none none", CYLINDERSPLAT_SELECT_FALLBACK="none")
    assert proc.returncode == 6
    # a re-run with another fallback must not choose a step and read test
    proc, calls_again, _ = long_arm(tmp_path, select="20000 fallback", CYLINDERSPLAT_SELECT_FALLBACK="best")
    assert proc.returncode == 6 and "selected none none" in proc.stdout
    assert len(calls_again) == len(calls)  # no new call: the log is shared and nothing was appended
    assert entries(run, "eval_mp3d_double_256_[0-9n]*") == [] and entries(run, "count_*") == []


def train_call(calls):
    (call,) = [c for c in calls if c[:2] == ["-m", "accelerate.commands.launch"]]
    return call


def count_call(calls):
    (call,) = [c for c in calls if c[0] == "tools/count_rendered_gaussians.py"]
    return call


def test_the_default_launch_keeps_the_released_init_and_steps(tmp_path):
    proc, calls, run = long_arm(tmp_path)
    assert proc.returncode == 0, proc.stderr
    call = train_call(calls)
    assert opt(call, "--resume-from") == str(tmp_path / "s3") and opt(call, "--max-steps") == "20000"
    assert opt(count_call(calls), "--py-config") == "configs/OmniScene/screen/long_c0.py"


def test_init_steps_and_eval_steps_overrides(tmp_path):
    init = tmp_path / "ls_ckpt"
    proc, calls, run = long_arm(tmp_path, select="12000 fallback", drop_s3=True, CYLINDERSPLAT_INIT=str(init),
                                CYLINDERSPLAT_STEPS="12000", CYLINDERSPLAT_EVAL_STEPS="12000")
    assert proc.returncode == 0, proc.stderr
    call = train_call(calls)
    assert opt(call, "--resume-from") == str(init) and opt(call, "--max-steps") == "12000"
    assert entries(run, "checkpoint-*") == ["checkpoint-12000", "checkpoint-5000"]
    # only the listed step is evaluated on val and offered to the selection
    assert entries(run, "eval_mp3d_double_256_val*") == ["eval_mp3d_double_256_val_12000",
                                                          "eval_mp3d_double_256_val_novel_12000"]
    assert select_call(calls)[-1] == f"12000={run}/eval_mp3d_double_256_val_novel_12000/metrics.json"
    assert opt(count_call(calls), "--py-config") == "configs/OmniScene/screen/long_c0.py"


@pytest.mark.parametrize("env, message, code", [
    (dict(CYLINDERSPLAT_STEPS="12k"), "positive integer", 2),
    (dict(CYLINDERSPLAT_STEPS="0"), "positive integer", 2),
    (dict(CYLINDERSPLAT_EVAL_STEPS="12000 x"), "not a step", 2),
])
def test_bad_overrides_are_refused_before_training(tmp_path, env, message, code):
    proc, calls, run = long_arm(tmp_path, **env)
    assert proc.returncode == code and message in proc.stderr
    assert calls == [] and not run.exists()


def test_an_eval_step_without_its_checkpoint_stops_before_any_evaluation(tmp_path):
    proc, calls, run = long_arm(tmp_path, CYLINDERSPLAT_EVAL_STEPS="7000")
    assert proc.returncode == 5 and "no checkpoint-7000" in proc.stderr
    assert entries(run, "eval_*") == [] and len(calls) == 1


def test_without_init_or_s3_the_script_refuses(tmp_path):
    proc, calls, _ = long_arm(tmp_path, drop_s3=True)
    assert proc.returncode == 2 and "CYLINDERSPLAT_S3" in proc.stderr and calls == []


@pytest.mark.parametrize("arm", ["stage3_screen", "other_arm"])
def test_non_long_configs_are_refused(tmp_path, arm):
    proc, calls, _ = long_arm(tmp_path, arm=arm)
    assert proc.returncode == 2 and "not a long config" in proc.stderr and calls == []


def test_test_is_read_once_and_a_reselection_keeps_the_recorded_choice(tmp_path):
    proc, calls, run = long_arm(tmp_path)  # selects 20000, reads test for it
    assert proc.returncode == 0, proc.stderr
    recorded = (run / "selection.json").read_text()
    for select in ("5000 vtol", "none none"):  # another step, or a null choice, after test was read
        proc, calls_again, _ = long_arm(tmp_path, select=select, CYLINDERSPLAT_SELECT_FALLBACK="none")
        assert proc.returncode == 3 and "test already read" in proc.stderr
        assert (run / "selection.json").read_text() == recorded and not (run / "selection.json.new").exists()
        new = calls_again[len(calls):]
        assert not any(c[0] == "evaluate.py" and opt(c, "--dataset") == "mp3d_double_256" for c in new)
        calls = calls_again


def test_a_run_directory_without_the_final_checkpoint_is_refused(tmp_path):
    (tmp_path / "runs" / ARM / "checkpoint-5000").mkdir(parents=True)  # a stopped or a concurrent run
    proc, calls, _ = long_arm(tmp_path)
    assert proc.returncode == 4 and "without checkpoint-20000" in proc.stderr and calls == []


@pytest.mark.parametrize("args", [("--transfer",), ("--transfer", ""), ("--transfr", "exact"), ("extra",),
                                  ("--transfer", "exact", "extra")])
def test_bad_arguments_are_refused(tmp_path, args):
    proc, calls, run = long_arm(tmp_path, args=args)
    assert proc.returncode == 2 and "usage" in proc.stderr and calls == [] and not run.exists()


def test_a_named_transfer_reaches_train_py(tmp_path):
    proc, calls, _ = long_arm(tmp_path, args=("--transfer", "stage2_to_stage3"))
    assert proc.returncode == 0, proc.stderr
    assert opt(train_call(calls), "--transfer") == "stage2_to_stage3"

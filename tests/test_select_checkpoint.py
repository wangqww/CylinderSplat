"""tools/select_checkpoint.py on synthetic val --novel-only records."""

import json

import pytest

from tools import select_checkpoint as sc

RELEASED = dict(wspsnr=25.0, ssim=0.85, lpips=0.12)


def write(tmp_path, name, wspsnr, ssim, lpips, dataset="mp3d_double_256_val", novel_only=True):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(dict(dataset=dataset, protocol=dict(novel_only=novel_only),
                                    total=dict(metrics=dict(wspsnr=wspsnr, ssim=ssim, lpips=lpips)))))
    return str(path)


def run(tmp_path, fallback, candidates):
    released = write(tmp_path, "released", **RELEASED)
    out = tmp_path / "selection.json"
    args = ["--released", released, "--fallback", fallback, "--out", str(out)]
    args += [f"{key}={write(tmp_path, f'c{i}', *values)}" for i, (key, values) in enumerate(candidates.items())]
    return args, out


def test_latest_step_meeting_the_tolerances_wins(tmp_path, capsys):
    args, out = run(tmp_path, "best", {"5000": (25.01, 0.85, 0.12), "10000": (24.96, 0.849, 0.121),
                                       "15000": (24.94, 0.85, 0.12), "20000": (25.2, 0.85, 0.123)})
    sc.main(args)
    assert capsys.readouterr().out.split() == ["10000", "vtol"]  # 15000: WS -0.06; 20000: LPIPS +0.003
    sel = json.loads(out.read_text())
    assert sel["chosen"] == "10000" and sel["reason"] == "vtol"
    assert [sel["candidates"][k]["meets_vtol"] for k in ("5000", "10000", "15000", "20000")] == [True, True, False, False]


@pytest.mark.parametrize("last, reason", [((25.0, 0.85, 0.12), "vtol"), ((24.9, 0.85, 0.12), "fallback")])
def test_the_same_key_carries_either_reason(tmp_path, capsys, last, reason):
    args, out = run(tmp_path, "best", {"5000": (24.7, 0.85, 0.12), "20000": last})
    sc.main(args)
    assert capsys.readouterr().out.split() == ["20000", reason]
    assert json.loads(out.read_text())["reason"] == reason


def test_a_fallback_tie_goes_to_the_later_step(tmp_path, capsys):
    args, _ = run(tmp_path, "best", {"10000": (24.9, 0.85, 0.12), "20000": (24.9, 0.85, 0.12), "5000": (24.8, 0.85, 0.12)})
    sc.main(args)
    assert capsys.readouterr().out.split() == ["20000", "fallback"]


def test_fallback_none_takes_nothing(tmp_path, capsys):
    args, out = run(tmp_path, "none", {"5000": (24.8, 0.85, 0.12), "10000": (24.9, 0.85, 0.12)})
    sc.main(args)
    assert capsys.readouterr().out.split() == ["none", "none"]
    assert json.loads(out.read_text())["chosen"] is None


def test_fractional_keys_compare_numerically_and_print_as_given(tmp_path, capsys):
    args, _ = run(tmp_path, "none", {"0.25": (25.0, 0.85, 0.12), "0.5": (24.97, 0.85, 0.12), "0.75": (24.9, 0.85, 0.12)})
    sc.main(args)
    assert capsys.readouterr().out.split() == ["0.5", "vtol"]


@pytest.mark.parametrize("dataset, novel_only", [("mp3d_double_256", True), ("mp3d_double_256_val", False)])
def test_a_record_that_is_not_val_novel_is_refused(tmp_path, dataset, novel_only):
    released = write(tmp_path, "released", **RELEASED)
    bad = write(tmp_path, "bad", 25.0, 0.85, 0.12, dataset=dataset, novel_only=novel_only)
    with pytest.raises(ValueError, match="novel-only"):
        sc.main(["--released", released, "--fallback", "best", "--out", str(tmp_path / "s.json"), f"5000={bad}"])


@pytest.mark.parametrize("candidate", ["5000", "x=a.json", "5000="])
def test_malformed_candidates_are_refused(tmp_path, candidate):
    released = write(tmp_path, "released", **RELEASED)
    with pytest.raises(ValueError):
        sc.main(["--released", released, "--fallback", "best", "--out", str(tmp_path / "s.json"), candidate])


STEPS = {"5000": (25.06, 0.85, 0.12), "10000": (25.0, 0.85, 0.12), "15000": (24.98, 0.851, 0.121),
         "20000": (24.9, 0.85, 0.12)}


def test_the_released_only_command(tmp_path, capsys):
    """`--released R --fallback best` with no --reference (long_arm.sh's default): the choice, the printed line and the
    JSON (keys and order included) are select() against the released numbers."""
    args, out = run(tmp_path, "best", STEPS)
    sc.main(args)
    released = sc.val_1m(args[1])
    chosen, reason, report = sc.select({key: dict(zip(sc.METRICS, v)) for key, v in STEPS.items()}, released, "best")
    assert (chosen, reason) == ("15000", "vtol")
    assert capsys.readouterr().out == "15000 vtol\n"
    sel = json.loads(out.read_text())
    assert list(sel) == ["chosen", "reason", "tolerances", "released", "candidates"]
    assert sel["chosen"] == chosen and sel["reason"] == reason
    assert sel["released"] == dict(path=args[1], metrics=released)
    for key, row in sel["candidates"].items():
        assert {k: row[k] for k in ("metrics", "deltas", "meets_vtol")} == report[key]


def test_a_reference_equal_to_released_selects_the_same_step(tmp_path, capsys):
    args, out = run(tmp_path, "best", STEPS)
    sc.main(args)
    first, plain = capsys.readouterr().out, json.loads(out.read_text())
    sc.main(args[:2] + ["--reference", args[1]] + args[2:])
    assert capsys.readouterr().out == first
    sel = json.loads(out.read_text())
    assert sel["chosen"] == plain["chosen"] and sel["candidates"] == plain["candidates"]
    assert sel["references"] == [dict(path=args[1], metrics=RELEASED)] and sel["reference"]["metrics"] == RELEASED


def test_a_better_reference_rejects_a_step_within_released_tolerance(tmp_path, capsys):
    """LS-like reference: +0.1 dB over released. 15000 and 10000 are within V-tol of released but not of the
    reference; 5000 (+0.06 dB over released, -0.04 vs the reference) is the largest step left."""
    args, out = run(tmp_path, "best", STEPS)
    better = write(tmp_path, "ls", 25.1, 0.849, 0.121)
    sc.main(args[:2] + ["--reference", better] + args[2:])
    assert capsys.readouterr().out.split() == ["5000", "vtol"]
    sel = json.loads(out.read_text())
    # per metric: WS-PSNR from the reference, SSIM and LPIPS from released
    assert sel["reference"]["metrics"] == dict(wspsnr=25.1, ssim=0.85, lpips=0.12)
    assert sel["references"] == [dict(path=better, metrics=dict(wspsnr=25.1, ssim=0.849, lpips=0.121))]
    assert sel["released"]["metrics"] == RELEASED
    assert [sel["candidates"][k]["meets_vtol"] for k in STEPS] == [True, False, False, False]
    assert sel["candidates"]["10000"]["deltas"]["wspsnr"] == pytest.approx(-0.1)


def test_references_repeat_and_take_the_per_metric_best(tmp_path, capsys):
    args, out = run(tmp_path, "best", {"5000": (25.0, 0.85, 0.12)})
    refs = [write(tmp_path, "r1", 24.9, 0.86, 0.13), write(tmp_path, "r2", 24.8, 0.84, 0.11)]
    sc.main(args[:2] + ["--reference", refs[0], "--reference", refs[1]] + args[2:])
    sel = json.loads(out.read_text())
    assert [r["path"] for r in sel["references"]] == refs
    assert sel["reference"]["metrics"] == dict(wspsnr=25.0, ssim=0.86, lpips=0.11)
    # SSIM -0.01 and LPIPS +0.01 against the combined reference: out of V-tol, best WS-PSNR as the fallback
    assert capsys.readouterr().out.split() == ["5000", "fallback"]


def test_fallback_none_against_a_reference_takes_nothing(tmp_path, capsys):
    args, out = run(tmp_path, "none", {"5000": (25.0, 0.85, 0.12), "10000": (24.99, 0.85, 0.12)})
    better = write(tmp_path, "ls", 25.2, 0.85, 0.12)
    sc.main(args[:2] + ["--reference", better] + args[2:])
    assert capsys.readouterr().out == "none none\n"
    sel = json.loads(out.read_text())
    assert sel["chosen"] is None and sel["reason"] == "none" and sel["references"][0]["path"] == better


def test_a_reference_that_is_not_val_novel_is_refused(tmp_path):
    args, _ = run(tmp_path, "best", {"5000": (25.0, 0.85, 0.12)})
    bad = write(tmp_path, "bad", 25.0, 0.85, 0.12, dataset="mp3d_double_256")
    with pytest.raises(ValueError, match="novel-only"):
        sc.main(args[:2] + ["--reference", bad] + args[2:])

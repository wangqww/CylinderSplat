"""tools/decision_metrics.py on records shaped like evaluate.py's metrics.json."""

import json

import pytest

from tools import decision_metrics as dm


def metrics(wspsnr, ssim, lpips):
    return dict(psnr=wspsnr + 1.0, wspsnr=wspsnr, ssim=ssim, lpips=lpips)


def record(tmp_path, name, dataset, total, scenes, novel_only=True):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(dict(
        dataset=dataset, protocol=dict(target_views=[0, 1, 2], context_views=[0, 2], evaluated_views=[1],
                                       novel_only=novel_only),
        scenes={k: dict(metrics=v, count=10) for k, v in scenes.items()}, total=dict(metrics=total, count=100))))
    return str(path)


def v_rec(tmp_path, name, ws, ssim, lp):
    # the val scene line differs from Total on purpose: the 1.0 m number is Total
    return record(tmp_path, name, dm.VAL, metrics(ws, ssim, lp), {"m3d_0.1": metrics(ws - 3, ssim - 0.1, lp + 0.1)})


def t_rec(tmp_path, name, ws2, ssim2, lp2, ws15=22.5, ssim15=0.78, lp15=0.20):
    scenes = {"m3d_1.0": metrics(ws2, ssim2, lp2), "m3d_0.75": metrics(ws15, ssim15, lp15),
              "m3d_0.5": metrics(28.0, 0.9, 0.1)}
    # test Total mixes every baseline and must not be read
    return record(tmp_path, name, dm.TEST, metrics(30.0, 0.95, 0.05), scenes)


def test_val_reads_total_and_test_reads_the_scene_lines(tmp_path):
    assert dm.val_1m(v_rec(tmp_path, "v", 25.0, 0.85, 0.12)) == dict(wspsnr=25.0, ssim=0.85, lpips=0.12)
    n = dm.numbers(v_rec(tmp_path, "v", 25.0, 0.85, 0.12), t_rec(tmp_path, "t", 19.4, 0.66, 0.34))
    assert n["2.0"] == dict(wspsnr=19.4, ssim=0.66, lpips=0.34)
    assert n["1.5"] == dict(wspsnr=22.5, ssim=0.78, lpips=0.20)


@pytest.mark.parametrize("dataset, novel_only, reader", [
    (dm.TEST, True, "val"), (dm.VAL, False, "val"), (dm.VAL, True, "test"), (dm.TEST, False, "test")])
def test_wrong_records_are_refused(tmp_path, dataset, novel_only, reader):
    path = record(tmp_path, "r", dataset, metrics(25, 0.85, 0.12), {"m3d_1.0": metrics(19, 0.66, 0.34)}, novel_only)
    with pytest.raises(ValueError, match="novel-only"):
        dm.val_1m(path) if reader == "val" else dm.test_baseline(path, "m3d_1.0")


def test_reference_is_per_metric():
    a = {"1.0": dict(wspsnr=25.0, ssim=0.85, lpips=0.13), "2.0": dict(wspsnr=19.4, ssim=0.66, lpips=0.34)}
    b = {"1.0": dict(wspsnr=24.8, ssim=0.86, lpips=0.12), "2.0": dict(wspsnr=19.8, ssim=0.68, lpips=0.33)}
    assert dm.reference([a, b]) == {"1.0": dict(wspsnr=25.0, ssim=0.86, lpips=0.12),
                                    "2.0": dict(wspsnr=19.8, ssim=0.68, lpips=0.33)}
    assert dm.reference([a, {"1.0": b["1.0"]}]) == {"1.0": dict(wspsnr=25.0, ssim=0.86, lpips=0.12)}


REF = {"1.0": dict(wspsnr=25.0, ssim=0.85, lpips=0.12), "2.0": dict(wspsnr=19.4, ssim=0.66, lpips=0.34),
       "1.5": dict(wspsnr=22.5, ssim=0.78, lpips=0.20)}


def shifted(d1=(0, 0, 0), d2=(0, 0, 0), d15=(0, 0, 0)):
    return {b: {k: REF[b][k] + d[i] for i, k in enumerate(dm.METRICS)} for b, d in (("1.0", d1), ("2.0", d2), ("1.5", d15))}


@pytest.mark.parametrize("x, passed", [
    (shifted(d2=(0.15, 0, 0)), True),                     # 2.0 m gain
    (shifted(d1=(0.10, 0, 0)), True),                     # 1.0 m gain
    (shifted(d2=(0.14, 0, 0)), False),                    # no gain
    (shifted(d1=(-0.06, 0, 0), d2=(0.5, 0, 0)), False),   # 1.0 m regression
    (shifted(d1=(0, 0, 0.0021), d2=(0.5, 0, 0)), False),  # 1.0 m LPIPS
    (shifted(d2=(0.5, -0.0051, 0)), False),               # 2.0 m SSIM
])
def test_keep_clauses(x, passed):
    assert dm.keep_clauses(x, [REF])["passed"] is passed


def test_inclusive_bounds_survive_decimal_rounding():
    # 19.4 + 0.15 - 19.4 is 0.14999999999999858 in binary floating point; the bound is inclusive
    assert dm.keep_clauses(shifted(d2=(0.15, 0, 0)), [REF])["gain"]
    assert dm.meets_vtol(dict(wspsnr=24.95, ssim=0.848, lpips=0.122), dict(wspsnr=25.0, ssim=0.85, lpips=0.12))
    assert not dm.meets_vtol(dict(wspsnr=24.9499, ssim=0.85, lpips=0.12), dict(wspsnr=25.0, ssim=0.85, lpips=0.12))


def test_the_1p5m_flag_reports_without_deciding():
    result = dm.keep_clauses(shifted(d2=(0.2, 0, 0), d15=(-0.2, 0, 0)), [REF])
    assert result["passed"] and result["flag_1p5m"] and "flag" in result["verdict"]


def test_the_per_metric_reference_changes_the_verdict():
    """K wins WS-PSNR at 2.0 m, released wins LPIPS: the LPIPS reference is released's, so T fails clause (iii),
    although against K alone (one winner per baseline) it would pass."""
    one = dict(wspsnr=25.0, ssim=0.85, lpips=0.12)
    released = {"1.0": one, "2.0": dict(wspsnr=19.4, ssim=0.66, lpips=0.30)}
    k = {"1.0": one, "2.0": dict(wspsnr=19.9, ssim=0.68, lpips=0.34)}
    t = {"1.0": one, "2.0": dict(wspsnr=20.1, ssim=0.68, lpips=0.306)}
    assert dm.reference([k, released])["2.0"] == dict(wspsnr=19.9, ssim=0.68, lpips=0.30)
    assert dm.keep_clauses(t, [k])["passed"] is True
    result = dm.keep_clauses(t, [k, released])
    assert result["passed"] is False and result["gain"] and not result["tolerances"]


def test_without_test_numbers_nothing_is_decided():
    assert dm.keep_clauses({"1.0": REF["1.0"]}, [REF])["passed"] is None


def test_cli_tables(tmp_path):
    rel = [v_rec(tmp_path, "rv", 25.0, 0.85, 0.12), t_rec(tmp_path, "rt", 19.4, 0.66, 0.34)]
    ctrl = [v_rec(tmp_path, "cv", 24.98, 0.85, 0.121), t_rec(tmp_path, "ct", 19.9, 0.68, 0.32)]
    treat = [v_rec(tmp_path, "tv", 25.01, 0.851, 0.12), t_rec(tmp_path, "tt", 20.1, 0.682, 0.318)]
    out = tmp_path / "table.json"
    dm.main(["--released", *rel, "--c0", *ctrl, "--model", "L0", *ctrl, "--model", "LS", *treat,
             "--model", "V", rel[0], "--effect", "LS", "L0", "--out", str(out)])
    table = json.loads(out.read_text())
    assert table["models"]["L0"]["rule_r"]["passed"]  # +0.5 dB at 2.0 m, 1.0 m within tolerance of released
    assert table["models"]["L0"]["vs_c0"]["2.0"]["wspsnr"] == 0.0
    assert table["models"]["V"]["rule_r"]["passed"] is None  # val only: test not read
    # LS over {L0, released}: reference 1.0 m = 25.0 / 0.85 / 0.12, 2.0 m = 19.9 / 0.68 / 0.32; +0.2 dB at 2.0 m
    assert table["effects"]["LS over L0"]["passed"]
    assert list(table) == ["models", "effects"]


def test_cli_rejects_duplicate_models(tmp_path):
    rel = [v_rec(tmp_path, "rv", 25.0, 0.85, 0.12), t_rec(tmp_path, "rt", 19.4, 0.66, 0.34)]
    with pytest.raises(ValueError):
        dm.main(["--released", *rel, "--model", "A", rel[0], "--model", "A", rel[0]])


def test_failed_arms_are_recorded_as_fails(tmp_path, capsys):
    rel = [v_rec(tmp_path, "rv", 25.0, 0.85, 0.12), t_rec(tmp_path, "rt", 19.4, 0.66, 0.34)]
    ls = [v_rec(tmp_path, "lv", 25.2, 0.85, 0.12), t_rec(tmp_path, "lt", 19.7, 0.66, 0.34)]
    out = tmp_path / "t.json"
    dm.main(["--released", *rel, "--model", "LS", *ls, "--failed-selection", "L55", "--failed-selection", "L40",
             "--out", str(out)])
    failed = json.loads(out.read_text())["failed"]
    assert failed["L55"] == dict(stage="selection", passed=False,
                                 verdict="fail: no saved step met the 1.0 m half; test not read")
    assert failed["L40"] == failed["L55"]
    lines = capsys.readouterr().out.splitlines()
    assert "L55: fail: no saved step met the 1.0 m half; test not read" in lines
    assert sum(line.split(":")[0] in failed for line in lines) == 2


def test_without_failed_arms_the_table_has_no_failed_section(tmp_path):
    rel = [v_rec(tmp_path, "rv", 25.0, 0.85, 0.12), t_rec(tmp_path, "rt", 19.4, 0.66, 0.34)]
    out = tmp_path / "t.json"
    dm.main(["--released", *rel, "--model", "V", rel[0], "--out", str(out)])
    assert list(json.loads(out.read_text())) == ["models", "effects"]


@pytest.mark.parametrize("scored", [
    ["--model", "L55", "{val}", "{test}"],
    ["--model", "L55", "{val}", "{test}", "--effect", "L55", "LS"],
    ["--effect", "L55", "LS"],
])
def test_a_failed_arm_cannot_also_be_scored(tmp_path, scored):
    rel = [v_rec(tmp_path, "rv", 25.0, 0.85, 0.12), t_rec(tmp_path, "rt", 19.4, 0.66, 0.34)]
    fill = dict(val=rel[0], test=rel[1])
    args = ["--released", *rel, "--model", "LS", *rel] + [a.format(**fill) for a in scored]
    args += ["--failed-selection", "L55"]
    with pytest.raises(ValueError, match="failed arm"):
        dm.main(args)


@pytest.mark.parametrize("failed", [
    ["--failed-selection", "L55", "--failed-selection", "L55"],
    ["--failed-selection", ""],
])
def test_a_failed_name_is_given_once(tmp_path, failed):
    rel = [v_rec(tmp_path, "rv", 25.0, 0.85, 0.12), t_rec(tmp_path, "rt", 19.4, 0.66, 0.34)]
    with pytest.raises(ValueError, match="failed arm"):
        dm.main(["--released", *rel, *failed])

"""Decision numbers and keep clauses of the fine-tune rules (the LS recipe, README "LS fine-tune"), read from
evaluate.py records.

A record is the metrics.json of one `evaluate.py --out-dir`. The numbers:
  1.0 m  total.metrics of an mp3d_double_256_val --novel-only record (its only scene key m3d_0.1 is the loader's label
         for the 1.0 m val split; Total is read, not that key);
  2.0 m  scenes["m3d_1.0"].metrics of an mp3d_double_256 --novel-only record;
  1.5 m  scenes["m3d_0.75"].metrics of the same record (report-only flag).
Test Total mixes every baseline and is never read. A record of another dataset or protocol is refused.

Keep clauses against a reference set (per metric: the best WS-PSNR, the best SSIM and the best LPIPS of the set):
(i) dWS(1.0) >= +0.10 or dWS(2.0) >= +0.15; (ii) dWS(1.0) >= -0.05 and dWS(2.0) >= -0.10; (iii) dLP(1.0) <= +0.002,
dLP(2.0) <= +0.005, dSSIM(1.0) >= -0.002, dSSIM(2.0) >= -0.005; 1.5 m flag with the 2.0 m tolerances (report-only).

  python tools/decision_metrics.py --released VAL TEST [--c0 VAL TEST] [--model NAME VAL [TEST] ...] \
      [--effect TREAT CONTROL] [--failed-selection NAME ...] [--out table.json]

prints rule R (clauses (i)-(iii) against {released}) and the deltas against C0 (the plain continuation long_c0) for
every model, and the switch effect of TREAT over CONTROL (clauses (i)-(iii) against {CONTROL, released}).

A run whose selection was null (scripts/long_arm.sh exit 6: no saved step met the 1.0 m half, so test was not read) has
no scored records: --failed-selection NAME (repeatable) records it as passed: false in the "failed" section. A failed
NAME is refused as a --model NAME, as the first argument of --effect, and a second time as a failed NAME.
"""

import argparse
import json

METRICS = ("wspsnr", "ssim", "lpips")
VAL, TEST = "mp3d_double_256_val", "mp3d_double_256"
TEST_SCENES = {"2.0": "m3d_1.0", "1.5": "m3d_0.75"}
VTOL = {"wspsnr": -0.05, "ssim": -0.002, "lpips": 0.002}
EPS = 1e-9  # the thresholds are inclusive; absorbs the binary rounding of decimal deltas (19.4 + 0.15 - 19.4 < 0.15)


def _ge(delta, bound):
    return delta >= bound - EPS


def _le(delta, bound):
    return delta <= bound + EPS


def read_record(path, dataset):
    with open(path) as f:
        record = json.load(f)
    if record.get("dataset") != dataset or not record.get("protocol", {}).get("novel_only"):
        raise ValueError(f"{path}: not an {dataset} --novel-only record")
    return record


def val_1m(path):
    """1.0 m numbers: Total of an mp3d_double_256_val --novel-only record."""
    total = read_record(path, VAL)["total"]["metrics"]
    return {k: float(total[k]) for k in METRICS}


def test_baseline(path, scene):
    """Numbers of one test scene line (m3d_1.0 = 2.0 m, m3d_0.75 = 1.5 m) of an mp3d_double_256 --novel-only record."""
    scene_metrics = read_record(path, TEST)["scenes"][scene]["metrics"]
    return {k: float(scene_metrics[k]) for k in METRICS}


def numbers(val_path, test_path=None):
    """{baseline: {metric: value}}; 2.0 / 1.5 m only when a test record is given."""
    out = {"1.0": val_1m(val_path)}
    if test_path:
        out.update({b: test_baseline(test_path, scene) for b, scene in TEST_SCENES.items()})
    return out


def meets_vtol(x_1m, ref_1m):
    """V-tol: 1.0 m numbers within the val tolerances of a reference's."""
    d = {k: x_1m[k] - ref_1m[k] for k in METRICS}
    return _ge(d["wspsnr"], VTOL["wspsnr"]) and _ge(d["ssim"], VTOL["ssim"]) and _le(d["lpips"], VTOL["lpips"])


def reference(refs):
    """Per-metric reference of a set of models at every baseline all of them have."""
    out = {}
    for b in ("1.0", "2.0", "1.5"):
        if all(b in r for r in refs):
            out[b] = dict(wspsnr=max(r[b]["wspsnr"] for r in refs), ssim=max(r[b]["ssim"] for r in refs),
                          lpips=min(r[b]["lpips"] for r in refs))
    return out


def keep_clauses(x, refs):
    """Clauses (i)-(iii) of x against the reference set refs."""
    ref = reference(refs)
    d = {b: {k: x[b][k] - ref[b][k] for k in METRICS} for b in x if b in ref}
    if "2.0" not in d:
        return dict(deltas=d, passed=None, verdict="test not read")
    clause_gain = _ge(d["1.0"]["wspsnr"], 0.10) or _ge(d["2.0"]["wspsnr"], 0.15)
    no_regress = _ge(d["1.0"]["wspsnr"], -0.05) and _ge(d["2.0"]["wspsnr"], -0.10)
    tolerances = (_le(d["1.0"]["lpips"], 0.002) and _le(d["2.0"]["lpips"], 0.005) and _ge(d["1.0"]["ssim"], -0.002)
                  and _ge(d["2.0"]["ssim"], -0.005))
    flag = "1.5" in d and not (_ge(d["1.5"]["wspsnr"], -0.10) and _ge(d["1.5"]["ssim"], -0.005)
                               and _le(d["1.5"]["lpips"], 0.005))
    passed = clause_gain and no_regress and tolerances
    verdict = ("PASS" if passed else "fail") + (" (1.5 m flag)" if flag else "")
    return dict(deltas=d, gain=clause_gain, no_regress=no_regress, tolerances=tolerances, flag_1p5m=flag,
                passed=passed, verdict=verdict)


FAILED_SELECTION = "fail: no saved step met the 1.0 m half; test not read"


def failed_arms(failed_selection):
    """{NAME: row} of the runs given to --failed-selection; a NAME given twice is refused."""
    failed = {}
    for name in failed_selection:
        if not name or name in failed:
            raise ValueError(f"a failed arm needs a new non-empty NAME, got {name!r}")
        failed[name] = dict(stage="selection", passed=False, verdict=FAILED_SELECTION)
    return failed


def _fmt(deltas):
    return "  ".join(f"{b}: WS {v['wspsnr']:+.3f} SSIM {v['ssim']:+.4f} LPIPS {v['lpips']:+.4f}" for b, v in deltas.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--released", nargs=2, required=True, metavar=("VAL", "TEST"))
    parser.add_argument("--c0", nargs=2, metavar=("VAL", "TEST"))
    parser.add_argument("--model", nargs="+", action="append", default=[], metavar="NAME VAL [TEST]")
    parser.add_argument("--effect", nargs=2, action="append", default=[], metavar=("TREAT", "CONTROL"))
    parser.add_argument("--failed-selection", action="append", default=[], metavar="NAME",
                        help="an arm whose selection was null (no saved step met the 1.0 m half; test not read)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    failed = failed_arms(args.failed_selection)
    scored = {spec[0] for spec in args.model} | {treat for treat, _ in args.effect}
    if failed.keys() & scored:
        raise ValueError(f"a failed arm cannot also be scored (--model / --effect): {sorted(failed.keys() & scored)}")
    released = numbers(*args.released)
    c0 = numbers(*args.c0) if args.c0 else None
    models = {}
    for spec in args.model:
        if len(spec) not in (2, 3) or spec[0] in models:
            raise ValueError(f"--model NAME VAL [TEST] with a new NAME, got {spec}")
        models[spec[0]] = numbers(*spec[1:])
    table = dict(models={}, effects={})
    for name, x in models.items():
        row = dict(numbers=x, rule_r=keep_clauses(x, [released]))
        if c0:
            row["vs_c0"] = {b: {k: x[b][k] - c0[b][k] for k in METRICS} for b in x if b in c0}
        table["models"][name] = row
        print(f"{name}: rule R {row['rule_r']['verdict']}  {_fmt(row['rule_r']['deltas'])}")
        if c0:
            print(f"{'':{len(name)}}  vs C0: {_fmt(row['vs_c0'])}")
    for treat, control in args.effect:
        result = keep_clauses(models[treat], [models[control], released])
        table["effects"][f"{treat} over {control}"] = result
        print(f"switch effect {treat} over {control}: {result['verdict']}  {_fmt(result['deltas'])}")
    if failed:  # the section exists only when an arm failed, so a table without failed arms is unchanged
        table["failed"] = failed
    for name, row in failed.items():
        print(f"{name}: {row['verdict']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()

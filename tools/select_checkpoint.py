"""Pick one candidate (a saved step) from its val numbers, before test is read.

  python tools/select_checkpoint.py --released <record> [--reference <record> ...] --fallback best|none \
      --out <selection.json> KEY=<record> [KEY=<record> ...]

Every record is the metrics.json of an `evaluate.py --dataset mp3d_double_256_val --novel-only` run, read through
tools/decision_metrics.py (1.0 m = its Total line); KEY is numeric (a step). The reference is the released
checkpoint's numbers or, with --reference (repeatable), the per-metric best of --released and every --reference (max
WS-PSNR, max SSIM, min LPIPS; tools/decision_metrics.reference on the 1.0 m block). The choice is the largest KEY whose
numbers meet V-tol against the reference (WS-PSNR >= -0.05 dB, SSIM >= -0.002, LPIPS <= +0.002); if none does,
--fallback best takes the KEY with the best WS-PSNR and --fallback none takes nothing. Prints
`<KEY as given, or none> <reason>` with reason vtol (within V-tol), fallback (best WS-PSNR, none within V-tol) or none;
writes the decision, the reason and every candidate's deltas to --out, plus, with --reference, the reference paths and
the combined reference numbers (without --reference the output is the released-only one, unchanged). With
--reference, every candidate's deltas are taken against the combined reference, while the "released" entry keeps the
released record's own numbers.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.decision_metrics import METRICS, VTOL, meets_vtol, reference, val_1m  # noqa: E402


def select(candidates, released, fallback):
    """candidates: {key string: 1.0 m numbers}; released: the reference's 1.0 m numbers (the released checkpoint's, or
    the combined reference); returns (chosen key or None, reason, per-key report)."""
    report = {
        key: dict(metrics=x, deltas={k: x[k] - released[k] for k in METRICS}, meets_vtol=meets_vtol(x, released))
        for key, x in candidates.items()
    }
    passing = [key for key in candidates if report[key]["meets_vtol"]]
    if passing:
        return max(passing, key=float), "vtol", report
    if fallback == "best" and candidates:
        # ties go to the larger key, like the vtol branch
        return max(candidates, key=lambda key: (candidates[key]["wspsnr"], float(key))), "fallback", report
    return None, "none", report


def combined_reference(released, references):
    """Per-metric best (max WS-PSNR, max SSIM, min LPIPS) of the released and the reference 1.0 m numbers."""
    return reference([{"1.0": x} for x in [released, *references]])["1.0"]


def parse_candidate(text):
    key, sep, path = text.partition("=")
    if not sep or not path:
        raise ValueError(f"expected KEY=<record>, got {text!r}")
    float(key)  # must be numeric
    return key, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--released", required=True, help="the released checkpoint's val --novel-only record")
    parser.add_argument("--reference", action="append", default=[],
                        help="another val --novel-only record of the reference set (repeatable)")
    parser.add_argument("--fallback", choices=["best", "none"], required=True)
    parser.add_argument("--out", required=True, help="selection.json to write")
    parser.add_argument("candidates", nargs="+", help="KEY=<val --novel-only record>")
    args = parser.parse_args(argv)
    paths = dict(parse_candidate(c) for c in args.candidates)
    if len(paths) != len(args.candidates):
        raise ValueError("a KEY is given twice")
    released = val_1m(args.released)
    references = [dict(path=path, metrics=val_1m(path)) for path in args.reference]
    ref = combined_reference(released, [r["metrics"] for r in references]) if references else released
    chosen, reason, report = select({key: val_1m(path) for key, path in paths.items()}, ref, args.fallback)
    out = dict(chosen=chosen, reason=reason, tolerances=VTOL, released=dict(path=args.released, metrics=released))
    if references:
        out.update(references=references, reference=dict(metrics=ref))
    out["candidates"] = {key: dict(path=paths[key], **report[key]) for key in paths}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(chosen if chosen is not None else "none", reason)


if __name__ == "__main__":
    main()

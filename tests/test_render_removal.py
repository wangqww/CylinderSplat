"""C1: renders and debug PNG writes that no loss reads are gone, the rest is unchanged.

Each stage model (every switch off) and its frozen f7b20b9 copy run the same batch
through the CPU stubs of tests/test_switches_identity_models.py. The stub renderer records
every call with the Gaussians it was given. The live model makes only the renders its
losses read (plus, in 'val', the ones its validation images read), each on bitwise the
same Gaussians as the legacy call it keeps; the loss, the loss terms and the RNG state
after the step are identical; no debug PNG is written into the working directory.
forward_test makes the same single render as before, and the validation images are
byte-identical.
"""

import filecmp

import pytest
import torch

from tests.test_switches_identity_models import (  # noqa: F401  (harness is a fixture)
    KINDS, assert_same, dropped_slots, harness, make_batch, run,
)

R, O = "render", "orthographic"

# (kind, loss_kind, split) -> (legacy calls, live calls, legacy index of each live call)
EXPECTED = {
    ("all", "all", "train"): ([R, O, R, R], [R, R], [0, 3]),            # fused, volume
    ("all", "all_entropy", "train"): ([R, O, R, R], [R, R], [0, 3]),    # alpha-entropy reads the volume render
    ("all", "all", "val"): ([R, O, R, R], [R, R, R], [0, 2, 3]),       # + pixel for the validation images
    ("pan2", "pan2", "train"): ([R, R, O, R], [R], [3]),               # fused (then blended)
    ("pan2", "pan2_vol", "train"): ([R, R, O, R], [R, R], [0, 3]),     # volume losses on: volume, fused
    ("pan2", "pan2", "val"): ([R, R, O, R], [R, R, R], [0, 1, 3]),     # volume, pixel, fused
    ("volume", "volume", "train"): ([O, R, R], [R], [1]),              # volume
    ("volume", "volume", "val"): ([O, R, R], [R], [1]),
    ("pixel", "pixel", "train"): ([R, O], [R], [0]),                   # fused (= the pixel render)
    ("pixel", "pixel", "val"): ([R, O], [R], [0]),
}
# debug PNGs the released code writes into the cwd per forward() / forward_test() call
LEGACY_PNGS = {"all": 5, "pan2": 5, "volume": 4, "pixel": 3}
LEGACY_TEST_PNGS = {"all": 2, "pan2": 0, "volume": 2, "pixel": 0}


def call_kinds(model):
    return [c[0] for c in model.renderer.calls]


@pytest.mark.parametrize("case", sorted(EXPECTED), ids="-".join)
@pytest.mark.parametrize("v", [1, 2])
def test_forward_keeps_only_read_renders(harness, case, v):
    kind, loss_kind, split = case
    legacy_calls, live_calls, kept = EXPECTED[case]
    new = harness.build(kind, loss_kind=loss_kind)
    old = harness.build(kind, legacy=True, loss_kind=loss_kind)

    out_new, rng_new = run(new, make_batch(v), split=split)
    assert harness.pngs() == []
    out_old, rng_old = run(old, make_batch(v), split=split)
    assert len(harness.pngs()) == LEGACY_PNGS[kind]

    assert call_kinds(old) == legacy_calls
    assert call_kinds(new) == live_calls
    for (_, gaussians), i in zip(new.renderer.calls, kept):
        assert_same(gaussians, old.renderer.calls[i][1], f"render input (legacy call {i})")

    # the losses read the same tensors: same value, same terms, same RNG draws
    assert torch.equal(out_new[0], out_old[0])
    assert out_new[1] == out_old[1]
    assert torch.equal(rng_new, rng_old)
    for slot in dropped_slots(kind, loss_kind, split):
        assert out_new[slot] is None and out_old[slot] is not None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v", [1, 2])
def test_forward_test_renders_unchanged(harness, kind, v):
    new, old = harness.build(kind), harness.build(kind, legacy=True)
    with torch.no_grad():
        out_new, _ = run(new, make_batch(v), fn="forward_test")
        assert harness.pngs() == []
        out_old, _ = run(old, make_batch(v), fn="forward_test")
    assert len(harness.pngs()) == LEGACY_TEST_PNGS[kind]
    assert call_kinds(new) == call_kinds(old) == [R]
    assert_same(new.renderer.calls, old.renderer.calls, "render calls")
    assert_same(out_new, out_old, "forward_test")


@pytest.mark.parametrize("kind", KINDS)
def test_validation_images_unchanged(harness, kind):
    new, old = harness.build(kind), harness.build(kind, legacy=True)
    new_dir, old_dir = harness.tmp / "val_new", harness.tmp / "val_old"
    torch.manual_seed(1234)
    new.validation_step(make_batch(2), str(new_dir / "batch-0"))
    # the legacy forward's outputs through the same (unchanged) save_val_results
    out_old, _ = run(old, make_batch(2), split="val")
    new.save_val_results(out_old[8], out_old[2], out_old[3], out_old[4], out_old[5], out_old[6], out_old[7],
                         str(old_dir / "batch-0"))
    names = sorted(p.name for p in new_dir.iterdir())
    assert names == sorted(p.name for p in old_dir.iterdir())
    assert len(names) == 2 * 3 * 3            # samples x target views x (omni, pixel, volume)
    match, mismatch, errors = filecmp.cmpfiles(new_dir, old_dir, names, shallow=False)
    assert mismatch == [] and errors == []

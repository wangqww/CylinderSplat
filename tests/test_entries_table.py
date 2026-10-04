"""Training rows (configs/entries.py) and train.py, on CPU.

The table: every config is trained by a row, and each row's fields agree. train.py: build_dataloader against
each loader module's own load_*() factory (dataset class stubbed), then fake runs of train.main with
Accelerate, the model builder and the dataset stubbed: per row the process count, setup order, loader
keywords, scheduler, forward and validation calls and saves; then ddp_forward, loc360_interleave,
depth_valid_mask, --save-final, --max-steps, --profile-steps and resuming before prepare(). For the
multi-process rows, rank 0 and rank 1 run one after the other and their barrier sequences are compared.
"""

import contextlib
import importlib
import json
import os
import textwrap
import types

import pytest

from tests.conftest import REPO_ROOT, kept_configs, load_by_path

TABLE = load_by_path("configs/entries.py", "cylindersplat_entries_under_test")
ENTRIES, STAGE4 = TABLE.ENTRIES, TABLE.STAGE4_ENTRIES
ALL_ENTRIES = dict(ENTRIES, **STAGE4)
ROWS = sorted(ALL_ENTRIES)
MULTI_PROCESS_ROWS = [row for row in ROWS if ALL_ENTRIES[row]["num_processes"] > 1]
SETUP_ORDER = {
    "loaders_before_model": ["set_seed", "loaders", "init_trackers", "model", "scheduler", "prepare"],
    "model_before_loaders": ["init_trackers", "set_seed", "model", "scheduler", "loaders", "prepare"],
}
SCHEDULER_BUILDERS = {"onecycle": "build_onecycle_scheduler", "warmup_cosine": "build_warmup_cosine_scheduler"}


# ----------------------------------------------------------------------------- the table


def test_rows():
    got = {
        row: (e["num_processes"], e["train_forward"], e["validation"], e["scheduler"]) for row, e in ALL_ENTRIES.items()
    }
    assert got == {
        "mp3d_double_256": (3, "module", "module", "onecycle"),
        "mp3d_single_256": (3, "module", "module", "onecycle"),
        "loc360_all_256": (3, "module", None, "onecycle"),
        "mp3d_double_512": (1, "plain", "plain", "warmup_cosine"),
        "mp3d_double_160": (1, "plain", "plain", "warmup_cosine"),
        "kansas_double_160": (1, "plain", "plain", "warmup_cosine"),
        "mp3d_double_512_ddp3": (3, "module", "module", "onecycle"),
        "mp3d_double_512_ddp4": (4, "module", "module", "onecycle"),
    }


def test_every_config_is_trained_by_a_row():
    assert sorted({cfg for e in ALL_ENTRIES.values() for cfg in e["configs"]}) == kept_configs()


@pytest.mark.parametrize("row", ROWS)
def test_row_is_consistent(row):
    e = ALL_ENTRIES[row]
    loader = e["loader"]
    assert os.path.isfile(os.path.join(REPO_ROOT, loader["module"].replace(".", "/") + ".py"))
    assert e["scheduler"] in TABLE.SCHEDULERS and e["setup_order"] in SETUP_ORDER
    # OneCycle needs len(train_dataloader), so its rows build the loaders first.
    assert (e["scheduler"] == "onecycle") == (e["setup_order"] == "loaders_before_model")
    # .module exists only on the DDP wrapper, i.e. with more than one process.
    assert (e["train_forward"] == "module") == (e["num_processes"] > 1)
    assert e["validation"] in (None, e["train_forward"])
    assert loader["num_workers"] == (1 if loader["dataset_class"] == "Dataset360Loc" else 32)
    assert loader["shuffle"] is False and loader["call"]["shuffle"] == "False"
    assert set(loader["dataset_kwargs"]) == set(e["batch_size"]) == {"train", "val"}


@pytest.mark.parametrize("row", sorted(STAGE4))
def test_stage4_row_is_the_512_loader_with_the_256_recipe(row):
    s = STAGE4[row]
    assert s["stage4"] == dict(loader_row="mp3d_double_512", recipe_row="mp3d_double_256", resolution=[512, 1024])
    assert s["loader"] is ENTRIES["mp3d_double_512"]["loader"]
    assert s["batch_size"] == ENTRIES["mp3d_double_512"]["batch_size"]
    for key in ("scheduler", "setup_order", "train_forward", "validation"):
        assert s[key] == ENTRIES["mp3d_double_256"][key], key


# ----------------------------------------------------------------------------- build_dataloader


def train_module():
    pytest.importorskip("torch")
    import train

    return train


def dummy_dataset_cls(iterable, records):
    torch = pytest.importorskip("torch")
    base = torch.utils.data.IterableDataset if iterable else torch.utils.data.Dataset

    class Dummy(base):
        def __init__(self, **kwargs):
            records.append(kwargs)
            self.kwargs = kwargs

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return torch.ones(2)

        def __iter__(self):
            return iter([torch.ones(2)] * 10)

    return Dummy


@pytest.mark.parametrize("row", sorted(ENTRIES))  # the stage-4 rows share the mp3d_double_512 loader
@pytest.mark.parametrize("stage", ["train", "val"])
def test_build_dataloader_equals_the_loader_factory(row, stage, monkeypatch):
    torch = pytest.importorskip("torch")
    train = train_module()
    loader = ENTRIES[row]["loader"]
    try:
        module = importlib.import_module(loader["module"])
    except ImportError as e:  # the loaders import model.utils.ops, i.e. the whole model package
        pytest.skip(f"loader not importable: {e}")
    is_iterable = issubclass(getattr(module, loader["dataset_class"]), torch.utils.data.IterableDataset)
    assert is_iterable == loader["iterable"]
    records = []
    monkeypatch.setattr(module, loader["dataset_class"], dummy_dataset_cls(loader["iterable"], records))
    reference = getattr(module, loader["factory"])(3, stage=stage)
    built = train.build_dataloader(ENTRIES[row], module, stage, 3)
    assert records == [loader["dataset_kwargs"][stage]] * 2
    for attr in (
        "batch_size",
        "num_workers",
        "persistent_workers",
        "drop_last",
        "pin_memory",
        "timeout",
        "prefetch_factor",
        "worker_init_fn",
        "collate_fn",
    ):
        assert getattr(built, attr) == getattr(reference, attr), attr
    assert type(built.sampler) is type(reference.sampler)
    assert built.generator.initial_seed() == reference.generator.initial_seed() == loader["stages"][stage]["seed"]
    assert torch.equal(built.generator.get_state(), reference.generator.get_state())


# ----------------------------------------------------------------------------- fake runs of train.main

FAKE_CONFIG = """
exp_name = "fake"
lr = 2e-4
grad_max_norm = 1.0
print_freq = 1
save_freq = 2
val_freq = 2
max_epochs = 2
max_train_steps = 100
warmup_steps = 10
mixed_precision = "no"
gradient_accumulation_steps = 1
resume_from = ""
report_to = "tensorboard"
seed = 0
dataset_params = dict(batch_size_train=2, batch_size_val=1)
model = dict(type="FakeModel")
"""
BATCH_SIZES = {"batch_size_train": 2, "batch_size_val": 1}
N_TRAIN, N_VAL = 5, 2  # batches the fake prepare() hands back per epoch
STEPS = 2 * N_TRAIN  # max_epochs = 2
SAVED_STEPS = [2, 4, 6, 8]  # save_freq = val_freq = 2; step 0 is never saved


def fakes(torch, events, num_processes, process_index):
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 1)
            self.head = torch.nn.Linear(2, 1)
            self.register_forward_pre_hook(lambda module, inputs: events.append(("call_hook",)))

        def forward(self, batch, split="train", iter=0, iter_end=100000):
            events.append(("forward", split, iter, iter_end, os.getcwd(), torch.is_grad_enabled()))
            loss = (self.backbone(batch) + self.head(batch)).sum()
            return loss, {"loss": loss.item()}, None, None, None, None, None, None, None

        def validation_step(self, batch, val_result_savedir):
            events.append(("validation_step", val_result_savedir, torch.is_grad_enabled(), self.training))
            return {"val_psnr": 1.0}

        def configure_optimizers(self, lr):
            optimizer = torch.optim.AdamW(
                [{"params": self.head.parameters()}, {"params": self.backbone.parameters(), "lr": lr * 0.1}], lr=lr
            )
            events.append(("configure_optimizers", lr, optimizer))
            return [optimizer]

    class FakeDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            events.append(("ddp_forward",))
            return self.module(*args, **kwargs)

    class FakeAccelerator:
        def __init__(self, **kwargs):
            events.append(("Accelerator", kwargs))
            self.num_processes = num_processes
            self.process_index = self.local_process_index = process_index
            self.is_main_process = process_index == 0
            self.device = torch.device("cpu")
            self.sync_gradients = True

        def init_trackers(self, project_name, init_kwargs=None):
            events.append(("init_trackers",))

        def prepare(self, model, optimizer, train_dl, val_dl, scheduler):
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            events.append(("prepare", model, optimizer, train_dl, val_dl, scheduler, state))
            wrapped = FakeDDP(model) if self.num_processes > 1 else model
            return wrapped, optimizer, [torch.ones(2)] * N_TRAIN, [torch.ones(2)] * N_VAL, scheduler

        @contextlib.contextmanager
        def accumulate(self, model):
            yield

        def backward(self, loss):
            loss.backward()

        def clip_grad_norm_(self, params, max_norm):
            return torch.nn.utils.clip_grad_norm_(params, max_norm)

        def wait_for_everyone(self):
            events.append(("wait_for_everyone",))

        def save_state(self, output_dir):
            events.append(("save_state", output_dir))
            os.makedirs(output_dir, exist_ok=True)

        def log(self, values, step=None):
            pass

        def print(self, *args):
            pass

        def end_training(self):
            events.append(("end_training",))

    return FakeModel, FakeAccelerator


def plain(node):
    return {k: plain(v) for k, v in node.items()} if isinstance(node, dict) else node


def record_calls(monkeypatch, module, name, record):
    """Patch module.<name> to call the original (never an earlier run's wrapper) and record its result."""
    original = getattr(module, name)
    original = getattr(original, "__wrapped__", original)

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        record(args, result)
        return result

    wrapper.__wrapped__ = original
    monkeypatch.setattr(module, name, wrapper)


def run_fake(
    monkeypatch, tmp_path, row, args=(), num_processes=None, process_index=0, work_dir=None, config=FAKE_CONFIG
):
    """Run train.main for `row` with Accelerate, the model builder and the dataset stubbed.

    Returns (events, work dir, model). The loaders are real DataLoaders from train.build_dataloader over a
    dummy dataset; the fake prepare() hands back N_TRAIN / N_VAL tensors instead. process_index != 0 runs
    a non-main rank.
    """
    torch = pytest.importorskip("torch")
    train = train_module()
    entry = ALL_ENTRIES[row]
    events = []
    FakeModel, FakeAccelerator = fakes(torch, events, num_processes or entry["num_processes"], process_index)
    model = FakeModel()

    def build_model(cfg, accelerator):
        events.append(("build_model", plain(cfg.model)))
        return model

    loader_stub = types.SimpleNamespace(
        get_generator=lambda seed: torch.Generator().manual_seed(seed),
        worker_init_fn=lambda worker_id: None,
        **{entry["loader"]["dataset_class"]: dummy_dataset_cls(entry["loader"]["iterable"], [])},
    )
    # build_loaders calls build_dataloader(entry, module, stage, batch_size[, dataset_extra]).
    record_calls(monkeypatch, train, "build_dataloader", lambda a, dl: events.append(("loader", a[2], a[3], dl)))
    for name in SCHEDULER_BUILDERS.values():
        record_calls(monkeypatch, train, name, lambda a, sched, name=name: events.append(("scheduler", name, sched)))
    monkeypatch.setattr(train, "Accelerator", FakeAccelerator)
    monkeypatch.setattr(train, "set_seed", lambda seed: events.append(("set_seed", seed)))
    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.chdir(tmp_path)  # train.main moves into <work dir>/cwd; restored at teardown

    cfg_path = tmp_path / "fake_cfg.py"
    cfg_path.write_text(textwrap.dedent(config))
    work = str(work_dir or tmp_path / "runs" / row)
    train.main(
        train.parse_args(["--entry", row, "--py-config", str(cfg_path), "--work-dir", work, *args]), entry, loader_stub
    )
    return events, work, model


def names(events):
    return [ev[0] for ev in events]


def setup_sequence(events):
    seq = []
    for ev in events:
        if ev[0] == "loader" and ev[1] == "train":
            seq.append("loaders")
        elif ev[0] == "build_model":
            seq.append("model")
        elif ev[0] in ("set_seed", "init_trackers", "scheduler", "prepare"):
            seq.append(ev[0])
    return seq


def read_switches(work):
    with open(os.path.join(work, "switches.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("row", ROWS)
def test_fake_run(row, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    e = ALL_ENTRIES[row]
    events, work, model = run_fake(monkeypatch, tmp_path, row)
    assert setup_sequence(events) == SETUP_ORDER[e["setup_order"]]
    assert ("set_seed", 0) in events

    # One InitProcessGroupKwargs (1800 s); no DDP keywords without ddp_forward.
    (acc,) = [ev[1] for ev in events if ev[0] == "Accelerator"]
    assert [type(h).__name__ for h in acc["kwargs_handlers"]] == ["InitProcessGroupKwargs"]
    assert acc["kwargs_handlers"][0].timeout.total_seconds() == 1800
    assert acc["project_config"].project_dir == work

    # Loaders: the row's batch-size keys, dataset keywords, workers and stage seeds; read in order.
    loaders = {ev[1]: ev for ev in events if ev[0] == "loader"}
    for stage in ("train", "val"):
        _, _, batch_size, dl = loaders[stage]
        assert batch_size == BATCH_SIZES[e["batch_size"][stage]]
        assert dl.dataset.kwargs == e["loader"]["dataset_kwargs"][stage]
        assert dl.num_workers == e["loader"]["num_workers"]
        assert dl.generator.initial_seed() == e["loader"]["stages"][stage]["seed"]
        assert dl.persistent_workers is e["loader"]["stages"][stage]["persistent_workers"]
        if not e["loader"]["iterable"]:
            assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

    # The row's scheduler, prepared together with the model, the optimizer and both loaders.
    (sched_ev,) = [ev for ev in events if ev[0] == "scheduler"]
    (opt_ev,) = [ev for ev in events if ev[0] == "configure_optimizers"]
    (prep,) = [ev for ev in events if ev[0] == "prepare"]
    assert sched_ev[1] == SCHEDULER_BUILDERS[e["scheduler"]] and opt_ev[1] == 2e-4
    prepared = (model, opt_ev[2], loaders["train"][3], loaders["val"][3], sched_ev[2])
    assert all(a is b for a, b in zip(prep[1:6], prepared))
    scheduler = sched_ev[2]
    if e["scheduler"] == "onecycle":
        assert type(scheduler).__name__ == "OneCycleLR"
        assert scheduler.total_steps == len(loaders["train"][3]) * 2 + 100  # max_epochs = 2
    else:
        assert type(scheduler).__name__ == "SequentialLR"
        assert scheduler._milestones == [10 * e["num_processes"]]  # warmup_steps x processes

    # Training forward every step through .module.forward / .forward (never the wrapper or __call__),
    # with gradients, from <work dir>/cwd.
    forwards = [ev for ev in events if ev[0] == "forward"]
    assert [ev[2] for ev in forwards] == list(range(STEPS))
    assert all(ev[1] == "train" and ev[3] == 100 and ev[5] for ev in forwards)
    assert all(os.path.realpath(ev[4]) == os.path.realpath(os.path.join(work, "cwd")) for ev in forwards)
    assert "ddp_forward" not in names(events) and "call_hook" not in names(events)

    # Validation every val_freq through the row's call, under no_grad in eval mode, into <work dir>/validation.
    vals = [ev for ev in events if ev[0] == "validation_step"]
    expected = [os.path.join(work, "validation", f"step-{s}/batch-{b}") for s in SAVED_STEPS for b in range(N_VAL)]
    assert [ev[1] for ev in vals] == ([] if e["validation"] is None else expected)
    assert all(ev[2] is False and ev[3] is False for ev in vals)
    assert model.training

    # Saves every save_freq; `latest` points at the last one; switches.json and the config dump.
    saves = [ev[1] for ev in events if ev[0] == "save_state"]
    assert saves == [os.path.join(work, f"checkpoint-{s}") for s in SAVED_STEPS]
    assert os.path.realpath(os.path.join(work, "latest")) == os.path.realpath(saves[-1])
    assert read_switches(work) == dict(ddp_forward=False, loc360_interleave=False, depth_valid_mask=False)
    assert os.path.isfile(os.path.join(work, "fake_cfg.py"))
    assert names(events)[-1] == "end_training"


@pytest.mark.parametrize("row", ROWS)
def test_other_process_counts_are_refused(row, monkeypatch, tmp_path):
    n = ALL_ENTRIES[row]["num_processes"]
    work = tmp_path / "runs" / "wrong_count"
    with pytest.raises(SystemExit, match=f"with {n} process"):
        run_fake(monkeypatch, tmp_path, row, num_processes=1 if n > 1 else 3, work_dir=work)
    assert not work.exists()  # refused before anything is written


@pytest.mark.parametrize("row", ["mp3d_double_256", "mp3d_double_160"])
def test_ddp_forward(row, monkeypatch, tmp_path):
    events, work, _ = run_fake(monkeypatch, tmp_path, row, ["--switch", "ddp_forward=true"])
    event_names = names(events)
    # The forward goes through the model's __call__, and through the DDP wrapper when there is one.
    before = ["ddp_forward", "call_hook"] if ALL_ENTRIES[row]["num_processes"] > 1 else ["call_hook"]
    forwards = [i for i, name in enumerate(event_names) if name == "forward"]
    assert len(forwards) == STEPS
    assert all(event_names[i - len(before) : i] == before for i in forwards)
    (acc,) = [ev[1] for ev in events if ev[0] == "Accelerator"]
    handlers = acc["kwargs_handlers"]
    assert [type(h).__name__ for h in handlers] == ["InitProcessGroupKwargs", "DistributedDataParallelKwargs"]
    assert handlers[1].find_unused_parameters is True
    assert event_names.count("validation_step") == len(SAVED_STEPS) * N_VAL  # still the row's call
    assert read_switches(work)["ddp_forward"] is True


def sync_skeleton(events):
    """The events that order the ranks: training forwards, barriers, rank 0's saves and validation."""
    out = []
    for ev in events:
        if ev[0] == "forward":
            out.append(("forward", ev[2]))
        elif ev[0] == "wait_for_everyone":
            out.append(("barrier",))
        elif ev[0] == "save_state":
            out.append(("save", os.path.basename(ev[1])))
        elif ev[0] == "validation_step":
            out.append(("validate", ev[1].split(os.sep)[-2]))
    return out


def expected_skeleton(row, rank, ddp):
    """Per step: forward, barrier, then rank 0's save / validation every second step; with ddp_forward one
    more barrier, so no rank starts the next (collective) forward while rank 0 saves or validates."""
    out = []
    for step in range(STEPS):
        out += [("forward", step), ("barrier",)]
        if rank == 0 and step in SAVED_STEPS:
            out.append(("save", f"checkpoint-{step}"))
            if ALL_ENTRIES[row]["validation"] is not None:
                out += [("validate", f"step-{step}")] * N_VAL
        if ddp:
            out.append(("barrier",))
    return out + [("barrier",)]  # before end_training


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("row", MULTI_PROCESS_ROWS)
def test_barriers_of_rank_0_and_rank_1(row, ddp, monkeypatch, tmp_path):
    switch = ["--switch", "ddp_forward=true"] if ddp else []
    for rank in (0, 1):
        events, _, _ = run_fake(
            monkeypatch, tmp_path, row, switch, process_index=rank, work_dir=tmp_path / "runs" / "ranks"
        )
        assert sync_skeleton(events) == expected_skeleton(row, rank, ddp)
        assert names(events).count("ddp_forward") == (STEPS if ddp else 0)


def test_loc360_interleave(monkeypatch, tmp_path):
    # Only the train dataset gets interleave=True; the val dataset keeps the row's keywords.
    events, work, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256", ["--switch", "loc360_interleave=true"])
    loaders = {ev[1]: ev[3] for ev in events if ev[0] == "loader"}
    assert loaders["train"].dataset.kwargs == {"stage": "train", "interleave": True}
    assert loaders["val"].dataset.kwargs == {"stage": "val"}
    assert read_switches(work)["loc360_interleave"] is True


@pytest.mark.parametrize("row", [r for r in ROWS if ALL_ENTRIES[r]["loader"]["dataset_class"] != "Dataset360Loc"])
def test_loc360_interleave_is_refused_for_other_loaders(row, monkeypatch, tmp_path):
    work = tmp_path / "runs" / "refused"
    with pytest.raises(SystemExit, match="loc360_interleave needs the 360Loc loader"):
        run_fake(monkeypatch, tmp_path, row, ["--switch", "loc360_interleave=true"], work_dir=work)
    assert not work.exists()


def test_model_switches_reach_the_built_model(monkeypatch, tmp_path):
    from tools import switches

    monkeypatch.setattr(switches, "_registered_class", lambda type_name: None)  # FakeModel is not registered
    events, work, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256", ["--switch", "depth_valid_mask=true"])
    assert [ev[1] for ev in events if ev[0] == "build_model"] == [{"type": "FakeModel", "depth_valid_mask": True}]
    assert read_switches(work)["depth_valid_mask"] is True
    events, _, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256", work_dir=tmp_path / "runs" / "default")
    assert [ev[1] for ev in events if ev[0] == "build_model"] == [{"type": "FakeModel"}]


def test_save_final(monkeypatch, tmp_path):
    # The periodic saves, then the final weights as checkpoint-<steps> before end_training; only on rank 0.
    events, work, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256", ["--save-final"])
    saves = [ev[1] for ev in events if ev[0] == "save_state"]
    assert saves == [os.path.join(work, f"checkpoint-{s}") for s in SAVED_STEPS + [STEPS]]
    assert names(events)[-3:] == ["save_state", "wait_for_everyone", "end_training"]
    assert os.path.realpath(os.path.join(work, "latest")) == os.path.realpath(saves[-1])
    events, _, _ = run_fake(
        monkeypatch, tmp_path, "loc360_all_256", ["--save-final"], process_index=1, work_dir=tmp_path / "runs" / "rank1"
    )
    assert "save_state" not in names(events)


def test_max_steps_and_profile_steps(monkeypatch, tmp_path):
    events, work, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_512", ["--max-steps", "3", "--profile-steps", "2"])
    assert names(events).count("forward") == 3
    with open(os.path.join(work, "profile_steps_rank0.json")) as f:
        profile = json.load(f)
    assert profile["steps"] == 2 and len(profile["step_ms"]) == 2


def test_resume_loads_weights_before_prepare(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from tools.resume import ResumeError

    source = {
        "backbone.weight": torch.full((1, 2), 3.0),
        "backbone.bias": torch.full((1,), 4.0),
        "head.weight": torch.full((1, 2), 5.0),
        "head.bias": torch.full((1,), 6.0),
    }
    ckpt = tmp_path / "src" / "checkpoint-3000"
    ckpt.mkdir(parents=True)
    save_file(source, str(ckpt / "model.safetensors"))
    events, _, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_256", ["--resume-from", str(ckpt)])
    (prep,) = [ev for ev in events if ev[0] == "prepare"]
    assert all(torch.equal(prep[6][k], v) for k, v in source.items())

    # An extra name is refused under the default transfer 'exact' and accepted where the transfer allows it.
    save_file(dict(source, **{"pixel_gs.mono_depth.w": torch.zeros(1)}), str(ckpt / "model.safetensors"))
    with pytest.raises(ResumeError, match="pixel_gs.mono_depth.w"):
        run_fake(
            monkeypatch, tmp_path, "mp3d_double_256", ["--resume-from", str(ckpt)], work_dir=tmp_path / "runs" / "exact"
        )
    events, _, _ = run_fake(
        monkeypatch,
        tmp_path,
        "mp3d_double_256",
        ["--resume-from", str(ckpt), "--transfer", "stage1_to_stage2"],
        work_dir=tmp_path / "runs" / "stage12",
    )
    assert "prepare" in names(events)


def test_no_resume_ignores_the_config(monkeypatch, tmp_path):
    from tools.resume import ResumeError

    config = FAKE_CONFIG.replace('resume_from = ""', 'resume_from = "/nonexistent/checkpoint-1"')
    assert config != FAKE_CONFIG
    with pytest.raises(ResumeError, match="not found"):
        run_fake(monkeypatch, tmp_path, "mp3d_double_160", config=config, work_dir=tmp_path / "runs" / "resume")
    events, _, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_160", ["--no-resume"], config=config)
    assert "prepare" in names(events)


def test_parse_args(monkeypatch, tmp_path):
    train = train_module()
    assert train.ENTRIES == ALL_ENTRIES and train.TRANSFERS == TABLE.TRANSFERS
    monkeypatch.setattr(train, "RUNS_ROOT", str(tmp_path / "runs"))
    base = ["--entry", "mp3d_double_256", "--py-config", "c.py"]
    args = train.parse_args(base + ["--run-id", "r1"])
    assert args.work_dir == str(tmp_path / "runs" / "r1")
    assert (args.transfer, args.switch, args.max_steps, args.profile_steps, args.save_final) == (
        "exact",
        [],
        0,
        0,
        False,
    )
    assert train.parse_args(base + ["--run-id", "r1", "--transfer", "stage1_to_stage2"]).transfer == "stage1_to_stage2"
    for bad in (
        [],
        ["--run-id", "../r1"],
        ["--run-id", "r1", "--transfer", "nope"],
        ["--run-id", "r1", "--resume-from", "a", "--no-resume"],
    ):
        with pytest.raises(SystemExit):
            train.parse_args(base + bad)

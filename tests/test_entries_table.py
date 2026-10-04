"""Table T (configs/entries.py) against the live legacy sources and against train.py.

Source checks parse with `ast`, so comments (the trainers keep several commented-out
schedulers and DataLoaders) never count: each loader's live DataLoader(...) call and
its per-stage seeds, each legacy trainer's scheduler assignments, forward call,
validation call, loader batch sizes, seeding and setup order. train.py is checked the
same way (keyword sets, scheduler calls, forward / validation calls), then at run time
on CPU: train.build_dataloader against the live load_*() factory with the dataset
class stubbed, the schedulers' LR against the frozen legacy copy, and a stubbed run of
train.main per row (setup order, forward mode, validation mode, saves, process-count
refusal, switches D1 / D1b, --max-steps, --profile-steps, resume before prepare); for the
multi-process rows, one stubbed run per simulated rank (0 and 1) whose barrier order shows
that with D1 no rank enters the next DDP forward while rank 0 saves or validates.
The Phase-2 screen rows (SCREEN_ENTRIES) are checked against their base table-T row,
the legacy OneCycle and the same stubbed run (step budget, final save, --screen-steps,
--seed); the D3 config against the all_256 config it inherits from. The stage-4 rows
(STAGE4_ENTRIES) are checked against the two table-T rows they are built from (the
mp3d_double_512 loader, the mp3d_double_256 recipe), with the same stubbed runs, and the
512x1024 loader against the batch keys the joint model reads.
"""

import ast
import contextlib
import importlib
import importlib.util
import json
import os
import textwrap
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_entries():
    spec = importlib.util.spec_from_file_location("cylindersplat_entries_under_test",
                                                  os.path.join(REPO, "configs", "entries.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRIES_MOD = _load_entries()
ENTRIES = ENTRIES_MOD.ENTRIES
ROWS = sorted(ENTRIES)
TABLE_T_ROWS = ["kansas_double_160", "loc360_all_256", "mp3d_double_160", "mp3d_double_256",
                "mp3d_double_512", "mp3d_single_256"]
SCREEN = ENTRIES_MOD.SCREEN_ENTRIES
SCREEN_ROWS = ["screen_d1_pair_256", "screen_loc360_all_256", "screen_mp3d_all_256", "screen_mp3d_single_256"]
STAGE4 = ENTRIES_MOD.STAGE4_ENTRIES
STAGE4_ROWS = ["mp3d_double_512_ddp3", "mp3d_double_512_ddp4"]
ALL_ENTRIES = dict(ENTRIES, **SCREEN, **STAGE4)
EXPECTED_ORDER = {
    "loaders_before_model": ["set_seed", "loaders", "init_trackers", "model", "scheduler", "resume", "prepare"],
    "model_before_loaders": ["init_trackers", "set_seed", "model", "scheduler", "loaders", "resume", "prepare"],
}
TRAIN_SCHEDULER_FUNCS = {"onecycle": "build_onecycle_scheduler", "warmup_cosine": "build_warmup_cosine_scheduler",
                         "onecycle_screen": "build_onecycle_screen_scheduler"}


# ----------------------------------------------------------------------------- AST helpers

def _tree(rel):
    with open(os.path.join(REPO, rel)) as f:
        return ast.parse(f.read(), filename=rel)


def _func(tree, name):
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f"{name}: {len(found)} definitions"
    return found[0]


def _calls(node, func):
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == func),
                  key=lambda n: (n.lineno, n.col_offset))


def _sig(call):
    return [ast.unparse(a) for a in call.args], {k.arg: ast.unparse(k.value) for k in call.keywords}


def _eval(node, env):
    return eval(compile(ast.Expression(body=node), "<ast>", "eval"), {"__builtins__": {}}, dict(env))


def _loader_file(entry):
    return entry["loader"]["module"].replace(".", "/") + ".py"


def _trace_factory(entry, stage):
    """Follow the load_*() body for one stage: literal assignments into env, calls by target name."""
    fn = _func(_tree(_loader_file(entry)), entry["loader"]["factory"])
    env = {"stage": stage}
    calls = {}

    def run(stmts):
        for st in stmts:
            if isinstance(st, ast.If):
                run(st.body if _eval(st.test, env) else st.orelse)
            elif isinstance(st, ast.Assign):
                (target,) = st.targets
                if isinstance(st.value, ast.Call):
                    calls[target.id] = st.value
                else:
                    env[target.id] = _eval(st.value, env)

    run(fn.body)
    return env, calls


def _scheduler_assigns(fn):
    out = []
    for n in sorted((n for n in ast.walk(fn) if isinstance(n, ast.Assign)), key=lambda n: n.lineno):
        if isinstance(n.value, ast.Call) and ast.unparse(n.value.func).startswith("torch.optim.lr_scheduler."):
            args, kwargs = _sig(n.value)
            out.append(dict(target=ast.unparse(n.targets[0]), call=ast.unparse(n.value.func), args=args, kwargs=kwargs))
    return out


def _forward_calls(fn):
    return [c for c in ast.walk(fn) if isinstance(c, ast.Call)
            and ast.unparse(c.func) in ("my_model.module.forward", "my_model.forward", "my_model")]


def _validation_calls(fn):
    return [c for c in ast.walk(fn) if isinstance(c, ast.Call)
            and ast.unparse(c.func) in ("my_model.module.validation_step", "my_model.validation_step")]


def _main_block(tree):
    blocks = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    assert len(blocks) == 1
    return blocks[0]


# ----------------------------------------------------------------------------- the table itself

def test_table_has_the_six_rows_of_table_t():
    assert ROWS == TABLE_T_ROWS


@pytest.mark.parametrize("row", ROWS)
def test_entry_is_consistent(row):
    e = ENTRIES[row]
    assert os.path.isfile(os.path.join(REPO, e["legacy_script"]))
    assert os.path.isfile(os.path.join(REPO, _loader_file(e)))
    for cfg in e["configs"]:
        assert os.path.isfile(os.path.join(REPO, cfg)), cfg
    assert e["scheduler"] in ENTRIES_MOD.SCHEDULERS
    assert e["setup_order"] in EXPECTED_ORDER
    # OneCycle needs len(train_dataloader), so its rows build the loaders first.
    assert (e["scheduler"] == "onecycle") == (e["setup_order"] == "loaders_before_model")
    assert e["train_forward"] in ("module", "plain")
    # .module exists only on the DDP wrapper, i.e. with more than one process.
    assert (e["train_forward"] == "module") == (e["num_processes"] > 1)
    assert e["validation"] in (None, "module", "plain")
    assert e["validation"] in (None, e["train_forward"])
    assert e["loader"]["shuffle"] is False
    # C3 / table T column 8: 1 worker for 360Loc, 32 elsewhere.
    assert e["loader"]["num_workers"] == (1 if row == "loc360_all_256" else 32)
    assert set(e["loader"]["dataset_kwargs"]) == {"train", "val"}
    assert set(e["batch_size"]) == {"train", "val"}


def test_table_t_process_counts_and_modes():
    got = {r: (ENTRIES[r]["num_processes"], ENTRIES[r]["train_forward"], ENTRIES[r]["validation"]) for r in ROWS}
    assert got == {
        "mp3d_double_256": (3, "module", "module"),
        "mp3d_single_256": (3, "module", "module"),
        "loc360_all_256": (3, "module", None),
        "mp3d_double_512": (1, "plain", "plain"),
        "mp3d_double_160": (1, "plain", "plain"),
        "kansas_double_160": (1, "plain", "plain"),
    }


def _names_dataset_name(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "dataset_name":
            return True
        if isinstance(n, ast.Constant) and n.value == "dataset_name":
            return True
        if isinstance(n, ast.Name) and n.id == "dataset_name":
            return True
    return False


def test_nothing_reads_dataset_name():
    assert not _names_dataset_name(_tree("train.py"))
    assert not _names_dataset_name(_tree("configs/entries.py"))


# ----------------------------------------------------------------------------- Phase-2 screen rows (the table)

def test_screen_rows_are_separate_from_table_t():
    assert sorted(SCREEN) == SCREEN_ROWS
    assert not set(SCREEN) & set(ENTRIES)
    got = {r: (SCREEN[r]["screen"]["base"], SCREEN[r]["num_processes"], SCREEN[r]["train_forward"],
               SCREEN[r]["validation"]) for r in SCREEN_ROWS}
    assert got == {
        "screen_mp3d_all_256": ("mp3d_double_256", 1, "plain", "plain"),
        "screen_loc360_all_256": ("loc360_all_256", 1, "plain", None),
        "screen_d1_pair_256": ("mp3d_double_256", 2, "module", "module"),
        "screen_mp3d_single_256": ("mp3d_single_256", 1, "plain", "plain"),
    }


@pytest.mark.parametrize("row", SCREEN_ROWS)
def test_screen_row_is_its_base_row_with_the_recipe(row):
    s = SCREEN[row]
    base = ENTRIES[s["screen"]["base"]]
    assert set(s) == set(base) | {"screen"}
    # The same loader dict (dataset, DataLoader keywords, workers, stage seeds), batch sizes, setup order.
    assert s["loader"] is base["loader"]
    assert s["batch_size"] == base["batch_size"]
    assert s["setup_order"] == base["setup_order"] == "loaders_before_model"
    assert base["scheduler"] == "onecycle" and s["scheduler"] == "onecycle_screen"
    assert s["screen"] == dict(base=s["screen"]["base"], steps=6000, seed=42)
    assert s["legacy_script"] is None and s["legacy_gpu_pin"] is None
    for cfg in s["configs"]:
        assert os.path.isfile(os.path.join(REPO, cfg)), cfg
    # .module exists only on the DDP wrapper, i.e. with more than one process.
    assert (s["train_forward"] == "module") == (s["num_processes"] > 1)
    # The base row's validation loop (or none), called the same way as the training forward.
    assert (s["validation"] is None) == (base["validation"] is None)
    assert s["validation"] in (None, s["train_forward"])


def test_screen_scheduler_is_the_row_onecycle_with_a_fixed_length():
    (screen,) = ENTRIES_MOD.SCHEDULERS["onecycle_screen"]
    (onecycle,) = ENTRIES_MOD.SCHEDULERS["onecycle"]
    assert {k: v for k, v in screen.items() if k != "kwargs"} == {k: v for k, v in onecycle.items() if k != "kwargs"}
    assert set(screen["kwargs"]) == set(onecycle["kwargs"])
    assert {k for k in onecycle["kwargs"] if screen["kwargs"][k] != onecycle["kwargs"][k]} == {"total_steps"}
    assert screen["kwargs"]["total_steps"] == "cfg.screen_steps + 100"
    assert screen["kwargs"]["max_lr"] == "cfg.lr"


ALL_256_CFG = "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256.py"
D3_CFG = "configs/OmniScene/omni_gs_160x320_mp3d_cylinder_all_256_single.py"
REF_T1_WEIGHTS = ("/home/qiwei/nips25/workdirs/omni_gs_160x320_mp3d_cylinder_double_all_256/"
                  "checkpoint-48000/model.safetensors")


def _cfg_as_dict(cfg):
    def plain(node):
        if isinstance(node, dict):
            return {k: plain(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return type(node)(plain(v) for v in node)
        return node
    return {k: plain(cfg[k]) for k in cfg.keys()}


def test_d3_config_is_all_256_with_one_frame():
    # The D3 row's config: all_256 (inherited through _base_) with its own run name, REF-T1's
    # weights as init and the pixel branch built for one view; nothing else differs.
    assert SCREEN["screen_mp3d_single_256"]["configs"] == [D3_CFG]
    assert "d3_single_view" in ENTRIES_MOD.TRANSFERS
    mmengine_config = pytest.importorskip("mmengine.config")
    base = _cfg_as_dict(mmengine_config.Config.fromfile(os.path.join(REPO, ALL_256_CFG)))
    d3 = _cfg_as_dict(mmengine_config.Config.fromfile(os.path.join(REPO, D3_CFG)))
    assert "num_frames" not in base["model"]["pixel_gs"]
    assert d3["model"]["pixel_gs"].pop("num_frames") == 1
    assert d3.pop("exp_name") == "omni_gs_160x320_mp3d_cylinder_single_all_256"
    assert base.pop("exp_name") == "omni_gs_160x320_mp3d_cylinder_double_all"
    assert d3.pop("resume_from") == REF_T1_WEIGHTS
    base.pop("resume_from")
    assert d3 == base
    assert d3["model"]["type"] == "OmniGaussianCylinderAll"


# ----------------------------------------------------------------------------- stage-4 rows (the table)

def test_stage4_rows_are_separate_from_table_t_and_screen():
    assert sorted(STAGE4) == STAGE4_ROWS
    assert not set(STAGE4) & set(ENTRIES) and not set(STAGE4) & set(SCREEN)
    assert {r: STAGE4[r]["num_processes"] for r in STAGE4_ROWS} == {"mp3d_double_512_ddp3": 3,
                                                                     "mp3d_double_512_ddp4": 4}


@pytest.mark.parametrize("row", STAGE4_ROWS)
def test_stage4_row_is_the_512_loader_with_the_256_recipe(row):
    s = STAGE4[row]
    loader_row, recipe_row = ENTRIES["mp3d_double_512"], ENTRIES["mp3d_double_256"]
    assert s["stage4"] == dict(loader_row="mp3d_double_512", recipe_row="mp3d_double_256", resolution=[512, 1024])
    assert set(s) == set(recipe_row) | {"stage4"}
    # The 512x1024 loader: the same loader dict (dataset, DataLoader keywords, workers, stage seeds) and
    # batch sizes as mp3d_double_512 (train and val both read batch_size_train).
    assert s["loader"] is loader_row["loader"]
    assert s["loader"]["module"] == "data.mp3d_dataloader_double_512"
    assert s["batch_size"] == loader_row["batch_size"] == dict(train="batch_size_train", val="batch_size_train")
    # The *_256 recipe: OneCycle over len(train loader) x max_epochs + 100, loaders first, .module forward
    # (ddp_forward=true goes through the wrapper), validation through .module.
    for key in ("scheduler", "setup_order", "train_forward", "validation"):
        assert s[key] == recipe_row[key], key
    assert s["scheduler"] == "onecycle"
    (onecycle,) = ENTRIES_MOD.SCHEDULERS["onecycle"]
    assert onecycle["kwargs"]["total_steps"] == "len(train_dataloader) * max_num_epochs + 100"
    assert onecycle["kwargs"]["max_lr"] == "cfg.lr"
    assert s["legacy_script"] is None and s["legacy_gpu_pin"] is None
    assert s["num_processes"] > 1 and (s["train_forward"] == "module") == (s["num_processes"] > 1)
    for cfg in s["configs"]:
        assert os.path.isfile(os.path.join(REPO, cfg)), cfg
    # The table-T rows the stage-4 rows are built from are unchanged.
    assert (loader_row["num_processes"], loader_row["train_forward"], loader_row["scheduler"]) == (1, "plain",
                                                                                                    "warmup_cosine")
    assert recipe_row["loader"]["module"] == "data.mp3d_dataloader_double_256"


def test_stage4_transfer_is_exact():
    # Same architecture (all_256), other image size: no name may be missing or extra.
    assert ENTRIES_MOD.TRANSFERS["stage3_to_stage4_512"] == dict(allowed_missing=[], allowed_extra=[])


def _batch_reads(tree, class_name):
    """{(top, key)} of every batch["top"]["key"] and {(top, None)} of every batch["top"] in the class."""
    (cls,) = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name]

    def const(node):
        node = getattr(node, "value", node) if type(node).__name__ == "Index" else node
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    reads = set()
    for n in ast.walk(cls):
        if not isinstance(n, ast.Subscript):
            continue
        if isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Name) and n.value.value.id == "batch":
            reads.add((const(n.value.slice), const(n.slice)))
        elif isinstance(n.value, ast.Name) and n.value.id == "batch":
            reads.add((const(n.slice), None))
    assert None not in {top for top, _ in reads}, "non-literal batch key"
    return reads


def _loader_provides(rel):
    """{(top, key)} and {(top, None)} of the dict DatasetMP3D.__getitem__ returns."""
    (fn,) = [n for n in ast.walk(_tree(rel)) if isinstance(n, ast.FunctionDef) and n.name == "__getitem__"]
    dicts = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and isinstance(n.value, ast.Dict):
            dicts[n.targets[0].id] = [k.value for k in n.value.keys]
    (ret,) = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    provides = set()
    for k, v in zip(ret.value.keys, ret.value.values):
        provides.add((k.value, None))
        if isinstance(v, ast.Name) and v.id in dicts:
            provides |= {(k.value, key) for key in dicts[v.id]}
    return provides


def test_512_loader_yields_every_batch_key_the_joint_model_reads():
    # The stage-4 rows feed OmniGaussianCylinderAll from the 512x1024 loader. That loader lacks the
    # 256 loader's inputs_pix depth / mono_image / cube_image, which the joint model never reads.
    reads = _batch_reads(_tree("model/omni_gs_cylinder_all.py"), "OmniGaussianCylinderAll")
    keyed = {r for r in reads if r[1] is not None}
    assert ("inputs_pix", "depth_m") in keyed and ("outputs", "mask_gt") in keyed  # the walk found the reads
    provides_512 = _loader_provides("data/mp3d_dataloader_double_512.py")
    provides_256 = _loader_provides("data/mp3d_dataloader_double_256.py")
    assert reads <= provides_512, sorted(reads - provides_512)
    assert reads <= provides_256
    only_256 = provides_256 - provides_512
    assert {("inputs_pix", "depth"), ("inputs_pix", "mono_image"), ("inputs_pix", "cube_image")} <= only_256
    assert not only_256 & reads
    # evaluate.py groups the samples by batch["scene"].
    assert ("scene", None) in provides_512


# ----------------------------------------------------------------------------- loaders (live source)

@pytest.mark.parametrize("row", ROWS)
def test_loader_dataloader_call_is_verbatim(row):
    e = ENTRIES[row]
    tree = _tree(_loader_file(e))
    live = _calls(tree, "DataLoader")
    assert len(live) == 1, "exactly one live DataLoader(...) call per loader"
    assert live[0] in list(ast.walk(_func(tree, e["loader"]["factory"])))
    args, kwargs = _sig(live[0])
    assert len(args) == 1
    assert kwargs == e["loader"]["call"]
    assert ast.literal_eval(kwargs["num_workers"]) == e["loader"]["num_workers"]
    assert ast.literal_eval(kwargs["shuffle"]) is e["loader"]["shuffle"]


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("stage", ["train", "val", "test", "other"])
def test_loader_stage_seed_and_persistence(row, stage):
    e = ENTRIES[row]
    env, _ = _trace_factory(e, "predict" if stage == "other" else stage)
    expected = e["loader"]["stages"][stage]
    assert env["seed"] == expected["seed"]
    assert env["persistent_workers"] is expected["persistent_workers"]


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("stage", ["train", "val"])
def test_loader_dataset_construction(row, stage):
    e = ENTRIES[row]
    env, calls = _trace_factory(e, stage)
    loader_calls = [c for c in calls.values() if ast.unparse(c.func) == "DataLoader"]
    assert len(loader_calls) == 1
    dataset_call = calls[ast.unparse(loader_calls[0].args[0])]
    assert ast.unparse(dataset_call.func) == e["loader"]["dataset_class"]
    assert not dataset_call.args
    assert {k.arg: _eval(k.value, env) for k in dataset_call.keywords} == e["loader"]["dataset_kwargs"][stage]
    classes = [n for n in _tree(_loader_file(e)).body
               if isinstance(n, ast.ClassDef) and n.name == e["loader"]["dataset_class"]]
    assert len(classes) == 1
    assert ("IterableDataset" in [ast.unparse(b) for b in classes[0].bases]) == e["loader"]["iterable"]


# ----------------------------------------------------------------------------- legacy trainers (live source)

@pytest.mark.parametrize("row", ROWS)
def test_legacy_script_imports_the_loader(row):
    e = ENTRIES[row]
    tree = _tree(e["legacy_script"])
    imports = [(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom)
               and (n.module or "").startswith("data.") for a in n.names]
    assert imports == [(e["loader"]["module"], e["loader"]["factory"])]


def _env_assignments(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]).startswith("os.environ["):
            out[ast.literal_eval(n.targets[0].slice)] = ast.literal_eval(n.value)
    return out


@pytest.mark.parametrize("row", ROWS)
def test_legacy_gpu_pin(row):
    env = _env_assignments(_tree(ENTRIES[row]["legacy_script"]))
    assert env.get("CUDA_VISIBLE_DEVICES") == ENTRIES[row]["legacy_gpu_pin"]


def test_train_never_pins_gpus():
    tree = _tree("train.py")
    assert _env_assignments(tree) == {}
    assert not any(isinstance(n, ast.Constant) and n.value == "CUDA_VISIBLE_DEVICES" for n in ast.walk(tree))


@pytest.mark.parametrize("row", ROWS)
def test_legacy_loader_batch_sizes(row):
    e = ENTRIES[row]
    main = _func(_tree(e["legacy_script"]), "main")
    got = {}
    for call in _calls(main, e["loader"]["factory"]):
        args, kwargs = _sig(call)
        got[ast.literal_eval(kwargs["stage"])] = args
    assert got == {stage: [f"dataset_config.{key}"] for stage, key in e["batch_size"].items()}


@pytest.mark.parametrize("row", ROWS)
def test_legacy_scheduler_assignments(row):
    e = ENTRIES[row]
    main = _func(_tree(e["legacy_script"]), "main")
    assert _scheduler_assigns(main) == ENTRIES_MOD.SCHEDULERS[e["scheduler"]]


@pytest.mark.parametrize("row", ROWS)
def test_legacy_forward_mode(row):
    e = ENTRIES[row]
    forwards = _forward_calls(_func(_tree(e["legacy_script"]), "main"))
    assert len(forwards) == 1
    mode = {"my_model.module.forward": "module", "my_model.forward": "plain"}[ast.unparse(forwards[0].func)]
    assert mode == e["train_forward"]
    args, kwargs = ENTRIES_MOD.TRAIN_FORWARD_ARGS
    assert _sig(forwards[0]) == (list(args), dict(kwargs))


@pytest.mark.parametrize("row", ROWS)
def test_legacy_validation_mode(row):
    e = ENTRIES[row]
    main = _func(_tree(e["legacy_script"]), "main")
    calls = _validation_calls(main)
    uses_val_freq = any(ast.unparse(n) == "cfg.val_freq" for n in ast.walk(main))
    if e["validation"] is None:
        assert calls == [] and not uses_val_freq
        return
    assert len(calls) == 1 and uses_val_freq
    mode = {"my_model.module.validation_step": "module", "my_model.validation_step": "plain"}[ast.unparse(calls[0].func)]
    assert mode == e["validation"]
    args, kwargs = ENTRIES_MOD.VALIDATION_STEP_ARGS
    assert _sig(calls[0]) == (list(args), dict(kwargs))


def _setup_positions(main, factory, scheduler_funcs=("torch.optim.lr_scheduler.",)):
    def first(pred):
        lines = [n.lineno for n in ast.walk(main) if pred(n)]
        assert lines, "statement not found"
        return min(lines)

    def call_to(name):
        return lambda n: isinstance(n, ast.Call) and ast.unparse(n.func) == name

    return {
        "set_seed": first(call_to("set_seed")),
        "init_trackers": first(call_to("accelerator.init_trackers")),
        "loaders": first(call_to(factory)),
        "model": first(call_to("model_builder.build")),
        "scheduler": first(lambda n: isinstance(n, ast.Call) and ast.unparse(n.func).startswith(scheduler_funcs)),
        "resume": first(lambda n: isinstance(n, ast.Assign) and ast.unparse(n) == "path = cfg.resume_from"),
        "prepare": first(call_to("accelerator.prepare")),
    }


@pytest.mark.parametrize("row", ROWS)
def test_legacy_setup_order_and_seeding(row):
    e = ENTRIES[row]
    tree = _tree(e["legacy_script"])
    main = _func(tree, "main")
    positions = _setup_positions(main, e["loader"]["factory"])
    assert sorted(positions, key=positions.get) == EXPECTED_ORDER[e["setup_order"]]
    (seed_call,) = _calls(main, "set_seed")
    assert _sig(seed_call) == (["cfg.seed + accelerator.local_process_index"], {})
    block = _main_block(tree)
    assert "SEED = 42" in [ast.unparse(n) for n in block.body]
    assert len(_calls(block, "torch.manual_seed")) == 1
    assert _sig(_calls(block, "torch.manual_seed")[0]) == (["SEED"], {})


LEGACY_ACCELERATOR_KWARGS = {
    "gradient_accumulation_steps": "cfg.gradient_accumulation_steps",
    "mixed_precision": "cfg.mixed_precision",
    "log_with": "cfg.report_to",
    "project_config": "accelerator_project_config",
    "kwargs_handlers": "[kwargs]",
}


@pytest.mark.parametrize("row", ROWS)
def test_accelerator_arguments_match_legacy(row):
    legacy = _func(_tree(ENTRIES[row]["legacy_script"]), "main")
    new = _func(_tree("train.py"), "main")
    for fn in (legacy, new):
        (acc,) = _calls(fn, "Accelerator")
        args, kwargs = _sig(acc)
        assert args == []
        expected = dict(LEGACY_ACCELERATOR_KWARGS)
        if fn is new:
            # train.py passes the same [kwargs] list, extended only by the ddp_forward switch.
            expected["kwargs_handlers"] = "kwargs_handlers"
        assert kwargs == expected
        (pg,) = _calls(fn, "InitProcessGroupKwargs")
        assert _sig(pg) == ([], {"timeout": "timedelta(seconds=1800)"})
        (pc,) = _calls(fn, "ProjectConfiguration")
        assert _sig(pc) == ([], {"project_dir": "cfg.work_dir", "logging_dir": "os.path.join(cfg.work_dir, 'logs')"})
    handlers = [ast.unparse(n) for n in ast.walk(new) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "kwargs_handlers"]
    assert handlers == ["kwargs_handlers = [kwargs]"]


# ----------------------------------------------------------------------------- train.py (source)

def test_train_dataloader_call_keywords():
    fn = _func(_tree("train.py"), "build_dataloader")
    (call,) = _calls(fn, "DataLoader")
    args, kwargs = _sig(call)
    assert args == ["dataset"]
    for row in ROWS:
        assert set(kwargs) == set(ENTRIES[row]["loader"]["call"]) | {"sampler"}
    assert kwargs["shuffle"] == "False"
    assert kwargs["worker_init_fn"] == "loader_module.worker_init_fn"
    assert kwargs["generator"] == "loader_module.get_generator(stage_spec['seed'])"
    assert kwargs["num_workers"] == "spec['num_workers']"


@pytest.mark.parametrize("kind", sorted(TRAIN_SCHEDULER_FUNCS))
def test_train_scheduler_calls_are_verbatim(kind):
    fn = _func(_tree("train.py"), TRAIN_SCHEDULER_FUNCS[kind])
    assert _scheduler_assigns(fn) == ENTRIES_MOD.SCHEDULERS[kind]


def test_train_forward_and_validation_calls():
    main = _func(_tree("train.py"), "main")
    forwards = _forward_calls(main)
    assert sorted(ast.unparse(c.func) for c in forwards) == ["my_model", "my_model.forward", "my_model.module.forward"]
    args, kwargs = ENTRIES_MOD.TRAIN_FORWARD_ARGS
    for call in forwards:
        assert _sig(call) == (list(args), dict(kwargs))
    calls = _validation_calls(main)
    assert sorted(ast.unparse(c.func) for c in calls) == ["my_model.module.validation_step", "my_model.validation_step"]
    args, kwargs = ENTRIES_MOD.VALIDATION_STEP_ARGS
    for call in calls:
        assert _sig(call) == (list(args), dict(kwargs))
    # Validation runs under torch.no_grad().
    no_grad = [w for w in ast.walk(main) if isinstance(w, ast.With)
               and [ast.unparse(i.context_expr) for i in w.items] == ["torch.no_grad()"]]
    assert len(no_grad) == 1
    inside = list(ast.walk(no_grad[0]))
    assert all(c in inside for c in calls)
    assert not any(c in inside for c in forwards)


def test_train_main_block_seeds_like_legacy():
    block = _main_block(_tree("train.py"))
    stmts = [ast.unparse(n) for n in block.body]
    assert "SEED = 42" in stmts
    (seed_call,) = _calls(block, "torch.manual_seed")
    assert _sig(seed_call) == (["SEED"], {})
    # The loader module (and data/dataloader.py) are imported before the seed, as at the legacy module top.
    seed_line = seed_call.lineno
    imports = [n.lineno for n in block.body if isinstance(n, ast.Import)] + \
              [c.lineno for c in _calls(block, "importlib.import_module")]
    assert imports and max(imports) < seed_line


def _run_main_block_seeding(seed):
    """Execute the __main__ block from `SEED = 42` to torch.manual_seed with args.seed = seed."""
    body = _main_block(_tree("train.py")).body
    start = [i for i, n in enumerate(body) if ast.unparse(n) == "SEED = 42"]
    end = [i for i, n in enumerate(body) if _calls(n, "torch.manual_seed")]
    assert len(start) == 1 and len(end) == 1 and start[0] < end[0]
    stmts = body[start[0]:end[0] + 1]
    # Nothing else runs between the legacy constant and the seed call.
    assert [ast.unparse(n) for n in stmts] == [
        "SEED = 42", "if args.seed is not None:\n    SEED = args.seed", "torch.manual_seed(SEED)"]
    seeded = []
    env = {"args": types.SimpleNamespace(seed=seed), "torch": types.SimpleNamespace(manual_seed=seeded.append)}
    exec(compile(ast.Module(body=stmts, type_ignores=[]), "train.py:__main__", "exec"), env)
    return seeded


def test_train_main_block_seed_override():
    # Table-T rows: parse_args leaves args.seed None, so the legacy constant is used.
    assert _run_main_block_seeding(None) == [42]
    # Screen rows: parse_args always sets args.seed (--seed, or the row's default).
    assert _run_main_block_seeding(43) == [43]
    assert _run_main_block_seeding(0) == [0]


# ----------------------------------------------------------------------------- runtime (CPU)

def _dummy_dataset_cls(iterable, records):
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


def _assert_same_loader(a, b):
    torch = pytest.importorskip("torch")
    for attr in ("batch_size", "num_workers", "persistent_workers", "drop_last", "pin_memory", "pin_memory_device",
                 "timeout", "prefetch_factor", "multiprocessing_context", "_dataset_kind"):
        assert getattr(a, attr) == getattr(b, attr), attr
    assert a.worker_init_fn is b.worker_init_fn
    assert a.collate_fn is b.collate_fn
    assert type(a.sampler) is type(b.sampler)
    assert type(a.batch_sampler) is type(b.batch_sampler)
    if a.batch_sampler is not None:
        assert (a.batch_sampler.batch_size, a.batch_sampler.drop_last) == (b.batch_sampler.batch_size, b.batch_sampler.drop_last)
    assert a.generator is not b.generator
    assert a.generator.initial_seed() == b.generator.initial_seed()
    assert torch.equal(a.generator.get_state(), b.generator.get_state())
    assert type(a.dataset) is type(b.dataset) and a.dataset.kwargs == b.dataset.kwargs


def _train_module():
    pytest.importorskip("torch")
    import train
    return train


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("stage", ["train", "val"])
def test_build_dataloader_equals_legacy_factory(row, stage, monkeypatch):
    torch = pytest.importorskip("torch")
    train = _train_module()
    e = ENTRIES[row]
    module = importlib.import_module(e["loader"]["module"])
    records = []
    monkeypatch.setattr(module, e["loader"]["dataset_class"], _dummy_dataset_cls(e["loader"]["iterable"], records))
    legacy = getattr(module, e["loader"]["factory"])(3, stage=stage)
    new = train.build_dataloader(e, module, stage, 3)
    assert records[0] == records[1] == e["loader"]["dataset_kwargs"][stage]
    _assert_same_loader(legacy, new)
    assert new.num_workers == e["loader"]["num_workers"]
    assert new.generator.initial_seed() == e["loader"]["stages"][stage]["seed"]
    if not e["loader"]["iterable"]:
        assert isinstance(new.sampler, torch.utils.data.SequentialSampler)


@pytest.mark.parametrize("row", [r for r in ROWS if not ENTRIES[r]["loader"]["iterable"]])
def test_shuffle_train_sampler(row, monkeypatch):
    torch = pytest.importorskip("torch")
    train = _train_module()
    e = ENTRIES[row]
    module = importlib.import_module(e["loader"]["module"])
    monkeypatch.setattr(module, e["loader"]["dataset_class"], _dummy_dataset_cls(False, []))
    seed = e["loader"]["stages"]["train"]["seed"]
    shuffled = train.build_dataloader(e, module, "train", 2, shuffle_train=True)
    assert isinstance(shuffled.sampler, torch.utils.data.RandomSampler)
    assert shuffled.sampler.generator.initial_seed() == seed
    assert shuffled.sampler.generator is not shuffled.generator
    assert shuffled.generator.initial_seed() == seed
    # The switch only touches the train split.
    val = train.build_dataloader(e, module, "val", 2, shuffle_train=True)
    assert isinstance(val.sampler, torch.utils.data.SequentialSampler)


def test_shuffle_train_refused_for_iterable_row():
    train = _train_module()
    for row in ROWS:
        values = {"shuffle_train": True}
        if ENTRIES[row]["loader"]["iterable"]:
            with pytest.raises(SystemExit, match="IterableDataset"):
                train.check_row_switches(row, ENTRIES[row], values)
        else:
            train.check_row_switches(row, ENTRIES[row], values)


def _two_group_optimizer(torch, lr):
    # The shape of configure_optimizers(): base params at lr, backbone at lr * 0.1.
    base, backbone = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    return torch.optim.AdamW([{"params": base.parameters()}, {"params": backbone.parameters(), "lr": lr * 0.1}],
                             lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)


def _lr_trace(torch, scheduler, optimizer, steps):
    trace = [[g["lr"] for g in optimizer.param_groups]]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        trace.append([g["lr"] for g in optimizer.param_groups])
    return trace


@pytest.mark.parametrize("lr,epochs,n_batches", [(2e-4, 25, 60), (1e-4, 20, 46), (2e-4, 15, 1000)])
def test_onecycle_lr_equals_legacy(lr, epochs, n_batches):
    torch = pytest.importorskip("torch")
    train = _train_module()
    from tests.legacy_ref import train_ref as legacy
    cfg = types.SimpleNamespace(lr=lr)
    loader = list(range(n_batches))
    o1, o2 = _two_group_optimizer(torch, lr), _two_group_optimizer(torch, lr)
    s1 = legacy.onecycle_scheduler(o1, cfg, loader, epochs)
    s2 = train.build_onecycle_scheduler(o2, cfg, loader, epochs)
    assert type(s1) is type(s2) and s1.total_steps == s2.total_steps == n_batches * epochs + 100
    t1, t2 = _lr_trace(torch, s1, o1, 1000), _lr_trace(torch, s2, o2, 1000)
    assert t1 == t2  # exact float equality at every step, incl. 0 / 1 / 100 / 1000
    for step in (0, 1, 100, 1000):
        assert t1[step] == t2[step]


@pytest.mark.parametrize("lr,warmup,max_steps,procs", [(2e-4, 1000, 5000, 1), (1e-4, 500, 5000, 1), (2e-4, 1000, 5000, 3)])
def test_warmup_cosine_lr_equals_legacy(lr, warmup, max_steps, procs):
    torch = pytest.importorskip("torch")
    train = _train_module()
    from tests.legacy_ref import train_ref as legacy
    cfg = types.SimpleNamespace(lr=lr, warmup_steps=warmup, max_train_steps=max_steps)
    accelerator = types.SimpleNamespace(num_processes=procs)
    o1, o2 = _two_group_optimizer(torch, lr), _two_group_optimizer(torch, lr)
    s1 = legacy.warmup_cosine_scheduler(o1, cfg, accelerator)
    s2 = train.build_warmup_cosine_scheduler(o2, cfg, accelerator)
    assert type(s1) is type(s2)
    assert _lr_trace(torch, s1, o1, 1200) == _lr_trace(torch, s2, o2, 1200)


@pytest.mark.parametrize("lr,steps", [(2e-4, 6000), (1e-4, 6000), (2e-4, 37)])
def test_onecycle_screen_lr_equals_legacy_onecycle_of_one_epoch(lr, steps):
    # The screen OneCycle is the legacy one with len(train_dataloader) * max_num_epochs = screen steps.
    torch = pytest.importorskip("torch")
    train = _train_module()
    from tests.legacy_ref import train_ref as legacy
    o1, o2 = _two_group_optimizer(torch, lr), _two_group_optimizer(torch, lr)
    s1 = legacy.onecycle_scheduler(o1, types.SimpleNamespace(lr=lr), list(range(steps)), 1)
    s2 = train.build_onecycle_screen_scheduler(o2, types.SimpleNamespace(lr=lr, screen_steps=steps))
    assert type(s1) is type(s2) and s1.total_steps == s2.total_steps == steps + 100
    assert [g["max_lr"] for g in o2.param_groups] == [lr, lr]  # max_lr = cfg.lr for both groups
    t1, t2 = _lr_trace(torch, s1, o1, steps), _lr_trace(torch, s2, o2, steps)
    assert t1 == t2  # exact, at every step the screen runs
    assert t2[0] == pytest.approx([lr / 25.0] * 2)
    if steps == 6000:
        # Warm-up ends at step 0.01 * 6100 - 1 = 60; then cosine towards lr / 25 / 1e4 at step 6099.
        assert t2[60] == pytest.approx([lr, lr])
        assert t2[59][0] < t2[60][0] and t2[61][0] < t2[60][0]
        assert t2[-1][0] < lr * 1e-3


# ----------------------------------------------------------------------------- stubbed train.main

FAKE_CONFIG = """
exp_name = "fake"
output_dir = "/data/qiwei/nips25/workdirs"
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
resume_from = ''
report_to = "tensorboard"
seed = 0
dataset_params = dict(batch_size_train=2, batch_size_val=1, num_workers=32)
model = dict(type='FakeModel')
"""
N_TRAIN, N_VAL = 5, 2


def write_fake_config(directory, text=FAKE_CONFIG):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "fake_cfg.py")
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))
    return path


def _fakes(torch, events, num_processes, process_index=0):
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
            opt = torch.optim.AdamW([{"params": self.head.parameters()},
                                     {"params": self.backbone.parameters(), "lr": lr * 0.1}], lr=lr)
            events.append(("configure_optimizers", lr, opt))
            return [opt]

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
            self.process_index = process_index
            self.local_process_index = process_index
            self.is_main_process = process_index == 0
            self.device = torch.device("cpu")
            self.sync_gradients = True

        def init_trackers(self, project_name, init_kwargs=None):
            events.append(("init_trackers", project_name, init_kwargs))

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
            open(os.path.join(output_dir, "model.safetensors"), "w").close()

        def log(self, values, step=None):
            pass

        def print(self, *args):
            pass

        def end_training(self):
            events.append(("end_training",))

    return FakeModel, FakeAccelerator


def run_fake(monkeypatch, tmp_path, row, extra_args=(), num_processes=None, work_dir=None, config_text=FAKE_CONFIG,
             process_index=0):
    """Run train.main for `row` with Accelerate, the model builder and the dataset stubbed.

    Returns (events, work_dir, model). Loaders are real DataLoaders built by
    train.build_dataloader (never iterated: the fake prepare() hands back a few tensors).
    process_index != 0 runs it as a non-main rank (see run_fake_ranks).
    """
    torch = pytest.importorskip("torch")
    train = _train_module()
    entry = ALL_ENTRIES[row]
    events = []
    FakeModel, FakeAccelerator = _fakes(torch, events, num_processes or entry["num_processes"], process_index)
    model = FakeModel()

    def build_model(cfg, accelerator):
        events.append(("build_model",))
        return model

    loader_stub = types.SimpleNamespace(
        get_generator=lambda seed: torch.Generator().manual_seed(seed),
        worker_init_fn=lambda worker_id: None,
    )
    setattr(loader_stub, entry["loader"]["dataset_class"], _dummy_dataset_cls(entry["loader"]["iterable"], []))

    orig_build_dataloader = train.build_dataloader

    def build_dataloader(e, module, stage, batch_size, shuffle_train=False, dataset_extra=None):
        dl = orig_build_dataloader(e, module, stage, batch_size, shuffle_train, dataset_extra)
        events.append(("loader", stage, batch_size, dl))
        return dl

    def wrap_scheduler(name):
        orig = getattr(train, name)

        def builder(*args, **kwargs):
            sched = orig(*args, **kwargs)
            events.append(("scheduler", name, sched))
            return sched

        monkeypatch.setattr(train, name, builder)

    monkeypatch.setattr(train, "Accelerator", FakeAccelerator)
    monkeypatch.setattr(train, "set_seed", lambda seed: events.append(("set_seed", seed)))
    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.setattr(train, "apply_render_switches", lambda values: events.append(("render", values["prune_opacity"])))
    monkeypatch.setattr(train, "build_dataloader", build_dataloader)
    for name in TRAIN_SCHEDULER_FUNCS.values():
        wrap_scheduler(name)
    monkeypatch.chdir(tmp_path)  # train.main moves into <work dir>/cwd; restored at teardown

    work = str(work_dir or tmp_path / "runs" / f"fake_{row}")
    cfg_path = write_fake_config(str(tmp_path / "cfg"), config_text)
    args = train.parse_args(["--entry", row, "--py-config", cfg_path, "--work-dir", work, *extra_args])
    try:
        train.main(args, entry, loader_stub)
    finally:
        events.append(("returned",))
    return events, work, model


def _names(events):
    return [ev[0] for ev in events]


def _setup_sequence(events):
    seq = []
    for ev in events:
        name = ev[0]
        if name == "loader" and ev[1] == "train":
            seq.append("loaders")
        elif name == "build_model":
            seq.append("model")
        elif name in ("set_seed", "init_trackers", "scheduler", "prepare"):
            seq.append(name)
    return seq


@pytest.mark.parametrize("row", ROWS + STAGE4_ROWS)
def test_fake_run_reproduces_the_row(row, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    e = ALL_ENTRIES[row]
    events, work, model = run_fake(monkeypatch, tmp_path, row)
    names = _names(events)

    # Setup order (resume is covered in test_fake_run_resumes_before_prepare).
    expected = [s for s in EXPECTED_ORDER[e["setup_order"]] if s != "resume"]
    assert _setup_sequence(events) == expected
    assert ("set_seed", 0) in events

    # Accelerator arguments: the legacy ones, a single InitProcessGroupKwargs(1800 s).
    (acc,) = [ev[1] for ev in events if ev[0] == "Accelerator"]
    assert acc["gradient_accumulation_steps"] == 1 and acc["mixed_precision"] == "no"
    assert acc["log_with"] == "tensorboard"
    assert acc["project_config"].project_dir == work
    assert acc["project_config"].logging_dir == os.path.join(work, "logs")
    assert len(acc["kwargs_handlers"]) == 1
    assert acc["kwargs_handlers"][0].timeout.total_seconds() == 1800

    # Loaders: batch sizes from the row, table-T keywords, prepared together with model/optimizer/scheduler.
    loaders = {ev[1]: ev for ev in events if ev[0] == "loader"}
    assert loaders["train"][2] == 2
    assert loaders["val"][2] == (1 if e["batch_size"]["val"] == "batch_size_val" else 2)
    assert loaders["train"][3].num_workers == e["loader"]["num_workers"]
    (prep,) = [ev for ev in events if ev[0] == "prepare"]
    (opt_ev,) = [ev for ev in events if ev[0] == "configure_optimizers"]
    (sched_ev,) = [ev for ev in events if ev[0] == "scheduler"]
    assert opt_ev[1] == 2e-4
    assert prep[1] is model and prep[2] is opt_ev[2] and prep[5] is sched_ev[2]
    assert prep[3] is loaders["train"][3] and prep[4] is loaders["val"][3]
    assert sched_ev[1] == TRAIN_SCHEDULER_FUNCS[e["scheduler"]]
    assert ("render", 0.0) in events

    # Forward mode: .module.forward / .forward never go through __call__ or the wrapper.
    forwards = [ev for ev in events if ev[0] == "forward"]
    assert len(forwards) == N_TRAIN * 2
    assert [ev[2] for ev in forwards] == list(range(N_TRAIN * 2))
    assert all(ev[1] == "train" and ev[3] == 100 and ev[5] for ev in forwards)
    assert all(os.path.realpath(ev[4]) == os.path.realpath(os.path.join(work, "cwd")) for ev in forwards)
    assert "ddp_forward" not in names and "call_hook" not in names

    # Validation every val_freq under no_grad in eval mode, output under <work>/validation.
    vals = [ev for ev in events if ev[0] == "validation_step"]
    if e["validation"] is None:
        assert vals == []
    else:
        steps = [i for i in range(N_TRAIN * 2) if i > 0 and i % 2 == 0]
        assert [ev[1] for ev in vals] == [os.path.join(work, "validation", f"step-{s}/batch-{b}")
                                          for s in steps for b in range(N_VAL)]
        assert all(ev[2] is False and ev[3] is False for ev in vals)
    assert model.training

    # Saves every save_freq under the work dir, `latest` points at the last one.
    saves = [ev[1] for ev in events if ev[0] == "save_state"]
    assert saves == [os.path.join(work, f"checkpoint-{s}") for s in (2, 4, 6, 8)]
    assert os.path.realpath(os.path.join(work, "latest")) == os.path.realpath(saves[-1])
    with open(os.path.join(work, "switches.json")) as f:
        assert json.load(f)["ddp_forward"] is False
    assert os.path.isfile(os.path.join(work, "fake_cfg.py"))
    assert names[-2:] == ["end_training", "returned"]


@pytest.mark.parametrize("row", ROWS + SCREEN_ROWS + STAGE4_ROWS)
def test_fake_run_refuses_other_process_counts(row, monkeypatch, tmp_path):
    wrong = 1 if ALL_ENTRIES[row]["num_processes"] != 1 else 3
    work = tmp_path / "runs" / "wrong_count"
    with pytest.raises(SystemExit, match=f"{ALL_ENTRIES[row]['num_processes']} process"):
        run_fake(monkeypatch, tmp_path, row, num_processes=wrong, work_dir=work)
    assert not work.exists()  # refused before anything was written


def test_fake_run_ddp_forward_goes_through_the_wrapper(monkeypatch, tmp_path):
    events, work, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_256", ["--switch", "ddp_forward=true"])
    names = _names(events)
    forwards = [i for i, n in enumerate(names) if n == "forward"]
    assert len(forwards) == N_TRAIN * 2
    # Each training forward is preceded by the wrapper's forward and the module's __call__ hooks.
    assert all(names[i - 2:i] == ["ddp_forward", "call_hook"] for i in forwards)
    (acc,) = [ev[1] for ev in events if ev[0] == "Accelerator"]
    assert [type(h).__name__ for h in acc["kwargs_handlers"]] == ["InitProcessGroupKwargs", "DistributedDataParallelKwargs"]
    assert acc["kwargs_handlers"][1].find_unused_parameters is True
    # Validation still calls the unwrapped module.
    assert len([n for n in names if n == "validation_step"]) == 8
    with open(os.path.join(work, "switches.json")) as f:
        assert json.load(f)["ddp_forward"] is True


def test_fake_run_ddp_forward_single_process_row(monkeypatch, tmp_path):
    events, _, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_160", ["--switch", "ddp_forward=true"])
    names = _names(events)
    assert "ddp_forward" not in names  # nothing to wrap with one process
    assert names.count("call_hook") == N_TRAIN * 2


def test_fake_run_shuffle_train(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    events, _, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_160", ["--switch", "shuffle_train=true"])
    (prep,) = [ev for ev in events if ev[0] == "prepare"]
    assert isinstance(prep[3].sampler, torch.utils.data.RandomSampler)
    assert prep[3].sampler.generator.initial_seed() == 1234
    assert isinstance(prep[4].sampler, torch.utils.data.SequentialSampler)


def test_fake_run_shuffle_train_refused_for_360loc(monkeypatch, tmp_path):
    work = tmp_path / "runs" / "loc_shuffle"
    with pytest.raises(SystemExit, match="IterableDataset"):
        run_fake(monkeypatch, tmp_path, "loc360_all_256", ["--switch", "shuffle_train=true"], work_dir=work)
    assert not work.exists()


def test_fake_run_loc360_interleave(monkeypatch, tmp_path):
    # D10: only the train dataset gets interleave=True; the val dataset keeps the row's keywords.
    events, work, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256", ["--switch", "loc360_interleave=true"])
    loaders = {ev[1]: ev for ev in events if ev[0] == "loader"}
    assert loaders["train"][3].dataset.kwargs == {"stage": "train", "interleave": True}
    assert loaders["val"][3].dataset.kwargs == {"stage": "val"}
    with open(os.path.join(work, "switches.json")) as f:
        assert json.load(f)["loc360_interleave"] is True


def test_fake_run_loc360_interleave_default_keeps_the_row_keywords(monkeypatch, tmp_path):
    events, _, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256")
    loaders = {ev[1]: ev for ev in events if ev[0] == "loader"}
    assert loaders["train"][3].dataset.kwargs == {"stage": "train"}


@pytest.mark.parametrize("row", ["mp3d_double_256", "screen_mp3d_all_256"])
def test_fake_run_loc360_interleave_refused_for_other_loaders(row, monkeypatch, tmp_path):
    work = tmp_path / "runs" / "interleave_refused"
    with pytest.raises(SystemExit, match="loc360_interleave needs the 360Loc loader"):
        run_fake(monkeypatch, tmp_path, row, ["--switch", "loc360_interleave=true"], work_dir=work)
    assert not work.exists()


def test_fake_run_save_final(monkeypatch, tmp_path):
    # --save-final: the periodic saves, then the final weights as checkpoint-<steps> before end_training.
    events, work, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256", ["--save-final"])
    names = _names(events)
    saves = [ev[1] for ev in events if ev[0] == "save_state"]
    assert saves == [os.path.join(work, f"checkpoint-{s}") for s in (2, 4, 6, 8, N_TRAIN * 2)]
    assert names.index("end_training") > max(i for i, n in enumerate(names) if n == "save_state")
    assert os.path.realpath(os.path.join(work, "latest")) == os.path.realpath(saves[-1])


def test_fake_run_save_final_non_main_rank_does_not_save(monkeypatch, tmp_path):
    events, _, _ = run_fake(monkeypatch, tmp_path, "loc360_all_256", ["--save-final"], process_index=1)
    assert [ev for ev in events if ev[0] == "save_state"] == []


def test_fake_run_max_steps_and_profile(monkeypatch, tmp_path):
    events, work, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_512", ["--max-steps", "3", "--profile-steps", "2"])
    assert _names(events).count("forward") == 3
    with open(os.path.join(work, "profile_steps_rank0.json")) as f:
        profile = json.load(f)
    assert profile["steps"] == 2 and len(profile["step_ms"]) == 2
    assert profile["median_ms"] >= 0


def test_fake_run_resumes_before_prepare(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    source = {"backbone.weight": torch.full((1, 2), 3.0), "backbone.bias": torch.full((1,), 4.0),
              "head.weight": torch.full((1, 2), 5.0), "head.bias": torch.full((1,), 6.0)}
    ckpt = tmp_path / "src" / "checkpoint-3000"
    ckpt.mkdir(parents=True)
    save_file(source, str(ckpt / "model.safetensors"))
    events, _, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_256", ["--resume-from", str(ckpt)])
    (prep,) = [ev for ev in events if ev[0] == "prepare"]
    for k, v in source.items():
        assert torch.equal(prep[6][k], v)

    # An extra name is refused under the default 'exact' transfer ...
    from tools.resume import ResumeError
    extra = dict(source, **{"pixel_gs.mono_depth.w": torch.zeros(1)})
    save_file(extra, str(ckpt / "model.safetensors"))
    with pytest.raises(ResumeError, match="pixel_gs.mono_depth.w"):
        run_fake(monkeypatch, tmp_path, "mp3d_double_256", ["--resume-from", str(ckpt)],
                 work_dir=tmp_path / "runs" / "exact")
    # ... and accepted when the transfer allows it.
    events, _, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_256",
                            ["--resume-from", str(ckpt), "--transfer", "stage1_to_stage2"],
                            work_dir=tmp_path / "runs" / "stage12")
    assert "prepare" in _names(events)


def test_fake_run_no_resume_ignores_config(monkeypatch, tmp_path):
    text = FAKE_CONFIG.replace("resume_from = ''", "resume_from = '/data/qiwei/nips25/workdirs/x/model.safetensors'")
    events, _, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_160", ["--no-resume"], config_text=text)
    assert "prepare" in _names(events)


# ----------------------------------------------------------------------------- Phase-2 screen rows (runtime)

def test_train_offers_table_t_screen_and_stage4_rows():
    train = _train_module()
    assert sorted(train.ENTRIES) == sorted(ROWS + SCREEN_ROWS + STAGE4_ROWS)
    assert all(train.ENTRIES[r] == ALL_ENTRIES[r] for r in ROWS + SCREEN_ROWS + STAGE4_ROWS)
    assert train.STAGE4_ENTRIES == STAGE4


@pytest.mark.parametrize("row", STAGE4_ROWS)
@pytest.mark.parametrize("flag", [("--screen-steps", "100"), ("--seed", "43")])
def test_screen_flags_refused_on_stage4_rows(row, flag, tmp_path, capsys):
    train = _train_module()
    args = _parse(train, row, tmp_path)
    assert args.screen_steps is None and args.seed is None
    with pytest.raises(SystemExit):
        _parse(train, row, tmp_path, *flag)
    err = capsys.readouterr().err
    assert f"{flag[0]} is only valid for the screen rows" in err
    assert "runs stage 4 (the mp3d_double_256 recipe on the mp3d_double_512 loader)" in err


def test_fake_run_stage4_row_ddp_forward_and_onecycle(monkeypatch, tmp_path):
    # The stage-4 launch: wrapped DDP forward (D1), OneCycle over len(train loader) x max_epochs + 100,
    # validation through .module, saves every save_freq; the 512 loader's batch sizes.
    events, work, _ = run_fake(monkeypatch, tmp_path, "mp3d_double_512_ddp3", ["--switch", "ddp_forward=true"])
    names = _names(events)
    forwards = [i for i, n in enumerate(names) if n == "forward"]
    assert len(forwards) == N_TRAIN * 2
    assert all(names[i - 2:i] == ["ddp_forward", "call_hook"] for i in forwards)
    (sched_ev,) = [ev for ev in events if ev[0] == "scheduler"]
    loaders = {ev[1]: ev for ev in events if ev[0] == "loader"}
    assert sched_ev[1] == "build_onecycle_scheduler"
    assert sched_ev[2].total_steps == len(loaders["train"][3]) * 2 + 100
    assert [g["max_lr"] for g in sched_ev[2].optimizer.param_groups] == [2e-4, 2e-4]
    assert loaders["train"][2] == loaders["val"][2] == 2  # batch_size_train for both
    assert names.count("validation_step") == 4 * N_VAL
    assert [ev[1] for ev in events if ev[0] == "save_state"] == [os.path.join(work, f"checkpoint-{s}") for s in (2, 4, 6, 8)]


def _parse(train, row, tmp_path, *extra):
    return train.parse_args(["--entry", row, "--py-config", "c.py", "--work-dir", str(tmp_path / "w"), *extra])


def test_parse_args_screen_defaults_and_overrides(tmp_path):
    train = _train_module()
    for row in SCREEN_ROWS:
        args = _parse(train, row, tmp_path)
        assert (args.screen_steps, args.seed) == (6000, 42)
    args = _parse(train, "screen_loc360_all_256", tmp_path, "--screen-steps", "500", "--seed", "43")
    assert (args.screen_steps, args.seed) == (500, 43)
    for row in ROWS:
        args = _parse(train, row, tmp_path)
        assert args.screen_steps is None and args.seed is None


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("flag", [("--screen-steps", "100"), ("--seed", "43")])
def test_screen_flags_refused_on_table_t_rows(row, flag, tmp_path, capsys):
    train = _train_module()
    with pytest.raises(SystemExit):
        _parse(train, row, tmp_path, *flag)
    err = capsys.readouterr().err
    assert f"{flag[0]} is only valid for the screen rows" in err
    assert ENTRIES[row]["legacy_script"] in err


@pytest.mark.parametrize("flag", [("--screen-steps", "0"), ("--screen-steps", "-5"), ("--seed", "-1"),
                                  ("--seed", str(2 ** 31))])
def test_screen_flag_ranges(flag, tmp_path):
    train = _train_module()
    with pytest.raises(SystemExit):
        _parse(train, "screen_mp3d_all_256", tmp_path, *flag)


def _screen_scheduler(events):
    (sched_ev,) = [ev for ev in events if ev[0] == "scheduler"]
    assert sched_ev[1] == "build_onecycle_screen_scheduler"
    return sched_ev[2]


def _dumped_config(work):
    from mmengine.config import Config
    return Config.fromfile(os.path.join(work, "fake_cfg.py"))


def test_fake_run_screen_mp3d_row(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    # 12 steps with 5 batches per epoch: three epochs although the config says max_epochs = 2.
    events, work, model = run_fake(monkeypatch, tmp_path, "screen_mp3d_all_256", ["--screen-steps", "12"])
    names = _names(events)
    assert _setup_sequence(events) == [s for s in EXPECTED_ORDER["loaders_before_model"] if s != "resume"]
    # Default seed 42 reaches set_seed(cfg.seed + local_process_index); the config's seed = 0 does not.
    assert [ev for ev in events if ev[0] == "set_seed"] == [("set_seed", 42)]

    sched = _screen_scheduler(events)
    assert sched.total_steps == 12 + 100
    assert [g["max_lr"] for g in sched.optimizer.param_groups] == [2e-4, 2e-4]  # cfg.lr
    assert sched.last_epoch == 12  # stepped once per train step, never near total_steps

    # One process: the model is not wrapped, the plain forward and validation_step are called.
    forwards = [ev for ev in events if ev[0] == "forward"]
    assert [ev[2] for ev in forwards] == list(range(12))
    assert "ddp_forward" not in names and "call_hook" not in names
    vals = [ev[1] for ev in events if ev[0] == "validation_step"]
    assert vals == [os.path.join(work, "validation", f"step-{s}/batch-{b}") for s in (2, 4, 6, 8, 10)
                    for b in range(N_VAL)]

    # Periodic saves below the budget, then the final weights as checkpoint-12 before end_training.
    saves = [ev[1] for ev in events if ev[0] == "save_state"]
    assert saves == [os.path.join(work, f"checkpoint-{s}") for s in (2, 4, 6, 8, 10, 12)]
    assert names.index("end_training") > max(i for i, n in enumerate(names) if n == "save_state")
    assert os.path.realpath(os.path.join(work, "latest")) == os.path.realpath(saves[-1])
    dumped = _dumped_config(work)
    assert (dumped.screen_steps, dumped.seed) == (12, 42)
    assert names[-2:] == ["end_training", "returned"]


def test_fake_run_screen_seed_override(monkeypatch, tmp_path):
    events, work, _ = run_fake(monkeypatch, tmp_path, "screen_mp3d_all_256", ["--screen-steps", "3", "--seed", "43"])
    assert [ev for ev in events if ev[0] == "set_seed"] == [("set_seed", 43)]
    assert _dumped_config(work).seed == 43
    # The loader stage seeds are not the run seed: the train generator still starts from 1234.
    loaders = {ev[1]: ev[3] for ev in events if ev[0] == "loader"}
    assert loaders["train"].generator.initial_seed() == 1234
    assert loaders["val"].generator.initial_seed() == 3456


def test_fake_run_screen_loc360_row(monkeypatch, tmp_path):
    text = FAKE_CONFIG.replace("\nlr = 2e-4\n", "\nlr = 1e-4\n")
    assert text != FAKE_CONFIG
    events, work, _ = run_fake(monkeypatch, tmp_path, "screen_loc360_all_256", ["--screen-steps", "3"],
                               config_text=text)
    sched = _screen_scheduler(events)
    assert sched.total_steps == 103
    assert [g["max_lr"] for g in sched.optimizer.param_groups] == [1e-4, 1e-4]  # the 360Loc config's lr
    assert _names(events).count("forward") == 3
    assert [ev for ev in events if ev[0] == "validation_step"] == []  # like loc360_all_256: no validation loop
    loaders = {ev[1]: ev for ev in events if ev[0] == "loader"}
    assert loaders["val"][2] == 1  # batch_size_val, as in the base row
    assert loaders["train"][3].num_workers == 1
    saves = [ev[1] for ev in events if ev[0] == "save_state"]
    assert saves == [os.path.join(work, "checkpoint-2"), os.path.join(work, "checkpoint-3")]


def test_fake_run_screen_mp3d_single_row(monkeypatch, tmp_path):
    # The D3 pair's row: the single-view MP3D loader (base mp3d_single_256) with the screen recipe
    # on one process, so the plain forward and validation_step (the base row calls .module).
    events, work, _ = run_fake(monkeypatch, tmp_path, "screen_mp3d_single_256", ["--screen-steps", "3"])
    names = _names(events)
    assert _setup_sequence(events) == [s for s in EXPECTED_ORDER["loaders_before_model"] if s != "resume"]
    assert [ev for ev in events if ev[0] == "set_seed"] == [("set_seed", 42)]
    sched = _screen_scheduler(events)
    assert sched.total_steps == 103
    assert [g["max_lr"] for g in sched.optimizer.param_groups] == [2e-4, 2e-4]  # cfg.lr
    assert [ev[2] for ev in events if ev[0] == "forward"] == [0, 1, 2]
    assert "ddp_forward" not in names and "call_hook" not in names
    assert [ev[1] for ev in events if ev[0] == "validation_step"] == [
        os.path.join(work, "validation", f"step-2/batch-{b}") for b in range(N_VAL)]
    loaders = {ev[1]: ev for ev in events if ev[0] == "loader"}
    assert loaders["val"][2] == 2  # batch_size_train, as in the base row
    assert loaders["train"][3].num_workers == 32
    assert [ev[1] for ev in events if ev[0] == "save_state"] == [os.path.join(work, f"checkpoint-{s}") for s in (2, 3)]


def test_fake_run_screen_loc360_refuses_shuffle_train(monkeypatch, tmp_path):
    work = tmp_path / "runs" / "screen_loc_shuffle"
    with pytest.raises(SystemExit, match="IterableDataset"):
        run_fake(monkeypatch, tmp_path, "screen_loc360_all_256", ["--switch", "shuffle_train=true"], work_dir=work)
    assert not work.exists()


def test_fake_run_screen_d1_pair(monkeypatch, tmp_path):
    # Legacy arm: .module.forward, no wrapper call, no gradient sync.
    events, work, _ = run_fake(monkeypatch, tmp_path, "screen_d1_pair_256", ["--screen-steps", "4"])
    names = _names(events)
    assert names.count("forward") == 4
    assert "ddp_forward" not in names and "call_hook" not in names
    assert names.count("validation_step") == N_VAL  # at step 2, through .module
    assert _screen_scheduler(events).total_steps == 104
    assert [ev[1] for ev in events if ev[0] == "save_state"] == [os.path.join(work, f"checkpoint-{s}") for s in (2, 4)]
    # DDP arm: the same row with ddp_forward goes through the wrapper.
    events, work, _ = run_fake(monkeypatch, tmp_path, "screen_d1_pair_256",
                               ["--screen-steps", "4", "--switch", "ddp_forward=true"],
                               work_dir=tmp_path / "runs" / "d1_ddp")
    names = _names(events)
    forwards = [i for i, n in enumerate(names) if n == "forward"]
    assert len(forwards) == 4
    assert all(names[i - 2:i] == ["ddp_forward", "call_hook"] for i in forwards)
    (acc,) = [ev[1] for ev in events if ev[0] == "Accelerator"]
    assert [type(h).__name__ for h in acc["kwargs_handlers"]] == ["InitProcessGroupKwargs", "DistributedDataParallelKwargs"]


@pytest.mark.parametrize("wrong", [1, 3])
def test_fake_run_screen_d1_pair_needs_two_processes(wrong, monkeypatch, tmp_path):
    work = tmp_path / "runs" / f"d1_{wrong}"
    with pytest.raises(SystemExit, match="2 process.*this launch has {}".format(wrong)):
        run_fake(monkeypatch, tmp_path, "screen_d1_pair_256", num_processes=wrong, work_dir=work)
    assert not work.exists()


@pytest.mark.parametrize("screen_steps,max_steps,expected", [(8, 3, 3), (3, 7, 3)])
def test_fake_run_screen_with_max_steps(screen_steps, max_steps, expected, monkeypatch, tmp_path):
    events, work, _ = run_fake(monkeypatch, tmp_path, "screen_mp3d_all_256",
                               ["--screen-steps", str(screen_steps), "--max-steps", str(max_steps)])
    assert _names(events).count("forward") == expected
    assert _screen_scheduler(events).total_steps == screen_steps + 100
    assert [ev[1] for ev in events if ev[0] == "save_state"][-1] == os.path.join(work, f"checkpoint-{expected}")


# ----------------------------------------------------------------------------- two simulated ranks (D1 barrier)

def run_fake_ranks(monkeypatch, tmp_path, row, extra_args=()):
    """Run train.main as rank 0 (main), then as rank 1, into one work dir; returns the two event logs.

    A deterministic stand-in for a multi-process launch: the ranks meet only at
    wait_for_everyone, so how their events are ordered against each other follows from
    the barriers each log records (_barrier_segments). Rank 1 stands for every non-main rank.
    """
    work = tmp_path / "runs" / f"ranks_{row}"
    logs = []
    for rank in (0, 1):
        events, _, _ = run_fake(monkeypatch, tmp_path, row, extra_args, work_dir=work, process_index=rank)
        # Copy now: the next run_fake wraps this run's loader / scheduler wrappers, which keep appending here.
        logs.append(list(events))
    return logs


def _barrier_segments(events):
    """(segment, event) for every other event; segment = the wait_for_everyone calls made before it.

    An event in segment a on one rank happens before an event in segment b on another
    rank only when b > a; with b == a the two can run at the same time.
    """
    segment, out = 0, []
    for ev in events:
        if ev[0] == "wait_for_everyone":
            segment += 1
        else:
            out.append((segment, ev))
    return out


def _sync_skeleton(events):
    """The events that order the ranks: training forwards, barriers, the main-only saves and validation."""
    out = []
    for ev in events:
        if ev[0] == "forward":
            out.append(("forward", ev[2]))
        elif ev[0] == "wait_for_everyone":
            out.append(("barrier",))
        elif ev[0] == "save_state":
            out.append(("save_state", os.path.basename(ev[1])))
        elif ev[0] == "validation_step":
            out.append(("validation_step", "/".join(ev[1].split(os.sep)[-2:])))
    return out


def _expected_skeleton(row, rank, steps, ddp):
    """The legacy loop for FAKE_CONFIG (save_freq = val_freq = 2): per step one barrier before rank 0's
    save / validation, one barrier before end_training; ddp_forward adds one after each step's save / validation."""
    e = ALL_ENTRIES[row]
    seq = []
    for it in range(steps):
        seq += [("forward", it), ("barrier",)]
        if rank == 0 and it > 0 and it % 2 == 0:
            seq.append(("save_state", f"checkpoint-{it}"))
            if e["validation"] is not None:
                seq += [("validation_step", f"step-{it}/batch-{b}") for b in range(N_VAL)]
        if ddp:
            seq.append(("barrier",))
    if e.get("screen") and rank == 0:
        seq.append(("save_state", f"checkpoint-{steps}"))  # the screen's final weights
    return seq + [("barrier",)]


# The rows whose forward ddp_forward sends through the DDP wrapper, with their extra arguments and step count.
DDP_ROWS = [("mp3d_double_256", (), N_TRAIN * 2), ("mp3d_single_256", (), N_TRAIN * 2),
            ("loc360_all_256", (), N_TRAIN * 2), ("screen_d1_pair_256", ("--screen-steps", "4"), 4),
            ("mp3d_double_512_ddp3", (), N_TRAIN * 2), ("mp3d_double_512_ddp4", (), N_TRAIN * 2)]


def test_ddp_rows_are_the_multi_process_rows():
    assert sorted(r for r, _, _ in DDP_ROWS) == sorted(r for r, e in ALL_ENTRIES.items() if e["num_processes"] > 1)


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("row,extra,steps", DDP_ROWS)
def test_fake_ranks_barrier_sequence(row, extra, steps, ddp, monkeypatch, tmp_path):
    # Default: every rank's forwards, barriers, saves and validation in the legacy order, no extra barrier.
    # ddp_forward: the same plus one barrier on every rank after each step's main-only save / validation.
    switch = ["--switch", "ddp_forward=true"] if ddp else []
    ranks = run_fake_ranks(monkeypatch, tmp_path, row, [*extra, *switch])
    for rank, events in enumerate(ranks):
        assert _sync_skeleton(events) == _expected_skeleton(row, rank, steps, ddp)
        assert _names(events).count("ddp_forward") == (steps if ddp else 0)


@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("row,extra,steps", DDP_ROWS)
def test_fake_ranks_next_forward_after_main_only_work(row, extra, steps, ddp, monkeypatch, tmp_path):
    switch = ["--switch", "ddp_forward=true"] if ddp else []
    main, other = run_fake_ranks(monkeypatch, tmp_path, row, [*extra, *switch])
    # Both ranks make the same barrier calls; a missing one would hang a real run.
    assert _names(main).count("wait_for_everyone") == _names(other).count("wait_for_everyone")
    # Rank 0's main-only work as (segment, step): each save_state and validation_step.
    work = []
    for segment, ev in _barrier_segments(main):
        if ev[0] == "save_state":
            work.append((segment, int(os.path.basename(ev[1]).split("-")[1])))
        elif ev[0] == "validation_step":
            work.append((segment, int(os.path.basename(os.path.dirname(ev[1])).split("-")[1])))
    assert 2 in {step for _, step in work}
    other_forward = {ev[2]: segment for segment, ev in _barrier_segments(other) if ev[0] == "forward"}
    # (segment of rank 1's forward of the next step, segment of rank 0's save / validation of this step)
    pairs = [(other_forward[step + 1], segment) for segment, step in work if step + 1 in other_forward]
    assert pairs
    if ddp:
        # Rank 1's next forward, a DDP collective, starts only after the barrier that follows rank 0's
        # save / validation, so it never waits inside NCCL for the whole of that work.
        assert all(forward_segment > segment for forward_segment, segment in pairs)
    else:
        # Legacy order, unchanged: rank 1's next .module forward (no collective) runs while rank 0 saves / validates.
        assert all(forward_segment == segment for forward_segment, segment in pairs)

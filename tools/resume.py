"""Weights-only resume with an explicit name / dtype / shape contract (plan A1, "Resume rule").

The legacy trainers loaded `cfg.resume_from` by keeping every checkpoint tensor
whose name and shape matched the model and silently dropping the rest. Here a
`model.safetensors` is loaded only when it fits the model exactly, except for
the names a named transfer allows to be missing or extra (configs/entries.py,
TRANSFERS). Only weights are loaded: the optimizer and the scheduler start at
step 0. The checkpoint is only read, so it may live in a protected tree.

A model name that is absent from the checkpoint but shares its storage with a
name that is present (tied weights) is not missing: Accelerate's safetensors
writer keeps one name per storage, and loading that name fills both.
"""

import os

WEIGHTS_NAME = "model.safetensors"
_SHOW = 20  # names listed per problem in an error message


class ResumeError(RuntimeError):
    pass


def resolve_weights_file(path):
    """A checkpoint dir (with model.safetensors) or the file itself -> the file; error if missing."""
    if os.path.isdir(path):
        path = os.path.join(path, WEIGHTS_NAME)
    if not os.path.isfile(path):
        raise ResumeError(f"resume checkpoint not found: {path}")
    return path


def _within(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def refuse_source_inside(path, work_dir):
    """A run must not save next to, inside, or around the checkpoint it starts from
    (checkpoint-N, latest, logs or the config dump could overwrite or mix with it)."""
    real = os.path.realpath(path)
    source_dir = os.path.dirname(real) if os.path.isfile(real) or real.endswith(".safetensors") else real
    work = os.path.realpath(work_dir)
    if _within(source_dir, work) or _within(work, source_dir):
        raise ResumeError(f"resume checkpoint {path} and the work dir {work_dir} overlap; "
                          "use a fresh --work-dir / --run-id")


def name_allowed(name, patterns):
    """'prefix.*' matches names starting with 'prefix.'; any other pattern is an exact name."""
    for pattern in patterns:
        if pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True
    return False


def _storage_key(tensor):
    # Same key as accelerate.utils.id_tensor_storage, which decides the names it deduplicates.
    storage = tensor.untyped_storage()
    return tensor.device, storage.data_ptr(), storage.nbytes()


def _listing(title, names):
    shown = ", ".join(names[:_SHOW]) + (" ..." if len(names) > _SHOW else "")
    return f"  {title} ({len(names)}): {shown}"


def check_state(state_dict, model_state, allowed_missing=(), allowed_extra=()):
    """Compare checkpoint tensors with model.state_dict(); raise ResumeError on any violation.

    Returns a report dict: matched / extra / missing / aliased name lists.
    """
    matched = [k for k in state_dict if k in model_state]
    if not matched:
        raise ResumeError(f"zero checkpoint tensors match the model ({len(state_dict)} in the checkpoint, "
                          f"{len(model_state)} in the model)")
    extra = sorted(k for k in state_dict if k not in model_state)
    absent = sorted(k for k in model_state if k not in state_dict)
    loaded_storage = {_storage_key(model_state[k]) for k in matched}
    aliased = [k for k in absent if _storage_key(model_state[k]) in loaded_storage]
    missing = [k for k in absent if k not in aliased]

    problems = []
    bad_extra = [k for k in extra if not name_allowed(k, allowed_extra)]
    if bad_extra:
        problems.append(_listing("extra names in the checkpoint", bad_extra))
    bad_missing = [k for k in missing if not name_allowed(k, allowed_missing)]
    if bad_missing:
        problems.append(_listing("model names missing from the checkpoint", bad_missing))
    bad_dtype = [f"{k} {state_dict[k].dtype}!={model_state[k].dtype}"
                 for k in matched if state_dict[k].dtype != model_state[k].dtype]
    if bad_dtype:
        problems.append(_listing("dtype mismatches (checkpoint!=model)", bad_dtype))
    bad_shape = [f"{k} {tuple(state_dict[k].shape)}!={tuple(model_state[k].shape)}"
                 for k in matched if state_dict[k].shape != model_state[k].shape]
    if bad_shape:
        problems.append(_listing("shape mismatches (checkpoint!=model)", bad_shape))
    if problems:
        raise ResumeError("checkpoint does not fit the model under this transfer:\n" + "\n".join(problems))
    return dict(matched=matched, extra=extra, missing=missing, aliased=aliased)


def load_weights_only(model, path, allowed_missing=(), allowed_extra=()):
    """Load model.safetensors into `model` in place (before accelerator.prepare); return the report.

    With a checkpoint that passes the rule, the loaded state equals what the legacy
    name+shape filter produced from the same file.
    """
    from safetensors.torch import load_file

    path = resolve_weights_file(path)
    state_dict = load_file(path, device="cpu")
    model_dict = model.state_dict()
    report = check_state(state_dict, model_dict, allowed_missing, allowed_extra)
    model_dict.update({k: state_dict[k] for k in report["matched"]})
    model.load_state_dict(model_dict)
    report["file"] = path
    return report

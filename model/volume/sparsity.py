"""Volume sparsity budget (switch volume_sparsity): a one-sided budget on the share of rendered volume Gaussians,
carried by their opacity logits, used by the joint model. Studied on MP3D 256x512 as an 8,000-step fine-tune of the LS
checkpoint (README, "Volume sparsity"); horizons are global training steps."""

import torch

VISIBLE_OPACITY = 1.0 / 255.0  # the rasteriser's alpha cut (prune_invisible), compared in float32

SPARSITY_DEFAULTS = dict(  # the studied schedule (configs/OmniScene/screen/s3k_sp3d.py)
    budget=0.35,  # target share rho of rendered volume Gaussians among all volume slots
    budget_start=None,  # rho_t at step 0, linear to `budget` at `ramp_steps`; no default: every training config sets
    # the ceiling of its init's rendered volume share (README, "Volume sparsity"), and train.py refuses a run without it
    ramp_steps=3600,
    budget_weight=0.05,
    collapse_floor=0.5,  # train.py SparsityMonitor: from collapse_from on, a trailing collapse_window mean of the share
    collapse_from=3600,  # below collapse_floor x that of its target stops the run (exit 7); None: no floor
    collapse_window=100,
)


def sparsity_rho(step, args):
    """rho_t: linear from budget_start at step 0 to budget at ramp_steps, then constant."""
    if args["budget_start"] is None:
        raise ValueError("sparsity_args.budget_start is not set")
    frac = min(step / args["ramp_steps"], 1.0)
    return args["budget_start"] + (args["budget"] - args["budget_start"]) * frac


def volume_sparsity_budget(gaussians_volume, split, step, args, rho_override=None):
    """Sparsity budget on the volume_gs output [(b v), N, 14] (opacity in channel 6). B is the share of volume slots
    the rasteriser blends (opacity >= 1/255). Training split: loss = budget_weight * relu(B_st - rho_t)^2 with
    B_st = B + (S - S.detach()) and S = sum of logit(opacity) over the blended Gaussians / slots, so while B > rho_t
    every blended Gaussian's pre-sigmoid opacity gets the gradient 2 w (B - rho_t) / slots, except a saturated one
    (opacity > 1 - 1e-6, where the logit is clamped and its gradient is 0); nothing else is touched and the forward is
    unchanged. Returns (loss or None off the training split, stats)."""
    opacity = gaussians_volume[..., 6]
    slots = gaussians_volume.shape[0] * gaussians_volume.shape[1]  # (batch x cylinders) x slots per cylinder
    alive = (opacity.detach().float() >= VISIBLE_OPACITY).to(opacity.dtype)
    share = alive.sum() / slots
    stats = dict(volume_share=float(share))
    if split != "train":
        return None, stats
    rho = sparsity_rho(step, args) if rho_override is None else rho_override
    carrier = (alive * torch.logit(opacity.float(), eps=1e-6).to(opacity.dtype)).sum() / slots
    share_st = share + (carrier - carrier.detach())
    stats["volume_share_target"] = float(rho)
    return args["budget_weight"] * torch.relu(share_st - rho) ** 2, stats


class VolumeSparsityMixin:
    """The sparsity budget on a model whose volume_gs returns [(b v), N, 14] cylinder-frame Gaussians. Call
    _init_volume_sparsity in __init__ and _apply_sparsity on the volume_gs output."""

    def _init_volume_sparsity(self, volume_sparsity, sparsity_args):
        unknown = set(sparsity_args or {}) - set(SPARSITY_DEFAULTS)
        if unknown:
            raise ValueError(f"sparsity_args: unknown keys {sorted(unknown)}")
        self.volume_sparsity = volume_sparsity
        self.sparsity_args = dict(SPARSITY_DEFAULTS, **(sparsity_args or {}))
        self._sparsity_rho_override = None  # test-only: pins rho_t (never set from a config)

    def _apply_sparsity(self, gaussians_volume, split, step):
        """volume_sparsity_budget with this model's args; (None, {}) when the switch is off."""
        if not self.volume_sparsity:
            return None, {}
        return volume_sparsity_budget(gaussians_volume, split, step, self.sparsity_args, self._sparsity_rho_override)

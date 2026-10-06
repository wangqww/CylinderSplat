# Volume sparsity, the stage-3 schedule (base of s3k_ls_sp3): s3k_base + the sparsity budget, rho_t budget_start ->
# 0.35 over 3,600 steps, budget_weight 0.05, and a stop if the trailing-100 share falls below 0.5 x rho_t from step
# 3,600 (at the final budget only). budget_start is set per run from its init (its largest training-batch share
# + 0.03, rounded up to a multiple of 0.05): train s3k_ls_sp3.py; this base alone has no budget_start and train.py
# refuses it.
_base_ = ['./s3k_base.py']

switches = dict(volume_sparsity=True)
model = dict(
    sparsity_args=dict(
        budget=0.35, ramp_steps=3600, budget_weight=0.05, collapse_floor=0.5, collapse_from=3600,
    )
)

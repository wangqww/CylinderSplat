# Volume sparsity recipe: s3k_sp3d from LS; budget_start 0.85 (largest training-batch share of LS 0.808).
_base_ = ['./s3k_sp3d.py']

model = dict(sparsity_args=dict(budget_start=0.85))

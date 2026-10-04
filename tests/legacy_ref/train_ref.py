"""Frozen copies of the legacy trainer code that train.py must reproduce.

Bodies are verbatim from f7b20b9 (git show f7b20b9:<path>); only the function
wrappers, whose parameters carry the names the bodies use, are added. Do not edit.
"""

import torch
from safetensors.torch import load_file


def onecycle_scheduler(optimizer, cfg, train_dataloader, max_num_epochs):
    # train_mp3d_cylinder_double_256.py:130-142 (same in train_mp3d_cylinder_single_256.py:130-142
    # and train_360Loc_cylinder_double_all_512.py:130-142)
    # PanSplat Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=len(train_dataloader) * max_num_epochs + 100,  # 稍微增加一点以确保覆盖所有步数
        pct_start=0.01,
        cycle_momentum=False,  # Adam 优化器通常设为 False
        anneal_strategy='cos',
        # 下面两个参数是 OneCycle 的标配，控制初始 LR 和最终 LR
        # 如果 cfg 没有定义，这里给了常用的默认值
        div_factor=25.0,       # init_lr = max_lr / 25
        final_div_factor=10000.0, # final_lr = init_lr / 10000
    )
    return scheduler


def warmup_cosine_scheduler(optimizer, cfg, accelerator):
    # train_mp3d_cylinder_double.py:116-124 (same in train_vigor_cylinder_double.py:116-124
    # and train_mp3d_cylinder_double_512.py:115-123)
    # consine lr scheduler
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        1 / (cfg.warmup_steps*accelerator.num_processes),
        1,
        total_iters=cfg.warmup_steps*accelerator.num_processes,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_train_steps*accelerator.num_processes, eta_min=cfg.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, scheduler], milestones=[cfg.warmup_steps*accelerator.num_processes])
    return scheduler


def resume(my_model, cfg, accelerator):
    # train_mp3d_cylinder_double_256.py:175-184 (same block in all six trainers: :175-184 in the
    # *_256 scripts, :160-169 in train_mp3d_cylinder_double.py / train_vigor_cylinder_double.py,
    # :147-156 in train_mp3d_cylinder_double_512.py)
    path = cfg.resume_from
    if path:
        accelerator.print(f"Resuming from checkpoint {path}")
        state_dict = load_file(path, device="cpu")
        model_dict = my_model.state_dict()

        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        my_model.load_state_dict(model_dict)
        accelerator.print("Model weights loaded successfully before prepare().")

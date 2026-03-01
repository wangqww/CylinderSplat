optimizer = dict(
    type='AdamW',
    lr=5e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01
)

grad_max_norm = 1.0


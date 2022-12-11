_base_ = [
    '../_base_/models/adabins.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=256,
        norm_cfg=norm_cfg),
)

find_unused_parameters = True
SyncBN = True

# optimizer
max_lr = 0.000357
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    weight_decay=0.1,
    paramwise_cfg=dict(
        custom_keys={
            'decode_head': dict(lr_mult=10),  # 10 lr
        }))

# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)

momentum_config = dict(
    policy='OneCycle'
)

# runtime
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=19997 * 25)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=19997)
evaluation = dict(
    by_epoch=False,
    start=0,
    interval=500,
    pre_eval=True,
    rule='less',
    save_best='silog',
    greater_keys=(
        "a1",
        "a2",
        "a3"),
    less_keys=(
        "abs_rel",
        "rmse",
        "log_10",
        "rmse_log",
        "silog",
        "sq_rel"))
# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

# optimizer

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings

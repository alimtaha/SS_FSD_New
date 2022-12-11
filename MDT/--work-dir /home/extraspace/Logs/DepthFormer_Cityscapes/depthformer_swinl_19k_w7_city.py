norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
conv_stem_norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
    backbone=dict(
        type='DepthFormerSwin',
        pretrain_img_size=224,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        conv_norm_cfg=dict(type='BN', requires_grad=True),
        depth=50,
        num_stages=0),
    decode_head=dict(
        type='DenseDepthHead',
        in_channels=[64, 192, 384, 768, 1536],
        up_sample_channels=[64, 192, 384, 768, 1536],
        channels=64,
        align_corners=True,
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=1.0),
        act_cfg=dict(type='LeakyReLU', inplace=True),
        min_depth=0.001,
        max_depth=256),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    neck=dict(
        type='HAHIHeteroNeck',
        positional_encoding=dict(type='SinePositionalEncoding', num_feats=256),
        in_channels=[64, 192, 384, 768, 1536],
        out_channels=[64, 192, 384, 768, 1536],
        embedding_dim=512,
        scales=[1, 1, 1, 1, 1]))
dataset_type = 'CSDataset'
data_root = '/home/extraspace/Datasets/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 600)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='RandomCrop', crop_size=(300, 600)),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='ColorAug',
        prob=1,
        gamma_range=[0.9, 1.1],
        brightness_range=[0.9, 1.1],
        color_range=[0.9, 1.1]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 2048),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CSDataset',
        data_root='/home/extraspace/Datasets/cityscapes/Depth_Training_Extra/',
        img_dir='leftImg8bit',
        cam_dir='camera',
        ann_dir='depth',
        depth_scale=256,
        split='cityscapes_train_extra_edited.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='DepthLoadAnnotations'),
            dict(type='RandomCrop', crop_size=(300, 600)),
            dict(type='RandomRotate', prob=0.5, degree=2.5),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='ColorAug',
                prob=1,
                gamma_range=[0.9, 1.1],
                brightness_range=[0.9, 1.1],
                color_range=[0.9, 1.1]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'depth_gt'])
        ],
        garg_crop=False,
        eigen_crop=False,
        min_depth=0.001,
        max_depth=256),
    val=dict(
        type='CSDataset',
        data_root='/home/extraspace/Datasets/cityscapes/Depth_Training/',
        img_dir='leftImg8bit',
        cam_dir='camera',
        ann_dir='depth',
        depth_scale=256,
        split='cityscapes_val_edited.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 2048),
                flip=True,
                flip_direction='horizontal',
                transforms=[
                    dict(type='RandomFlip', direction='horizontal'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        garg_crop=False,
        eigen_crop=False,
        min_depth=0.001,
        max_depth=256),
    test=dict(
        type='CSDataset',
        data_root='/home/extraspace/Datasets/cityscapes/Depth_Training/',
        img_dir='leftImg8bit',
        cam_dir='camera',
        ann_dir='depth',
        depth_scale=256,
        split='cityscapes_val_edited.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 2048),
                flip=True,
                flip_direction='horizontal',
                transforms=[
                    dict(type='RandomFlip', direction='horizontal'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        garg_crop=False,
        eigen_crop=False,
        min_depth=0.001,
        max_depth=256))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
max_lr = 0.0001
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=12800,
    warmup_ratio=0.001,
    min_lr_ratio=1e-08,
    by_epoch=False)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=499925)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(
    by_epoch=False,
    start=0,
    interval=10,
    pre_eval=True,
    rule='less',
    save_best='silog',
    greater_keys=('a1', 'a2', 'a3'),
    less_keys=('abs_rel', 'rmse', 'log_10', 'rmse_log', 'silog', 'sq_rel'))
work_dir = '--work-dir /home/extraspace/Logs/DepthFormer_Cityscapes/'
gpu_ids = range(0, 1)

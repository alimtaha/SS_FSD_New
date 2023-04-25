# dataset settings
dataset_type = 'CSDataset'
data_root = '/home/extraspace/Datasets/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 600)  # (352,704)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # changed from disparity load annotations to depth
    dict(type='DepthLoadAnnotations'),
    #dict(type='Resize', img_scale=(2048, 1024), keep_ratio=True),
    #dict(type='KBCrop', depth=True),
    dict(type='RandomCrop', crop_size=(300, 600)),  # (352,704)
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='ColorAug', prob=1, gamma_range=[
            0.9, 1.1], brightness_range=[
            0.9, 1.1], color_range=[
                0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=False),
    #dict(type='KBCrop', depth=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 2048),  # (1024,2048)
        flip=True,
        flip_direction='horizontal',
        transforms=[
            #dict(type='RandomCrop', crop_size=(400, 800)),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root='/home/extraspace/Datasets/cityscapes/Depth_Training_Extra/',
        img_dir='leftImg8bit',
        cam_dir='camera',
        ann_dir='depth',
        depth_scale=256,
        split='cityscapes_train_extra_edited.txt',
        pipeline=train_pipeline,
        garg_crop=False,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=256),
    val=dict(
        type=dataset_type,
        data_root='/home/extraspace/Datasets/cityscapes/Depth_Training/',
        img_dir='leftImg8bit',
        cam_dir='camera',
        ann_dir='depth',
        depth_scale=256,
        split='cityscapes_val_edited.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=256),
    test=dict(
        type=dataset_type,
        data_root='/home/extraspace/Datasets/cityscapes/Depth_Training/',
        img_dir='leftImg8bit',
        cam_dir='camera',
        ann_dir='depth',
        depth_scale=256,
        split='cityscapes_val_edited.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=256))

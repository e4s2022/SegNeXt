# dataset settings
dataset_type = 'CelebAMaskHQDataset'
data_root = './data/CelebAMaskHQ'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(1.0, 1.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        img_ratios=[0.5],  # Single-scale testing
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip', prob=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])

    # dict(type='Resize', img_scale=(512, 512), ratio_range=(1.0, 1.0)),
    # # dict(type='RandomFlip'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img'], 
    #      meta_keys=('filename', 'ori_filename', 'ori_shape',
    #                 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='CelebA-HQ-img/',
            ann_dir='CelebA-HQ-mask/',
            pipeline=train_pipeline,
            split="train_split.txt")),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='CelebA-HQ-img/',
        ann_dir='CelebA-HQ-mask/',
        pipeline=test_pipeline,
        split="val_split.txt"),
    # test=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='images/validation',
    #     ann_dir='annotations/validation',
    #     pipeline=test_pipeline)
)

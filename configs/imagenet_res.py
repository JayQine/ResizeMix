_base_ = './train_base.py'

net_name = "resnet101"  # res101
train_params=dict(
    epochs=300,
)

optim=dict(
    weight_decay=4e-5,
    if_resume=False,
    resume=dict(
        load_path='',
        load_epoch=191,
    ),
    use_warm_up=False,
    warm_up=dict(
        epoch=5,
        init_lr=1e-5,
        target_lr=0.5,
    ),
)

data=dict(
    num_threads=4,
    resize_batch=16, # 32 in default
    batch_size=64,
    dataset='imagenet',
    color=False,
    crop_test=False
)

augment=dict(
    randaugment=dict(
        if_use=False,
        N=2,
        M=14
    ),
    cutout=0,
    cutmix=dict(
        prob=0.0,
        beta=1.0
    ),
    resizemix=dict(
        if_use=True,
        alpha=0.1,
        beta=0.8
    ),
    batch_aug=False,
    target_layer=['layer4', '2']
)
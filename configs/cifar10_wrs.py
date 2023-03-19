_base_ = './train_base.py'

net_name = "wresnet28_10"
train_params=dict(
    epochs=200,
)

optim=dict(
    init_lr=0.2,
    weight_decay=5e-4,
    if_resume=False,
    resume=dict(
        load_path='',
        load_epoch=191,
    ),
    use_warm_up=True,
    warm_up=dict(
        epoch=5,
        init_lr=1e-5,
        target_lr=0.2,
    ),
)

data=dict(
    use_dali=False,
    num_examples=50000,
    batch_size=256,
    dataset='cifar10',
    input_size=(3,32,32),
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
    target_layer=['layer3', '3']
)
#!/usr/bin/env sh

# train cifar
python run_apis/train_dist.py \
    --report_freq 100 \
    --data_path /data/cifar \
    --port 23333 \
    --config cifar100_wrs.py \
    --save ./


# train imagenet
# python -m torch.distributed.launch --nproc_per_node=8 run_apis/train_dist.py \
#     --launcher pytorch \
#     --report_freq 400 \
#     --data_path /data/imagenet \
#     --port 23333 \
#     --config imagenet_res.py
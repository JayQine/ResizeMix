import argparse
import ast
import logging
import os
import sys; sys.path.append(os.path.join(sys.path[0], '..'))
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn import DataParallel
from dataset.data import get_dataloaders
from dataset import imagenet_data_dali
from mmcv import Config
from models import get_model, num_class
from tensorboardX import SummaryWriter
from tools import env, utils
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds

from trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train_Params")
    parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
    parser.add_argument('--data_path', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--load_path', type=str, default='./model_path', help='model loading path')
    parser.add_argument('--save', type=str, default='../', help='experiment name')
    parser.add_argument('--tb_path', type=str, default='', help='tensorboard output path')
    parser.add_argument('--meas_lat', type=ast.literal_eval, default='False', help='whether to measure the latency of the model')
    parser.add_argument('--job_name', type=str, default='', help='job_name')
    parser.add_argument('--port', type=int, default=23333, help='dist port')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--evaluation', type=ast.literal_eval, default='False', help='first evaluation')
    parser.add_argument('--config', type=str, default='', help='the file of the config')
    args = parser.parse_args()

    config = Config.fromfile(os.path.join('configs', args.config))
    if config.net_config:
        net_config = config.pop('net_config')

    if args.launcher == 'none':
        distributed = False
        local_rank = args.local_rank
        world_size = 0
    else:
        distributed = True
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '%d' % args.port
        env.init_dist(args.launcher)
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S-") + args.job_name
        args.save = os.path.join(args.save, args.job_name)
        if local_rank == 0:
            utils.create_exp_dir(args.save)
            os.system('cp -r ./* '+args.save)
    else:
        args.save = os.path.join(args.save, 'output')
        if local_rank == 0:
            utils.create_exp_dir(args.save)

    if args.tb_path == '':
        args.tb_path = args.save

    env.get_root_logger(log_dir=args.save, rank=local_rank)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    if config.train_params.use_seed:
        utils.set_seed(config.train_params.seed)

    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(config.pretty_text)
    writer = SummaryWriter(args.tb_path)

    model = get_model(config, num_class(config.data.dataset))
    model.eval()
    if hasattr(model, 'net_config'):
        logging.info("Network Structure: \n" + '|\n'.join(map(str, model.net_config)))
    if args.meas_lat:
        latency_cpu = utils.latency_measure(model, (3, 224, 224), 1, 2000, mode='cpu')
        logging.info('latency_cpu (batch 1): %.2fms' % latency_cpu)
        latency_gpu = utils.latency_measure(model, (3, 224, 224), 32, 1000, mode='gpu')
        logging.info('latency_gpu (batch 32): %.2fms' % latency_gpu)
        
    params = utils.count_parameters_in_MB(model)
    logging.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(model, input_size=config.data.input_size)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)

    if distributed:
        model.cuda(local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model.cuda()
        model = DataParallel(model)

    # whether to resume from a checkpoint
    if config.optim.if_resume:
        utils.load_model(model, config.optim.resume.load_path, distributed)
        start_epoch = config.optim.resume.load_epoch + 1
    else:
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        config.optim.init_lr,
        momentum=config.optim.momentum,
        weight_decay=config.optim.weight_decay
    )
    
    train_loader, val_loader = get_dataloaders(
        config.data, config.augment, world_size, local_rank, args.data_path
    )

    trainloader_size = train_loader._size if config.data.use_dali else len(train_loader.dataset)
    scheduler = get_lr_scheduler(config, optimizer, trainloader_size)
    scheduler.last_step = start_epoch * (trainloader_size // config.data.batch_size + 1)-1

    trainer = Trainer(train_loader, val_loader, optimizer, criterion, 
                    scheduler, config, args.report_freq, distributed)

    best_epoch = [0, 0, 0] # [epoch, acc_top1, acc_top5]
    if args.evaluation:
        with torch.no_grad():
            val_acc_top1, val_acc_top5, batch_time, data_time = trainer.infer(model, start_epoch-1)
        if val_acc_top1 > best_epoch[1]:
            best_epoch = [start_epoch-1, val_acc_top1, val_acc_top5]
        logging.info('BEST EPOCH %d  val_top1 %.2f val_top5 %.2f', best_epoch[0], best_epoch[1], best_epoch[2])

    for epoch in range(start_epoch, config.train_params.epochs):
        train_acc_top1, train_acc_top5, train_obj, batch_time, data_time = trainer.train(model, epoch)
        
        with torch.no_grad():
            val_acc_top1, val_acc_top5, batch_time, data_time = trainer.infer(model, epoch)
        if val_acc_top1 > best_epoch[1]:
            best_epoch = [epoch, val_acc_top1, val_acc_top5]
            if local_rank==0:
                utils.save(model, os.path.join(args.save, 'weights.pt'))
        logging.info('BEST EPOCH %d  val_top1 %.2f val_top5 %.2f', best_epoch[0], best_epoch[1], best_epoch[2])

        if local_rank == 0:
            writer.add_scalar('train_acc_top1', train_acc_top1, epoch)
            writer.add_scalar('train_loss', train_obj, epoch)
            writer.add_scalar('val_acc_top1', val_acc_top1, epoch)

    if hasattr(model.module, 'net_config'):
        logging.info("Network Structure: \n" + '|\n'.join(map(str, model.module.net_config)))
    logging.info("Params = %.2fMB" % params)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)

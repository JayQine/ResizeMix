import logging
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tools import env, utils
from tools.grad_cam import GradCam
from tools.utils import data_prefetcher
from tools.utils import rand_bbox, crop_test, cutbox, getbox, getbox_new

class Trainer(object):
    def __init__(self, train_data, val_data, optimizer=None, criterion=None, 
                scheduler=None, config=None, report_freq=None, distributed=False):
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.augment = self.config.augment
        self.report_freq = report_freq
        self.distributed = distributed
        if True:
            from augment import augmentations_cuda
            self.randaugment = augmentations_cuda.RandAugment_cuda(2, 14)
        if hasattr(config.data, 'use_dali'):
            self.use_dali = config.data.use_dali
        else:
            self.use_dali = False
    
    def train(self, model, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        model.train()

        start = time.time()
        if self.use_dali:
            try:
                data = next(self.train_data)
                input = data[0]["data"]
                target = data[0]["label"].squeeze().cuda().long()
            except StopIteration:
                input, target = None, None
        else:
            try:
                train_loader = iter(self.train_data)
                input, target = train_loader.next()
                input, target = input.cuda(), target.cuda()
            except StopIteration:
                input, target = None, None
        step = 0
        while input is not None:      
            data_t = time.time() - start
            self.scheduler.step()
            n = input.size(0)
            if step==0:
                logging.info('epoch %d lr %e', epoch, self.optimizer.param_groups[0]['lr'])
            self.optimizer.zero_grad()
            # cutmix
            rand = np.random.rand(1)
            if self.augment.cutmix.beta > 0 and rand < self.augment.cutmix.prob :
                # generate mixed sample
                lam = np.random.beta(self.augment.cutmix.beta, self.augment.cutmix.beta)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                logits = model(input)
                loss = self.criterion(logits, target_a) * lam + self.criterion(logits, target_b) * (1. - lam)
            
            # resizemix
            elif self.augment.resizemix.if_use:
                rate = random.uniform(self.augment.resizemix.alpha, self.augment.resizemix.beta)
                data_small = F.interpolate(input, scale_factor=rate)
                m1, n1, m2, n2 = getbox_new(input.size(), data_small.shape[2:])
                rand_index = torch.randperm(input.size()[0])
                for i in range(input.shape[0]):
                    input[i, :, m1[i]:m2[i], n1[i]:n2[i]] = data_small[rand_index[i]]
                lam = 1 - (data_small.size()[-1] * data_small.size()[-2]) / (input.size()[-1] * input.size()[-2])
                target_a = target
                target_b = target[rand_index]
                if self.augment.batch_aug:
                    data1 = self.randaugment(input)
                    data2 = self.randaugment(input)
                    input  = torch.cat((data1, data2), dim=0)
                    target_a = target_a.repeat(2)
                    target_b = target_b.repeat(2)
                logits = model(input)
                loss = self.criterion(logits, target_a) * lam + self.criterion(logits, target_b) * (1. - lam)
            
            else:
                logits= model(input)
                loss = self.criterion(logits, target)
                target_a = target 
                
            loss.backward()
            if self.config.optim.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.optim.grad_clip)
            self.optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target_a, topk=(1, 5))
            
            if self.distributed:
                loss = env.reduce_tensor(loss.data)
                prec1 = env.reduce_tensor(prec1)
                prec5 = env.reduce_tensor(prec5)
                torch.cuda.synchronize()

            batch_t = time.time() - start
            start = time.time()
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)
            if step!=0 and step % self.report_freq == 0:
                logging.info(
                    'Train epoch %03d step %03d | loss %.4f  top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f', 
                    epoch, step, objs.avg, top1.avg, top5.avg, batch_time.avg, data_time.avg)

            if self.use_dali:
                try:
                    data = next(self.train_data)
                    input = data[0]["data"]
                    target = data[0]["label"].squeeze().cuda().long()
                except StopIteration:
                    input, target = None, None
            else:
                try:
                    input, target = train_loader.next()
                    input, target = input.cuda(), target.cuda()
                except StopIteration:
                    input, target = None, None
            step += 1
        
        logging.info('EPOCH%d Train_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', 
                                epoch, top1.avg, top5.avg, batch_time.avg, data_time.avg)

        return top1.avg, top5.avg, objs.avg, batch_time.avg, data_time.avg


    def infer(self, model, epoch=0):
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        data_time = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        model.eval()

        start = time.time()
        if self.use_dali:
            try:
                data = next(self.val_data)
                input = data[0]["data"]
                target = data[0]["label"].squeeze().cuda().long()
            except StopIteration:
                input, target = None, None
        else:
            try:
                val_loader = iter(self.val_data)
                input, target = val_loader.next()
                input, target = input.cuda(), target.cuda()
            except StopIteration:
                input, target = None, None

        step = 0
        while input is not None:
            step += 1
            data_t = time.time() - start
            n = input.size(0)
            
            # crop test
            cropped, alpha = crop_test(input, 24)
            if self.config.data.crop_test:
                output1 = model(input)
                output2 = model(cropped)
                logits = output1 * (1 - alpha) + output2 * alpha
            else:
                logits = model(input)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            
            if self.distributed:
                prec1 = env.reduce_tensor(prec1)
                prec5 = env.reduce_tensor(prec5)
                torch.cuda.synchronize()
            batch_t = time.time() - start
            start = time.time()
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            data_time.update(data_t)
            batch_time.update(batch_t)

            if step % self.report_freq == 0:
                logging.info(
                    'Val epoch %03d step %03d | top1_acc %.2f  top5_acc %.2f | batch_time %.3f  data_time %.3f', 
                    epoch, step, top1.avg, top5.avg, batch_time.avg, data_time.avg)
            if self.use_dali:
                try:
                    data = next(self.val_data)
                    input = data[0]["data"]
                    target = data[0]["label"].squeeze().cuda().long()
                except StopIteration:
                    input, target = None, None
            else:
                try:
                    input, target = val_loader.next()
                    input, target = input.cuda(), target.cuda()
                except StopIteration:
                    input, target = None, None

        logging.info('EPOCH%d Valid_acc  top1 %.2f top5 %.2f batch_time %.3f data_time %.3f', 
                                epoch, top1.avg, top5.avg, batch_time.avg, data_time.avg)
        return top1.avg, top5.avg, batch_time.avg, data_time.avg


import logging
import os
import shutil
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.cur = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load_net_config(path):
    with open(path, 'r') as f:
        net_config = ''
        while True:
            line = f.readline().strip()
            if 'net_type' in line:
                net_type = line.split(': ')[-1]
                break
            else:
                net_config += line
    return net_config, net_type


def load_model(model, model_path, distributed=False):
    logging.info('Start loading the model from ' + model_path)
    if 'http' in model_path:
        model_addr = model_path
        model_path = model_path.split('/')[-1]
        if (not distributed) or (distributed and dist.get_rank()==0):
            if os.path.isfile(model_path):
                os.system('rm ' + model_path)
            os.system('wget -q ' + model_addr)
        if distributed:
            dist.barrier()
    model.load_state_dict(torch.load(model_path, 
        map_location='cuda:{}'.format(dist.get_rank()) if distributed else None))
    logging.info('Loading the model finished!')


def create_exp_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def parse_net_config(net_config):
    str_configs = net_config.split('|')
    return [eval(str_config) for str_config in str_configs]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_logging(save_path, log_name='log.txt'):
    log_format = '%(asctime)s %(message)s'
    date_format = '%m/%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt=date_format)
    fh = logging.FileHandler(os.path.join(save_path, log_name))
    fh.setFormatter(logging.Formatter(log_format, date_format))
    logging.getLogger().addHandler(fh)


def create_save_dir(save_path, job_name):
    if job_name != '':
        job_name = time.strftime("%Y%m%d-%H%M%S-") + job_name
        save_path = os.path.join(save_path, job_name)
        create_exp_dir(save_path)
        os.system('cp -r ./* '+save_path)
        save_path = os.path.join(save_path, 'output')
        create_exp_dir(save_path)
    else:
        save_path = os.path.join(save_path, 'output')
        create_exp_dir(save_path)
    return save_path, job_name


def latency_measure(module, input_size, batch_size, meas_times, mode='gpu'):
    assert mode in ['gpu', 'cpu']
    
    latency = []
    module.eval()
    input_size = (batch_size,) + tuple(input_size)
    input_data = torch.randn(input_size)
    if mode=='gpu':
        input_data = input_data.cuda()
        module.cuda()

    for i in range(meas_times):
        with torch.no_grad():
            start = time.time()
            _ = module(input_data)
            torch.cuda.synchronize()
            if i >= 100:
                latency.append(time.time() - start)
    # print(np.mean(latency) * 1e3, 'ms')
    return np.mean(latency) * 1e3


def latency_measure_fw(module, input_data, meas_times):
    latency = []
    module.eval()
    
    for i in range(meas_times):
        with torch.no_grad():
            start = time.time()
            output_data = module(input_data)
            torch.cuda.synchronize()
            if i >= 100:
                latency.append(time.time() - start)
    # print(np.mean(latency) * 1e3, 'ms')
    return np.mean(latency) * 1e3, output_data


def record_topk(k, rec_list, data, comp_attr, check_attr):
    def get_insert_idx(orig_list, data, comp_attr):
        start = 0
        end = len(orig_list)
        while start < end:
            mid = (start + end) // 2
            if data[comp_attr] < orig_list[mid][comp_attr]:
                start = mid + 1
            else:
                end = mid
        return start
    
    if_insert = False
    insert_idx = get_insert_idx(rec_list, data, comp_attr)
    if insert_idx < k:
        rec_list.insert(insert_idx, data)
        if_insert = True
    while len(rec_list) > k:
        rec_list.pop()
    return if_insert

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        # nump_array = img
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def crop_test(img, length):
    h = img.size(2) 
    w = img.size(3)
    mask = np.zeros((h, w), np.float32)
    # y = np.random.randint(h)
    # x = np.random.randint(w)
    y = int(h / 2)
    x = int(w / 2)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    alpha = ((x2 - x1) * (y2 - y1)) / ((img.size()[-1]) * (img.size()[-2]))
    mask[y1: y2, x1: x2] = 1.                
    mask = torch.from_numpy(mask)       
    mask = mask.expand_as(img).cuda()
    img = img.mul(mask)
    return img, alpha

def cutout(img, center, v):
    n, h, w = img.size(0), img.size(-2), img.size(-1)
    mask = np.ones((n, h, w), np.float32)
    
    y = center[:, 0]
    x = center[:, 1]

    y1 = (np.clip(y - v // 2, 0, h))
    y2 = (np.clip(y + v // 2, 0, h))
    x1 = (np.clip(x - v // 2, 0, w))
    x2 = (np.clip(x + v // 2, 0, w))
    
    for i in range(n):
        if y[i] == 0 and x[i] == 0:
            continue
        else:
            mask[i, y1[i]:y2[i], x1[i]:x2[i]] = 0.
    # mask[:][y1: y2][x1: x2] = 0.                
    mask = torch.from_numpy(mask).cuda()  
    mask = mask.unsqueeze(1).repeat(1, 3, 1, 1) 
    img *= mask
    return img

def cutbox(size, center, dxy):
    W = size[2]
    H = size[3]
    center = center.cpu().numpy()
    rate = random.uniform(0.1, 0.8)
    dxy = (rate * dxy).cpu().numpy()

    dxy = dxy.astype(int)
    cx = center[:, 0].astype(int)
    cy = center[:, 1].astype(int)
    
    bbx1 = np.clip(cx - dxy[:, 0], 0, W)
    bby1 = np.clip(cy - dxy[:, 1], 0, H)
    bbx2 = np.clip(cx + dxy[:, 0], 0, W)
    bby2 = np.clip(cy + dxy[:, 1], 0, H)
    
    return bbx1, bby1, bbx2, bby2

def rand_crop(size, alpha, dxy):
    W = size[2]
    H = size[3]
    
    rate = random.uniform(0.1, 0.8)
    dxy = (rate * dxy).cpu().numpy()

    dxy = dxy.astype(int)
    # cx = center[:, 0].astype(int)
    # cy = center[:, 1].astype(int)

    # uniform
    cx = np.random.randint(W, size=size[0])
    cy = np.random.randint(H, size=size[0])
    
    bbx1 = np.clip(cx - dxy[:, 0], 0, W)
    bby1 = np.clip(cy - dxy[:, 1], 0, H)
    bbx2 = np.clip(cx + dxy[:, 0], 0, W)
    bby2 = np.clip(cy + dxy[:, 1], 0, H)
    
    return bbx1, bby1, bbx2, bby2


def cutbox_prob(size, center, dxy, prob):
    W = size[2]
    H = size[3]
    center = center.cpu().numpy()
    
    dxy = dxy.cpu().numpy()

    dxy = dxy.astype(int)
    cx = center[:, 0].astype(int)
    cy = center[:, 1].astype(int)

    bbx1 = np.clip(cx - dxy[:, 0], 0, W)
    bby1 = np.clip(cy - dxy[:, 1], 0, H)
    bbx2 = np.clip(cx + dxy[:, 0], 0, W)
    bby2 = np.clip(cy + dxy[:, 1], 0, H)

    small_sizes = []
    new_bbx1_list = []
    new_bbx2_list = []
    new_bby1_list = []
    new_bby2_list = []
    for i in range(size[0]):
        left_p = bbx1[i]
        right_p = bbx2[i]
        up_p = bby1[i]
        down_p = bby2[i]
        area = (bbx2[i] - bbx1[i]) * (bby2[i] - bby1[i])
        small_area = area * prob
        small_size = np.sqrt(small_area).astype(int)
        small_sizes.append([small_size, small_size])
        
        right_p = right_p + 1 if (right_p == left_p) else right_p
        down_p = down_p + 1 if (down_p == up_p) else down_p
        sm_cx = np.random.randint(left_p, right_p)
        sm_cy = np.random.randint(up_p, down_p)
        
        new_bbx1 = np.clip(sm_cx - small_size // 2, left_p, right_p)
        new_bbx2 = np.clip(new_bbx1 + small_size, left_p, right_p)
        new_bbx1 = np.clip(new_bbx2 - small_size, left_p, right_p)
        new_bby1 = np.clip(sm_cy - small_size // 2, up_p, down_p)
        new_bby2 = np.clip(new_bby1 + small_size, up_p, down_p)
        new_bby1 = np.clip(new_bby2 - small_size, up_p, down_p)

        new_bbx1_list.append(new_bbx1)
        new_bbx2_list.append(new_bbx2)
        new_bby1_list.append(new_bby1)
        new_bby2_list.append(new_bby2)
    
    return new_bbx1_list, new_bby1_list, new_bbx2_list, new_bby2_list, small_sizes

def getbox(size, dxy):
    W = size[2]
    H = size[3]
    dxy = dxy.cpu().numpy()

    dxy = dxy.astype(int)
    cx = np.random.randint(0, W, dxy.shape[0])
    cy = np.random.randint(0, H, dxy.shape[0])
    
    bbx1 = np.clip(cx - dxy[:, 0] // 2, 0, W)
    bbx2 = np.clip(bbx1 + dxy[:, 0], 0, W)
    bbx1 = np.clip(bbx2 - dxy[:, 0], 0, W)
    bby1 = np.clip(cy - dxy[:, 1] // 2, 0, H)
    bby2 = np.clip(bby1 + dxy[:, 1], 0, H)
    bby1 = np.clip(bby2 - dxy[:, 1], 0, H)
    
    return bbx1, bby1, bbx2, bby2

def getbox_cnt(size, center, dxy):
    W = size[2]
    H = size[3]
    dxy = dxy.cpu().numpy()

    dxy = dxy.astype(int)
    cx = center[:, 0].cpu().numpy().astype(int)
    cy = center[:, 1].cpu().numpy().astype(int)
    
    bbx1 = np.clip(cx - dxy[:, 0] // 2, 0, W)
    bbx2 = np.clip(bbx1 + dxy[:, 0], 0, W)
    bbx1 = np.clip(bbx2 - dxy[:, 0], 0, W)
    bby1 = np.clip(cy - dxy[:, 1] // 2, 0, H)
    bby2 = np.clip(bby1 + dxy[:, 1], 0, H)
    bby1 = np.clip(bby2 - dxy[:, 1], 0, H)
    
    return bbx1, bby1, bbx2, bby2

def getbox_new(size, dxy):
    W = size[2]
    H = size[3]
    
    cx = np.random.randint(0, W, size[0])
    cy = np.random.randint(0, H, size[0])

    bbx1 = np.clip(cx - dxy[0] // 2, 0, W)
    bbx2 = np.clip(bbx1 + dxy[0], 0, W)
    bbx1 = np.clip(bbx2 - dxy[0], 0, W)
    bby1 = np.clip(cy - dxy[1] // 2, 0, H)
    bby2 = np.clip(bby1 + dxy[1], 0, H)
    bby1 = np.clip(bby2 - dxy[1], 0, H)
    
    return bbx1, bby1, bbx2, bby2

def rand_flip(image):
    if random.random() > 0.5:
        image = torch.flip(image, (2,))
    return image

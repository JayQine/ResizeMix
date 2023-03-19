import random
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as function
import torch.distributed as dist
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import kornia
from kornia.augmentation import AugmentationBase2D
import kornia.augmentation as K

# import kornia.augmentation.functional as F
from typing import Callable, Tuple, Union, List, Optional, Dict, cast

mean = torch.Tensor([0.4914, 0.4822, 0.4465]).cuda()
std = torch.Tensor([0.2023, 0.1994, 0.2010]).cuda()

def RGB2Gray(data):
    image = data
    R = image[:, 0]
    G = image[:, 1]
    B = image[:, 2]
    output = (0.299*R + 0.587*G + 0.114*B).unsqueeze(1)
    output = output.repeat(1, 3, 1, 1)
    return output


class GaussianBlur(nn.Module):

    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = np.array([[1, 1, 1],
                            [1, 5, 1],
                            [1, 1, 1]]) / 13.0
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel.cuda(), requires_grad=False)
 
    def forward(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x1 = function.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = function.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = function.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

cifar_mean = torch.Tensor([[[[0.4914]], [[0.4822]], [[0.4465]]]]).cuda()
cifar_std = torch.Tensor([[[[0.2023]], [[0.1994]], [[0.2010]]]]).cuda()

class MyColor(AugmentationBase2D):
   def __init__(self, value: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
   return_transform: bool = False, same_on_batch: bool = False, p: float=1.) -> None:
      super(MyColor, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch)
      self.value = value

   def generate_parameters(self, input_shape: torch.Size):
      values = torch.ones(input_shape) * self.value
      return dict(values=values)

   def compute_transformation(self, input, params):
      B, _, H, W = input.shape
      # compute transformation
      transform = [K.Denormalize(mean, std)]
      transform.append(K.Normalize(mean, std))

      return transform

   def apply_transform(self, input, params):
      image1 = input
      aug = K.RandomGrayscale(p=1.0)
      image2 = aug(input)
      lam = torch.Tensor(params["values"]).cuda()
      output = image1 * lam + image2 * (1 - lam)
      return output

class MyInvert(AugmentationBase2D):
   def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float=1.) -> None:
      super(MyInvert, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch)

   def generate_parameters(self, input_shape: torch.Size):
      return dict()

   def apply_transform(self, input, params):
      # apply transformation and return
      image1 = input
      image2 = torch.ones_like(input).cuda()
      output = image2 - image1
      return output

class MyAutoContrast(AugmentationBase2D):
   def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float=1.) -> None:
      super(MyAutoContrast, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch)

   def generate_parameters(self, input_shape: torch.Size):
      return dict()

   def apply_transform(self, input, params):
      # apply transformation and return
      histogram = torch.histc(input*256, bins=256, min=0, max=255)
      lut = []
      for layer in range(0, len(histogram), 256):
          h = histogram[layer : layer + 256]
          # find lowest/highest samples after preprocessing
          for lo in range(256):
              if h[lo]:
                  break
          for hi in range(255, -1, -1):
              if h[hi]:
                  break
          if hi <= lo:
              # don't bother
              lut.extend(list(range(256)))
          else:
              scale = 255.0 / (hi - lo)
              offset = -lo * scale
              for ix in range(256):
                  ix = int(ix * scale + offset)
                  if ix < 0:
                      ix = 0
                  elif ix > 255:
                      ix = 255
                  lut.append(ix)
      lut = lut + lut + lut
      img = (input*256).type(torch.long)  
      table = torch.Tensor(lut).cuda()
      output = table[img]
      return output / 255.

def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def rotate_cuda(image, val):
    assert -30 <= val <= 30
    transform = K.RandomRotation(val, same_on_batch=True, p=1.)
    return transform(image)

def invert_cuda(image, prob):
    transform = MyInvert(same_on_batch=True, p=1.)
    return transform(image)

def equalize_cuda(image, prob):
    transform = K.RandomEqualize(same_on_batch=True, p=1.)
    return transform(image)

def flip_cuda(image, prob):
    transform = K.RandomHorizontalFlip(prob)
    return transform(image)

def solarize_cuda(image, val):
    assert 0 < val < 1
    transform = K.RandomSolarize(thresholds=val, same_on_batch=True, p=1.)
    return transform(image)

def posterize_cuda(image, val):
    assert 4 <= val <= 8
    val = int(val)
    transform = K.RandomPosterize(bits=val, same_on_batch=True, p=1.)
    return transform(image) 

def contrast_cuda(image, val):
    transform = K.ColorJitter(0, (val, val), 0, 0, same_on_batch=True, p=1.)
    return transform(image)

def contrast(image, v):
    h = image.size()[-2]
    w = image.size()[-1]
    level = v 
    img1 = image
    img2 = RGB2Gray(image)
    mean0 = torch.mean(img2[:,0], 1)
    mean1 = torch.mean(mean0, 1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    img3 = mean1.repeat(1, 3, h, w)
    output = img1 * level + img3 * (1 - level)
    return output

def brightness_cuda(image, val):
    transform = K.ColorJitter(val, 0, 0, 0, same_on_batch=True, p=1.)
    return transform(image)

def brightness(image, v):
    level = v 
    img1 = torch.zeros_like(image)
    output = image * level + img1 * (1 - level)
    return output

def color_cuda(image, val):  # [0.1,1.9]
    transform = MyColor(val, same_on_batch=True)
    return transform(image)

def autocontrast_cuda(image, val):
    transform = MyAutoContrast(p=val)
    return transform(image)

def sharpness_cuda(image, val):
    image = image.cpu()
    transform = K.RandomSharpness(val, same_on_batch=True, p=1.)
    return transform(image).cuda()

def sharpness(image, v):
    level = v 
    img1 = image
    img2 = GaussianBlur().forward(image).squeeze(0)
    output = img1 * level + img2 * (1 - level) 
    return output

def shearX_cuda(image, val):
    transform = K.RandomAffine(degrees=0, shear=(val, val), same_on_batch=True, p=1)
    return transform(image)

def shearY_cuda(image, val):
    if random.random() > 0.5:
        val = -val
    transform = K.RandomAffine(degrees=0, shear=(0, 0, val, val), same_on_batch=True, p=1)
    return transform(image)

def translateX_cuda(image, val):
    transform = K.RandomAffine(degrees=0, translate=(val, val), same_on_batch=True, p=1)
    return transform(image)

def translateY_cuda(image, val):
    transform = K.RandomAffine(degrees=0, translate=(0, val), same_on_batch=True, p=1)
    return transform(image)

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  
    l = [
        (Identity, 0., 1.0),
        (shearX_cuda, 0., 0.5),  # 0
        (shearY_cuda, 0., 0.5),  # 1
        (translateX_cuda, 0., 0.5),  # 2
        (translateY_cuda, 0., 0.5),  # 3
        (rotate_cuda, 0, 45.),  # 4
        (autocontrast_cuda, 0, 1.),  # 5
        (invert_cuda, 0, 1),  # 6
        # (equalize_cuda, 0, 1),  # 7
        (solarize_cuda, 0, 1.),  # 8
        (posterize_cuda, 4, 8),  # 9
        (contrast, 0.1, 1.9),  # 10
        (color_cuda, 0.1, 1.9),  # 11
        (sharpness, 0.1, 1.9),  # 12
        (brightness, 0.1, 1.9),  # 13
        # (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]

    # l = [
    #     (AutoContrast, 0, 1),
    #     (Equalize, 0, 1),
    #     (Invert, 0, 1),
    #     (Rotate, 0, 30),
    #     (Posterize, 0, 4),
    #     (Solarize, 0, 256),
    #     (SolarizeAdd, 0, 110),
    #     (Color, 0.1, 1.9),
    #     (Contrast, 0.1, 1.9),
    #     (Brightness, 0.1, 1.9),
    #     (Sharpness, 0.1, 1.9),
    #     (ShearX, 0., 0.3),
    #     (ShearY, 0., 0.3),
    #     (CutoutAbs, 0, 40),
    #     (TranslateXabs, 0., 100),
    #     (TranslateYabs, 0., 100),
    # ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.                
        mask = torch.from_numpy(mask)       
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment_cuda(nn.Module):
    def __init__(self, n, m):
        super(RandAugment_cuda, self).__init__()
        self.n = n
        self.m = m
        self.augment_list = augment_list()
        self.normalize = K.Normalize(mean, std)
        self.denormalize = K.Denormalize(mean, std)

    def forward(self, x):
        ops = random.choices(self.augment_list, k=self.n)
        x = self.denormalize(x)
        for transform, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            x = transform(x, val)
        x = self.normalize(x)
        return x
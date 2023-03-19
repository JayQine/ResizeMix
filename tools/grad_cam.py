import numpy as np
import torch
import time
from torch import nn
from torch.autograd import Function 
import torch.nn.functional as F
from torchvision import models

class FeatureExtractor(nn.Module):

    def __init__(self, model, target_layers):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.features = []

    def hook_grad(self, module, grad_input, grad_output):
        self.gradients.append(grad_output)

    def hook_feature(self, module, input, output):
        self.features.append(output.data)

    def forward(self):
        self.features = []
        self.gradients = []
        module = self.model.module._modules.get(self.target_layers[0])
        for name in self.target_layers[1:]:
            module = module._modules.get(name)
        self.hook_f = module.register_forward_hook(self.hook_feature)
        self.hook_g = module.register_backward_hook(self.hook_grad)
        
    def get_feature(self):
        return self.features

    def get_gradients(self):
        return self.gradients

    def set_hook(self):
        self.hook_f.remove()
        self.hook_g.remove()

class GradCam(nn.Module):
    def __init__(self, model, target_layer):
        super(GradCam, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.model.train(True)
        self.extractor = FeatureExtractor(self.model, self.target_layer)

    def forward(self, input):
        self.extractor.forward()
        output = self.model(input)
        return output
    
    def get_mask(self, input, output):
        features = self.extractor.get_feature()
        index = torch.argmax(output, dim=1)  #h
        one_hot = torch.zeros((output.size())).cuda()
        for i in range(len(index)):
            one_hot[i][index[i]] = 1
        one_hot = torch.sum(one_hot * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True) 
        grads = self.extractor.get_gradients()
        self.extractor.set_hook()
        
        grad_val = grads[0][0]
        for i in range(1, len(grads)):
            grad_val = torch.cat((grad_val, grads[i][0].to(grad_val.device)), 0)
        target = features[0]
        for i in range(1, len(features)):
            target = torch.cat((target, features[i].to(target.device)), 0)
        weights = grad_val.mean(axis=(2, 3), keepdims=True)  # [0, :]
        weights = weights.to(target.device)

        cam = (weights * target).mean(dim=1)
        cam = F.relu(cam, inplace=True)   
        max_val = torch.max(cam, dim = 1, keepdims=True)[0]
        max_val = torch.max(max_val, dim = 2, keepdims=True)[0]
        min_val = torch.zeros_like(max_val)
        diff = max_val - min_val
        cam = (cam - min_val) / diff
        s2 = time.time()
        cam[torch.isnan(cam)] = 0  #h
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=input.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(1)
        return cam


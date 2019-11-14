import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from .quantization import *

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class VGG(nn.Module):
    def __init__(self, depth=16, init_weights=True, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.feature = self.make_layers(cfg, True)
        num_classes = 10
        # self.classifier = nn.Linear(cfg[-1], num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            quan_Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            quan_Linear(512, 512),
            nn.ReLU(True),
            quan_Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()
            
    def make_layers(self, cfg, batch_norm=False):
        
        is_firstlayer = True
        
        layers = []
        in_channels = 3
        layers += [nn.BatchNorm2d(3)]
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if is_firstlayer:
                    conv2d = quan_Conv2d(
                        in_channels, v, kernel_size=3, padding=1, bias=False)
                    is_firstlayer = False
                else:
                    conv2d = quan_Conv2d(
                        in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
def vgg_test(num_classes=10):
    
    model = VGG()
    pretrained_dict = torch.load('/home/elliot/Documents/CVPR_2020/BFA_defense/BFA_defense/save/2019-11-13/test_vgg/cifar_vgg_pretrain.pt')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    
    return model
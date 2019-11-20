import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

# class _bin_func(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input):
        
#         # with torch.no_grad():
#         ctx.mu = input.abs().mean()
#         # ctx.mu = mu
#         # output = input.clone()
            
#         output = input.clone().zero_()
#         output[input.ge(0)] = ctx.mu
#         output[input.lt(0)] = -ctx.mu

#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return grad_input
    
# w_bin = _bin_func.apply


# class quan_Conv2d(nn.Conv2d):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  bias=True):
#         super(quan_Conv2d, self).__init__(in_channels,
#                                           out_channels,
#                                           kernel_size,
#                                           stride=stride,
#                                           padding=padding,
#                                           dilation=dilation,
#                                           groups=groups,
#                                           bias=bias)

#     def forward(self, input):
#         try:
#             with torch.no_grad():
#                 weight_change = (self.bin_weight - w_bin(self.weight)).abs()
#                 self.bin_weight_change = weight_change[weight_change.ge(0.5)].sum().item()
#                 self.bin_weight_change_ratio = self.bin_weight_change 
#                 print(self.bin_weight_change_ratio)
#         except:
#             pass
        
#         # with torch.no_grad():
#         #     mu = input.abs().mean()
        
#         self.bin_weight = w_bin(self.weight)
#         output = F.conv2d(input, self.bin_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
#         return output 

# class quan_Linear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(quan_Linear, self).__init__(in_features, out_features, bias=bias)
        
#     def forward(self, input):
#         try:
#             with torch.no_grad():
#                 self.bin_weight_change = (self.bin_weight - w_bin(self.weight)).abs().sum().item()
#                 self.bin_weight_change_ratio = self.bin_weight_change / self.weight.numel()
#         except:
#             pass

#         # with torch.no_grad():
#         #     mu = input.abs().mean()
                    
#         self.bin_weight = w_bin(self.weight)
#         output = F.linear(input, self.bin_weight, self.bias)

#         return output


class _bin_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mu):
        
        ctx.mu = mu 
        output = input.clone().zero_()
        output[input.ge(0)] = 1
        output[input.lt(0)] = -1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.mu
        return grad_input, None
    
w_bin = _bin_func.apply


class quan_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

    def forward(self, input):
        if self.training:
            try:
                with torch.no_grad():
                    weight_change = (self.bin_weight - w_bin(self.weight,1)).abs()
                    self.bin_weight_change = weight_change.sum().item()
                    self.bin_weight_change_ratio = self.bin_weight_change / self.weight.numel()
                    # print(self.bin_weight_change, self.bin_weight_change_ratio)
            except:
                pass
        
        with torch.no_grad():
            mu = self.weight.abs().mean()
        
        self.bin_weight = w_bin(self.weight, mu)
        output = F.conv2d(input, self.bin_weight* mu, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 

class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)
        
    def forward(self, input):
        if self.training:
            try:
                with torch.no_grad():
                    weight_change = (self.bin_weight - w_bin(self.weight,1)).abs()
                    self.bin_weight_change = weight_change.sum().item()
                    self.bin_weight_change_ratio = self.bin_weight_change  / self.weight.numel()
                    # print(self.bin_weight_change, self.bin_weight_change_ratio)
            except:
                pass

        with torch.no_grad():
            mu = self.weight.abs().mean()
                    
        self.bin_weight = w_bin(self.weight, mu)
        output = F.linear(input, self.bin_weight * mu, self.bias)

        return output







class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = quan_Conv2d(inplanes,
                                planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = quan_Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
    def __init__(self, block, depth, num_classes):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
        super(CifarResNet, self).__init__()

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth -
                2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(
            depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = quan_Conv2d(3,
                                    16,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = quan_Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion,
                                     stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def resnet20_bin(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model

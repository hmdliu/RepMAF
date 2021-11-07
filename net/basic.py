
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBn(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, groups=1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        ))
        self.add_module('bn', nn.BatchNorm2d(
            num_features=out_channels
        ))

class ConvBnActPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, groups=1, act='relu', pool=False):
        super().__init__()
        act_dict = {
            'idt': nn.Identity,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'hardswish': nn.Hardswish,
        }
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        ))
        self.add_module('bn', nn.BatchNorm2d(
            num_features=out_channels
        ))
        self.add_module('act', act_dict[act]())
        self.add_module('pool', nn.MaxPool2d(2) if pool else nn.Identity())

class SE_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, in_feats // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // r, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x

class RepVGG_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', use_att=False):
        super().__init__()
        act_dict = {
            'idt': nn.Identity,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'hardswish': nn.Hardswish,
        }
        self.br_3x3 = ConvBn(in_channels, out_channels, kernel_size=3, padding=1)
        self.br_1x1 = ConvBn(in_channels, out_channels, kernel_size=1, padding=0)
        self.br_idt = nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        self.att = SE_Block(out_channels) if use_att else nn.Identity()
        self.act = act_dict[act]()
        print('=> RepVGG Block: in_ch=%3d, out_ch=%3d, act=%s' % (in_channels, out_channels, act))
    
    def forward(self, x):
        x = self.br_3x3(x) + self.br_1x1(x) + (self.br_idt(x) if self.br_idt is not None else 0)
        return self.act(self.att(x))

class IRB_Block(nn.Module):
    def __init__(self, in_feats, out_feats, pool=True, expand_ratio=6):
        super().__init__()
        mid_feats = round(in_feats * expand_ratio)
        self.pool = pool
        self.irb_unit = nn.Sequential(
            # point-wise conv
            nn.Conv2d(in_feats, mid_feats, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_feats),
            nn.ReLU6(inplace=True),
            # depth-wise conv
            nn.Conv2d(mid_feats, mid_feats, kernel_size=3, stride=1, padding=1, groups=mid_feats, bias=False),
            nn.BatchNorm2d(mid_feats),
            nn.ReLU6(inplace=True),
            # point-wise conv
            nn.Conv2d(mid_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feats),
        )

    def forward(self, x):
        return F.max_pool2d(self.irb_unit(x), 2) if self.pool else self.irb_unit(x)

class SimAM_Block(torch.nn.Module):
    def __init__(self, in_feats, lamb=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.lamb = lamb

    def forward(self, x):
        _, _, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.lamb)) + 0.5
        return x * self.act(y)        
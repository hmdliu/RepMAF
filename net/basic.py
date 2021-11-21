
import random
import numpy as np

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
        self.add_module('bn', nn.BatchNorm2d(out_channels))

class ConvBnActPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, groups=1, act='relu', pool=False):
        super().__init__()
        act_dict = {
            'idt': nn.Identity,
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
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
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        self.add_module('act', act_dict[act]())
        self.add_module('pool', nn.MaxPool2d(2) if pool else nn.Identity())

class RepVGG_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', att='idt', att_kwargs={}):
        super().__init__()
        act_dict = {
            'idt': nn.Identity,
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'hardswish': nn.Hardswish,
        }
        att_dict = {
            'se': SE_Block,
            'sem': SEM_Block,
            'idt': IDT_Block,
            'simam': SimAM_Block
        }
        self.deploy = False
        self.in_channels = in_channels
        self.br_3x3 = ConvBn(in_channels, out_channels, kernel_size=3, padding=1)
        self.br_1x1 = ConvBn(in_channels, out_channels, kernel_size=1, padding=0)
        self.br_idt = nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        self.att = att_dict[att](out_channels, **att_kwargs)
        self.act = act_dict[act]()
        print('  => RepVGG Block: in_ch=%s, out_ch=%s, act=%s' % (in_channels, out_channels, act))
    
    def get_equivalent_kernel_bias(self):
        ker_3x3, bias_3x3 = self._fuse_bn_tensor(self.br_3x3)
        ker_1x1, bias_1x1 = self._fuse_bn_tensor(self.br_1x1)
        ker_idt, bias_idt = self._fuse_bn_tensor(self.br_idt)
        rep_ker = ker_3x3 + self._pad_1x1_to_3x3(ker_1x1) + ker_idt
        rep_bias = bias_3x3 + bias_1x1 + bias_idt
        return rep_ker, rep_bias

    def _pad_1x1_to_3x3(self, ker_1x1):
        return F.pad(ker_1x1, [1, 1, 1, 1]) if ker_1x1 is not None else 0

    # convert bn to conv params
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # assume conv groups = 1
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                kernel_value = np.zeros((self.in_channels, self.in_channels, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % self.in_channels, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    # convert to testing architecture
    def switch_to_deploy(self):
        if self.deploy:
            return
        self.br_rep = nn.Conv2d(
            in_channels=self.br_3x3.conv.in_channels,
            out_channels=self.br_3x3.conv.out_channels,
            kernel_size=self.br_3x3.conv.kernel_size,
            stride=self.br_3x3.conv.stride,
            padding=self.br_3x3.conv.padding,
            dilation=self.br_3x3.conv.dilation,
            groups=self.br_3x3.conv.groups,
            bias=True
        )
        kernel, bias = self.get_equivalent_kernel_bias()
        self.br_rep.weight.data = kernel
        self.br_rep.bias.data = bias
        for p in self.parameters():
            p.detach_()
        self.__delattr__('br_3x3')
        self.__delattr__('br_1x1')
        if hasattr(self, 'br_idt'):
            self.__delattr__('br_idt')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.act(self.att(self.br_rep(x)))
        x = self.br_3x3(x) + self.br_1x1(x) + (self.br_idt(x) if self.br_idt is not None else 0)
        return self.act(self.att(x))

class Fwd_Seq_Block(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_num = len(module_list)
        for i in range(self.module_num):
            self.add_module('%d' % i, module_list[i])

    def forward(self, x, fwd):
        for i in range(self.module_num):
            x = self.__getattr__('%d' % i)(x, fwd)
        return x

class RepTree_Module(nn.Module):
    def __init__(self, in_channels, out_channels, branch=2, shuffle=True, 
                    branch_dropout=0, repvgg_kwargs={}, device='cuda:0'):
        super().__init__()

        self.device = device
        self.branch = branch
        self.shuffle = shuffle
        self.out_channels = out_channels
        self.branch_dropout = branch_dropout
        self.inter_channels = branch * out_channels
        print('=> RepTree Block: branch=%s, inter_ch=%s, shuffle=%s, dropout=%.2f' %
                (branch, self.inter_channels, shuffle, branch_dropout))

        for i in range(1, branch+1):
            self.add_module('br%d' % i, RepVGG_Module(
                in_channels=in_channels,
                out_channels=out_channels,
                **repvgg_kwargs
            ))
        self.merge = nn.Sequential(
            nn.BatchNorm2d(self.inter_channels),
            nn.Conv2d(
                in_channels=self.inter_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=out_channels,
                bias=False
            ) if branch > 1 else nn.Identity(),
            # nn.BatchNorm2d(out_channels),
        )
        self.merge[1].weight.data.fill_(1 / self.branch)

    def forward(self, x, fwd=None):
        b, _, h, w = x.size()
        feats = []
        if fwd is None:
            fwd = [True for i in range(self.branch)]
        for i in range(1, self.branch+1):
            if fwd[i-1]:
                feats.append(self.__getattr__('br%d' % i)(x))
            else:
                feats.append(torch.zeros(
                    size=(b, self.out_channels, h, w),
                    requires_grad=False,
                    device=self.device,
                ))
        feats = torch.cat(tuple(feats), dim=(2 if self.shuffle else 1))
        return self.merge(feats.view(b, self.inter_channels, h, w))

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

class IDT_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

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

class SEM_Block(nn.Module):
    def __init__(self, in_feats, mid_feats=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, mid_feats, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_feats, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x

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

def test_repvgg_deploy():

    # init module & input
    x = torch.randn(2, 3, 4, 4)
    m = RepVGG_Module(3, 3)
    m.eval()  

    # pred with training architecture
    a = m(x)

    # switch to testing architecture
    m.switch_to_deploy()

    # pred with testing architecture
    b = m(x)

    # check equivalence
    print('[pred 1]:', torch.sum(a))
    print('[pred 2]:', torch.sum(b))
    print('[pred diff]:', torch.sum(a) - torch.sum(b))

def test_reptree_deploy():
    x = torch.randn(2, 3, 4, 4)
    m = RepTree_Module(3, 5, branch=2)
    m.eval() 

    print(m)
    for k, v in m.state_dict().items():
        print(k, v.size())

if __name__ == '__main__':
    # test_repvgg_deploy()
    test_reptree_deploy()

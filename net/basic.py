
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

def get_rep_branch(in_channels, out_channels, br_type):
    assert br_type in ('1p3', '1pp')
    if br_type == '1p3':
        return nn.Sequential(
            ConvBn(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ZeroPad2d(padding=1),
            ConvBn(in_channels, out_channels, kernel_size=3, padding=0)
        )
    else:
        return nn.Sequential(
            ConvBn(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ZeroPad2d(padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

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

class DBB_Module(nn.Module):
    def __init__(self, in_channels, out_channels, merge='add', idt_flag=True, act='relu'):
        super().__init__()
        act_dict = {
            'idt': nn.Identity,
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'hardswish': nn.Hardswish,
        }
        self.branch_num = 4
        self.in_channels = in_channels
        self.br1 = ConvBn(in_channels, out_channels, kernel_size=3, padding=1)
        self.br2 = ConvBn(in_channels, out_channels, kernel_size=1, padding=0)
        self.br3 = nn.Sequential(
            ConvBn(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ZeroPad2d(padding=1),
            ConvBn(in_channels, out_channels, kernel_size=3, padding=0)
        )
        self.br4 = nn.Sequential(
            ConvBn(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ZeroPad2d(padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        if idt_flag and (in_channels == out_channels):
            self.br5 = nn.BatchNorm2d(out_channels)
            self.branch_num += 1
        if merge == 'group':
            self.inter_channels = self.branch_num * out_channels
            self.merge = nn.Sequential(
                # nn.BatchNorm2d(self.inter_channels),
                nn.Conv2d(
                    in_channels=self.inter_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    groups=out_channels,
                    bias=False
                ),
                # nn.BatchNorm2d(out_channels),
                act_dict[act]()
            )
            # self.merge[1].weight.data.fill_(1 / self.branch_num)
            self.merge[0].weight.data.fill_(1 / self.branch_num)
        else:
            self.act = act_dict[act]()
        print('=> DBB Block: in_ch=%s, out_ch=%s, act=%s, merge=%s' % \
                (in_channels, out_channels, act, merge))

    def forward(self, x):
        b, _, h, w = x.size()
        feats = []
        for i in range(1, self.branch_num+1):
            feats.append(self.__getattr__('br%d' % i)(x))
        if hasattr(self, 'merge'):
            feats = torch.cat(tuple(feats), dim=2)
            return self.merge(feats.view(b, self.inter_channels, h, w))
        else:
            return self.act(sum(feats))

class RepMSS_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', block='vgg', version=1):
        super().__init__()
        act_dict = {
            'idt': nn.Identity,
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'hardswish': nn.Hardswish,
        }
        self.block = block
        self.version = version
        self.in_channels = in_channels
        # # use_lamb exp
        # self.lamb1 = nn.Parameter(torch.ones(1))
        # self.lamb2 = nn.Parameter(torch.ones(1))
        self.idt_flag = (in_channels == out_channels)
        if self.idt_flag:
            self.br1_idt = nn.BatchNorm2d(out_channels)
            self.br2_idt = nn.BatchNorm2d(out_channels)
        self.br1_1x1 = ConvBn(in_channels, out_channels, kernel_size=1, padding=0)
        self.br2_1x1 = ConvBn(in_channels, out_channels, kernel_size=1, padding=0)
        self.br1_3x3 = ConvBn(in_channels, out_channels, kernel_size=3, padding=1)
        self.br2_3x3 = ConvBn(in_channels, out_channels, kernel_size=3, padding=1)
        if self.block == 'dbb':
            self.br1_1p3 = get_rep_branch(in_channels, out_channels, br_type='1p3')
            self.br2_1p3 = get_rep_branch(in_channels, out_channels, br_type='1p3')
            self.br1_1pp = get_rep_branch(in_channels, out_channels, br_type='1pp')
            self.br2_1pp = get_rep_branch(in_channels, out_channels, br_type='1pp')
        self.act = act_dict[act]()
        print('=> RepMSS Block: in_ch=%s, out_ch=%s, act=%s, block=%s' \
                % (in_channels, out_channels, act, block))

    def forward(self, feats):
        x, y = feats[:2]
        x_feats, y_feats = [], []
        x_feats += [self.br1_1x1(x), self.br1_3x3(x)]
        y_feats += [self.br2_1x1(y), self.br2_3x3(y)]
        if self.idt_flag:
            x_feats.append(self.br1_idt(x))
            y_feats.append(self.br2_idt(y))
        if self.block == 'dbb':
            x_feats += [self.br1_1p3(x), self.br1_1pp(x)]
            y_feats += [self.br2_1p3(y), self.br2_1pp(y)]
        if self.version == 1:
            x, y = self.act(sum(x_feats)), self.act(sum(y_feats))

            # # same-size exp
            # return x+y, x+y, x, y

            # default setting
            p, q = F.max_pool2d(x, kernel_size=2), F.interpolate(y, scale_factor=2)
            return x+q, y+p, x, y

            # # use_lamb exp
            # return x + self.lamb1 * q, y + self.lamb2 * p, x, y
        else:
            x, y = sum(x_feats), sum(y_feats)

            # # same-size exp
            # return x+y, x+y, x, y

            # default setting
            p, q = F.max_pool2d(x, kernel_size=2), F.interpolate(y, scale_factor=2)
            return self.act(x+q), self.act(y+p), self.act(x), self.act(y)
            
            # # use_lamb exp
            # return self.act(x + self.lamb1 * q), self.act(y + self.lamb2 * p), self.act(x), self.act(y)

class Fwd_Seq_Block(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_num = len(module_list)
        for i in range(self.module_num):
            self.add_module('%d' % i, module_list[i])

    def forward(self, x, fwd=None):
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

        # group conv fusion
        self.merge = nn.Sequential(
            nn.BatchNorm2d(self.inter_channels),
            nn.Conv2d(
                in_channels=self.inter_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=out_channels,
                bias=True
            ) if branch > 1 else nn.Identity(),
            nn.BatchNorm2d(
                num_features=out_channels
            ) if branch > 1 else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        if isinstance(self.merge[1], nn.Conv2d):
            self.merge[1].weight.data.fill_(1 / self.branch)
        
        # # sum fusion
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x, fwd=None):
        b, _, h, w = x.size()
        feats = []

        # branch dropout version
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

        # # all branch version
        # for i in range(1, self.branch+1):
        #     feats.append(self.__getattr__('br%d' % i)(x))
        # return self.act(sum(feats))

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

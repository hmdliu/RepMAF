
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # calling from train
    from net.utils import *
except:
    # calling from __main__
    from utils import *

def get_att_block(att_type, planes, **att_kwargs):
    att_dict = {
        'se': SE_Block,
        'idt': IDT_Block,
        'ses': SES_Block,
        'sef': SEF_Block,
    }
    return att_dict[att_type](planes, **att_kwargs)

class RepVGG_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', att='idt', att_kwargs={}):
        super().__init__()
        self.deploy = False
        self.in_channels = in_channels
        self.br_3x3 = ConvBn(in_channels, out_channels, kernel_size=3, padding=1)
        self.br_1x1 = ConvBn(in_channels, out_channels, kernel_size=1, padding=0)
        self.br_idt = nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        self.att = get_att_block(att, out_channels, **att_kwargs)
        self.act = get_act_func(act)
        print('=> RepVGG Block: in_ch=%s, out_ch=%s, act=%s' % (in_channels, out_channels, act))
    
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

class BiRepVGG_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', fwd_size=(16, 8),
                    att='idt', att_kwargs={}):
        super().__init__()
        self.fwd_size = fwd_size
        self.in_channels = in_channels
        self.br1 = RepVGG_Module(
            in_channels=in_channels, 
            out_channels=out_channels,
            act='idt',
            att=att,
            att_kwargs=att_kwargs
        )
        self.br2 = RepVGG_Module(
            in_channels=in_channels, 
            out_channels=out_channels,
            act='idt',
            att=att,
            att_kwargs=att_kwargs
        )
        self.act = get_act_func(act)
        print('=> BiRepVGG Block: in_ch=%s, out_ch=%s, act=%s' % (in_channels, out_channels, act))

    def forward(self, x):
        if self.fwd_size == (16, 8):
            x1, x2 = x, F.max_pool2d(x, kernel_size=2)
            out = self.br1(x1) + F.interpolate(self.br1(x2), scale_factor=2)
        else:
            out = self.br1(x) + self.br2(x)
        return self.act(out)

class RepMAF_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', fwd_size=(16, 8), att_kwargs={}):
        super().__init__()
        self.fwd_size = fwd_size
        self.br1 = RepVGG_Module(in_channels, out_channels, att='idt', act='idt')
        self.br2 = RepVGG_Module(in_channels, out_channels, att='idt', act='idt')
        self.fusion = SEF_Block(out_channels, **att_kwargs)
        self.act = get_act_func(act)
        print('=> RepMAF Block: in_ch=%s, out_ch=%s, act=%s' % (in_channels, out_channels, act))

    def forward(self, x):
        y = F.max_pool2d(x, kernel_size=2) if self.fwd_size == (16, 8) else x.clone()
        return self.act(self.fusion(self.br1(x), self.br2(y)))

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

class SES_Block(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x

class SEF_Block(nn.Module):
    def __init__(self, in_feats, version=1, r=16):
        super().__init__()
        mid_feats = in_feats // r
        self.version = version
        self.squeeze = self.get_squeeze(in_feats, mid_feats)
        if version == 1:
            self.excite1 = self.get_excite(mid_feats, in_feats)
            self.excite2 = self.get_excite(mid_feats, in_feats)
        else:
            self.excite1 = self.get_excite(2 * mid_feats, in_feats)
            self.excite2 = self.get_excite(2 * mid_feats, in_feats)

    @staticmethod
    def get_squeeze(in_feats, out_feats, bn=True):
        return nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=1),
            nn.BatchNorm2d(out_feats) if bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def get_excite(in_feats, out_feats):
        return nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, _, _, _ = x.size()
        if x.size() != y.size():
            y = F.interpolate(y, scale_factor=2)
        if self.version == 1:
            guide_feats = self.squeeze(F.adaptive_avg_pool2d(x+y, 1))
        else:
            x_ave, y_ave = F.adaptive_avg_pool2d(x, 1), F.adaptive_avg_pool2d(y, 1)
            f_ave = torch.cat((x_ave, y_ave), dim=1).view(2 * b, -1, 1, 1)
            guide_feats = self.squeeze(f_ave).view(b, -1, 1, 1)
        return self.excite1(guide_feats) * x + self.excite2(guide_feats) * y


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # calling from train
    from net.basic import *
    from net.utils import *
except:
    # calling from __main__
    from basic import *
    from utils import *

class VGG_CIFAR(nn.Module):
    def __init__(self, blocks_seq=[1, 3, 4, 1], planes_seq=[64, 128, 256, 512], 
                    act='relu', num_classes=10, **kwargs):
        super().__init__()

        self.act = act
        self.blocks_seq = blocks_seq
        self.planes_seq = planes_seq
        self.planes = planes_seq[0]
        assert len(self.blocks_seq) == len(self.planes_seq)

        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act=act,
            pool=kwargs.get('in_pool', True)
        )
        for i in range(len(self.planes_seq)):
            self.add_module('block%d' % (i+1), self._make_block(
                planes=self.planes_seq[i],
                num_blocks=self.blocks_seq[i]
            ))
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.planes_seq[-1], num_classes)
        )
    
    def _make_block(self, planes, num_blocks):
        assert (planes > 0) and (num_blocks > 0)
        blocks = []
        for i in range(num_blocks):
            blocks.append(ConvBnAct(
                in_channels=self.planes,
                out_channels=planes,
                kernel_size=3,
                padding=1,
                act=self.act
            ))
            self.planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block0(x)
        for i in range(len(self.planes_seq)):
            out = self.__getattr__('block%d' % (i+1))(out)
        return self.fc(out)

class RepVGG_CIFAR(nn.Module):
    def __init__(self, act='relu', att='idt', att_kwargs={}, num_classes=10,
                    blocks_seq=[1, 3, 4, 1], planes_seq=[64, 128, 256, 512], **kwargs):
        super().__init__()

        self.act = act
        self.att = att
        self.att_kwargs = att_kwargs
        self.blocks_seq = blocks_seq
        self.planes_seq = planes_seq
        self.planes = planes_seq[0]
        assert len(self.blocks_seq) == len(self.planes_seq)

        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act=act,
            pool=kwargs.get('in_pool', True)
        )
        for i in range(len(self.planes_seq)):
            self.add_module('block%d' % (i+1), self._make_block(
                planes=self.planes_seq[i],
                num_blocks=self.blocks_seq[i]
            ))
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.planes_seq[-1], num_classes)
        )
    
    def _make_block(self, planes, num_blocks):
        assert (planes > 0) and (num_blocks > 0)
        blocks = []
        for i in range(num_blocks):
            blocks.append(RepVGG_Module(
                in_channels=self.planes,
                out_channels=planes,
                act=self.act,
                att=self.att,
                att_kwargs=self.att_kwargs
            ))
            self.planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block0(x)
        for i in range(len(self.planes_seq)):
            out = self.__getattr__('block%d' % (i+1))(out)
        return self.fc(out)

class BiRepVGG_CIFAR(nn.Module):
    def __init__(self, act='relu', att='idt', att_kwargs={}, num_classes=10, fwd_size=(16, 8),
                    blocks_seq=[1, 3, 4, 1], planes_seq=[64, 128, 256, 512], **kwargs):
        super().__init__()

        self.act = act
        self.att = att
        self.fwd_size = fwd_size
        self.att_kwargs = att_kwargs
        self.blocks_seq = blocks_seq
        self.planes_seq = planes_seq
        self.planes = planes_seq[0]
        assert len(self.blocks_seq) == len(self.planes_seq)

        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act=act,
            pool=kwargs.get('in_pool', True)
        )
        for i in range(len(self.planes_seq)):
            self.add_module('block%d' % (i+1), self._make_block(
                planes=self.planes_seq[i],
                num_blocks=self.blocks_seq[i]
            ))
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.planes_seq[-1], num_classes)
        )
    
    def _make_block(self, planes, num_blocks):
        assert (planes > 0) and (num_blocks > 0)
        blocks = []
        for i in range(num_blocks):
            blocks.append(BiRepVGG_Module(
                in_channels=self.planes,
                out_channels=planes,
                act=self.act,
                att=self.att,
                att_kwargs=self.att_kwargs,
                fwd_size=self.fwd_size
            ))
            self.planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block0(x)
        if self.fwd_size == (8, 8):
            out = F.max_pool2d(out, kernel_size=2)
        for i in range(len(self.planes_seq)):
            out = self.__getattr__('block%d' % (i+1))(out)
        return self.fc(out)

class RepMAF_CIFAR(nn.Module):
    def __init__(self, blocks_seq=[1, 3, 4, 1], planes_seq=[64, 128, 256, 512], 
                    fwd_size=(16, 8), act='relu', num_classes=10, **kwargs):
        super().__init__()

        self.act = act
        self.kwargs = kwargs
        self.fwd_size = fwd_size
        self.blocks_seq = blocks_seq
        self.planes_seq = planes_seq
        self.planes = planes_seq[0]
        assert len(self.blocks_seq) == len(self.planes_seq)

        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act=act,
            pool=kwargs.get('in_pool', True)
        )
        for i in range(len(self.planes_seq)):
            self.add_module('block%d' % (i+1), self._make_block(
                planes=self.planes_seq[i],
                num_blocks=self.blocks_seq[i]
            ))
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.planes_seq[-1], num_classes)
        )
    
    def _make_block(self, planes, num_blocks):
        assert (planes > 0) and (num_blocks > 0)
        blocks = []
        for i in range(num_blocks):
            blocks.append(RepMAF_Module(
                in_channels=self.planes,
                out_channels=planes,
                act=self.act,
                att_kwargs=self.kwargs.get('att_kwargs', {})
            ))
            self.planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block0(x)
        if self.fwd_size == (8, 8):
            out = F.max_pool2d(self.block0(x), kernel_size=2)
        for i in range(len(self.planes_seq)):
            out = self.__getattr__('block%d' % (i+1))(out)
        return self.fc(out)

def get_model(model_name, model_config):
    avail_models = {
        'vgg': VGG_CIFAR,
        'repmaf': RepMAF_CIFAR,
        'repvgg': RepVGG_CIFAR,
        'birepvgg': BiRepVGG_CIFAR
    }
    return avail_models[model_name](**model_config)
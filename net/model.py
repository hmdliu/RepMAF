
from numpy.core.shape_base import block
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # calling from train
    from net.basic import *
    from net.utils import get_fwd_branch
except:
    # calling from __main__
    from basic import *
    from utils import get_fwd_branch

# RepVGG CIFAR module
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

        # Simple CONV-BN-ACT-POOL layer
        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act=act,
            pool=kwargs.get('in_pool', True)
        )
        # Creating RepVGG blocks
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
    
    # Separate repVGG block creation
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

    # Forward inference
    def forward(self, x):
        out = self.block0(x)
        for i in range(len(self.planes_seq)):
            out = self.__getattr__('block%d' % (i+1))(out)
        return self.fc(out)
        # return F.log_softmax(self.fc(out), dim=1)

# RepTree CIFAR module
class RepTree_CIFAR(nn.Module):
    def __init__(self, blocks_seq=[1, 3, 4, 1], planes_seq=[64, 128, 256, 512], repvgg_kwargs={},
                    branch=2, branch_dropout=0, shuffle=True, num_classes=10, device='cuda:0', **kwargs):
        super().__init__()

        self.device = device
        self.branch = branch
        self.shuffle = shuffle
        self.blocks_seq = blocks_seq
        self.planes_seq = planes_seq
        self.planes = planes_seq[0]
        self.repvgg_kwargs = repvgg_kwargs
        self.branch_dropout = branch_dropout
        assert len(self.blocks_seq) == len(self.planes_seq)

        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act='relu', # repvgg_kwargs.get('act', 'relu')
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
            blocks.append(RepTree_Module(
                in_channels=self.planes,
                out_channels=planes,
                branch=self.branch,
                shuffle=self.shuffle,
                branch_dropout=self.branch_dropout,
                repvgg_kwargs=self.repvgg_kwargs,
                device=self.device
            ))
            self.planes = planes
        return Fwd_Seq_Block(blocks)

    def forward(self, x):
        out = self.block0(x)
        fwd = get_fwd_branch(self.branch, self.branch_dropout, self.training)
        for i in range(len(self.planes_seq)):
            # out = self.__getattr__('block%d' % (i+1))(out)
            out = self.__getattr__('block%d' % (i+1))(out, fwd)
        return self.fc(out)

# RepDBB CIFAR module
class RepDBB_CIFAR(nn.Module):
    def __init__(self, act='relu', merge='add', idt_flag=True, num_classes=10,
                    blocks_seq=[1, 3, 4, 1], planes_seq=[64, 128, 256, 512], **kwargs):
        super().__init__()

        self.act = act
        self.merge = merge
        self.idt_flag = idt_flag
        self.blocks_seq = blocks_seq
        self.planes_seq = planes_seq
        self.planes = planes_seq[0]
        assert len(self.blocks_seq) == len(self.planes_seq)

        # Simple CONV-BN-ACT-POOL layer
        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act=act,
            pool=kwargs.get('in_pool', True)
        )
        # Creating RepDBB blocks
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
    
    # Separate RepDBB block creation
    def _make_block(self, planes, num_blocks):
        assert (planes > 0) and (num_blocks > 0)
        blocks = []
        for i in range(num_blocks):
            blocks.append(DBB_Module(
                in_channels=self.planes,
                out_channels=planes,
                merge=self.merge,
                idt_flag=self.idt_flag,
                act=self.act
            ))
            self.planes = planes
        return nn.Sequential(*blocks)

    # Forward inference
    def forward(self, x):
        out = self.block0(x)
        for i in range(len(self.planes_seq)):
            out = self.__getattr__('block%d' % (i+1))(out)
        return self.fc(out)

# RepMSS CIFAR module
class RepMSS_CIFAR(nn.Module):
    def __init__(self, blocks_seq=[1, 3, 4, 1], planes_seq=[64, 128, 256, 512], act='relu', 
                    block='vgg', version=1, out_concat=True, num_classes=10, **kwargs):
        super().__init__()

        self.act = act
        self.block = block
        self.version = version
        self.out_concat = out_concat
        self.blocks_seq = blocks_seq
        self.planes_seq = planes_seq
        self.planes = planes_seq[0]
        assert len(self.blocks_seq) == len(self.planes_seq)

        # Simple CONV-BN-ACT-POOL layer
        self.block0 = ConvBnActPool(
            in_channels=3,
            out_channels=self.planes,
            kernel_size=5,
            padding=2,
            act=act,
            pool=kwargs.get('in_pool', True)
        )
        # Creating RepMSS blocks
        for i in range(len(self.planes_seq)):
            self.add_module('block%d' % (i+1), self._make_block(
                planes=self.planes_seq[i],
                num_blocks=self.blocks_seq[i]
            ))
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear((out_concat + 1) * self.planes_seq[-1], num_classes)
        )
    
    # Separate RepMSS block creation
    def _make_block(self, planes, num_blocks):
        assert (planes > 0) and (num_blocks > 0)
        blocks = []
        for i in range(num_blocks):
            blocks.append(RepMSS_Module(
                in_channels=self.planes,
                out_channels=planes,
                act=self.act,
                block=self.block,
                version=self.version
            ))
            self.planes = planes
        return nn.Sequential(*blocks)

    # Forward inference
    def forward(self, x):
        x1 = self.block0(x)
        # # code for same-size exp
        # x1 = F.max_pool2d(x1, kernel_size=2)
        # y1 = x1.clone()
        y1 = F.max_pool2d(x1, kernel_size=2)
        for i in range(len(self.planes_seq)):
            x1, y1, x0, y0 = self.__getattr__('block%d' % (i+1))((x1, y1))
        if self.out_concat:
            x0 = torch.cat((x0, F.interpolate(y0, scale_factor=2)), dim=1)
            # # code for same-size exp
            # x0 = torch.cat((x0, y0), dim=1)
        return self.fc(x0)

class RepVGG_Simple(nn.Sequential):
    def __init__(self, in_channels=64, out_channels=512, num_classes=10):
        super().__init__()
        self.add_module('block0', ConvBnActPool(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=5,
            padding=2,
            act='relu',
            pool=True
        ))
        self.add_module('block1', RepVGG_Module(in_channels, out_channels))
        self.add_module('fc', nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes)
        ))

class RepVGG_Tree(nn.Sequential):
    def __init__(self, in_channels=64, out_channels=256, branch=2, num_classes=10):
        super().__init__()
        self.add_module('block0', ConvBnActPool(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=5,
            padding=2,
            act='relu',
            pool=True
        ))
        self.add_module('block1', RepTree_Module(in_channels, out_channels, branch))
        self.add_module('fc', nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes)
        ))

# Test Version
class HMDNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            IRB_Block(16, 48, pool=True),
            IRB_Block(48, 128, pool=False),
            SimAM_Block(128),
            nn.AdaptiveMaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.5),
        )
        
        self.fc_unit = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
            nn.BatchNorm1d(num_classes),
        )

    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.conv_unit(x).view(b, -1)
        x = self.fc_unit(x)
        return x
        # return F.log_softmax(x, dim=1)

def get_model(model_name, model_config): # Return model based on name and config
    avail_models = {
        'hmd': HMDNet,
        'repmss_cifar': RepMSS_CIFAR,
        'repdbb_cifar': RepDBB_CIFAR,
        'repvgg_cifar': RepVGG_CIFAR,
        'reptree_cifar': RepTree_CIFAR,
        'simple': RepVGG_Simple,
        'tree': RepVGG_Tree,
    }
    return avail_models[model_name](**model_config)

if __name__ == '__main__': # Default if called from model main
    test_model = get_model(
        model_name='repvgg_cifar',
        model_config={
            'act': 'relu',
            'att': 'sem',
            'att_kwargs': {},
            'num_classes': 10,
        }
    )
    print(test_model)
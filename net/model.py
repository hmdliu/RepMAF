
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # calling from train
    from net.basic import *
except:
    # calling from __main__
    from basic import *

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
        # return F.log_softmax(self.fc(out), dim=1)

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

def get_model(model_name, model_config):
    avail_models = {
        'hmd': HMDNet,
        'repvgg_cifar': RepVGG_CIFAR
    }
    return avail_models[model_name](**model_config)

if __name__ == '__main__':
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
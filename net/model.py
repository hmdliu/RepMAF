
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return F.log_softmax(x, dim=1)

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

def get_model(model_name, num_classes):
    avail_models = {
        'hmd': HMDNet,
    }
    return avail_models[model_name](num_classes=num_classes)
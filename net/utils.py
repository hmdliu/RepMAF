
import torch.nn as nn

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

class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, groups=1, act='relu'):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True
        ))
        self.add_module('act', get_act_func(act))

class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, groups=1, act='relu'):
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
        self.add_module('act', get_act_func(act))

class ConvBnActPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, groups=1, act='relu', pool=False):
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
        self.add_module('act', get_act_func(act))
        self.add_module('pool', nn.MaxPool2d(2) if pool else nn.Identity())

def get_act_func(act_type='relu', **act_kwargs):
    act_dict = {
        'idt': nn.Identity,
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'hardswish': nn.Hardswish,
    }
    return act_dict[act_type](**act_kwargs)

class Poly_LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, iters_per_epoch,
                    base_poly=0.9, thresh=1e-6, quiet=False):
        self.epoch = -1
        self.quiet = quiet
        self.thresh = thresh
        self.curr_lr = base_lr
        self.base_lr = base_lr
        self.base_poly = base_poly
        self.iters_per_epoch = iters_per_epoch
        self.total_iters = num_epochs * iters_per_epoch

    def __call__(self, optimizer, writer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.curr_lr > self.thresh:
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), self.base_poly)
            self._adjust_learning_rate(optimizer, lr)
            self.curr_lr = lr
        if (not self.quiet) and (epoch != self.epoch):
            self.epoch = epoch
            writer.add_scalar('lr', self.curr_lr, epoch)
            print('=> Epoch %i, lr = %.4f, best_pred = %.2f%s' % (epoch, self.curr_lr, best_pred, '%'))

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

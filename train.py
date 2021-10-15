
import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from net.utils import *
from net.model import get_model
from net.dataset import get_dataset

PATH = os.getcwd()
CONFIG = {
    'seed': 0,
    'aug': False,
    'model': 'hmd',
    'dataset': 'cifar10',
    'num_classes': 10,
    'use_cuda': True,
    'batch_size': 64,
    'epochs': 30,
    'poly': 0.9,
    'optim': {
        'method': 'adam',
        'lr': 0.001,
    },
    'dump_summary': True,
    'export_bound': 100
}

class Trainer():
    def __init__(self, config):
        self.config = config
        
        # init trainer
        self.best_pred = 0.0
        self.start_time = time.time()
        self.path = os.path.join(PATH, 'results', str(int(self.start_time)))
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        torch.manual_seed(self.config['seed'])
        print('[Path]: %s' % self.path)
        print('[Config]: %s\n' % self.config)

        # init dataloader
        trainset, valset = get_dataset(self.config['dataset'])
        self.train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=2
        )
        self.val_loader = DataLoader(
            dataset=valset,
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=2
        )
        self.log_interval = len(self.train_loader) // 4
    
        # init model
        self.use_cuda = (self.config['use_cuda'] and torch.cuda.is_available())
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.model = get_model(
            model_name=self.config['model'],
            num_classes=self.config['num_classes'],
        ).to(self.device)
        print(self.model)

        # init optim
        if self.config['optim']['method'] == 'sgd':
            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.config['optim']['lr'],
                momentum=self.config['optim']['momentum']
            )
        elif self.config['optim']['method'] == 'adam':
            self.optimizer = optim.Adam(
                params=self.model.parameters(),
                lr=self.config['optim']['lr']
            )
        else:
            raise ValueError('Invalid optim: %s.' % self.config['optim']['method'])
        self.scheduler = Poly_LR_Scheduler(
            base_lr=self.config['optim']['lr'],
            base_poly=self.config['poly'],
            num_epochs=self.config['epochs'],
            iters_per_epoch=len(self.train_loader),
        )
        print(self.optimizer)

        # init summary writer
        if self.config['dump_summary']:
            self.writer = SummaryWriter(self.path)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.curr_epoch = epoch
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data, target = data.to(self.device), target.to(self.device)
            self.scheduler(self.optimizer, self.writer, batch_idx, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Step: [%5d/%5d (%3.0f%s)], Loss: %.6f' %(
                    batch_idx * self.config['batch_size'], len(self.train_loader.dataset),
                    100.0 * batch_idx / len(self.train_loader), '%', loss.item()
                ))
        self.check_accuracy()

    def eval(self, data_loader):
        self.model.eval()
        loss, correct = 0, 0
        for data, target in data_loader:
            if self.use_cuda:
                data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        accuracy = 100. * correct / len(data_loader.dataset)
        loss /= len(data_loader.dataset)
        return accuracy, loss
    
    def check_accuracy(self):
        train_acc, train_loss = self.eval(self.train_loader)
        val_acc, val_loss = self.eval(self.val_loader)
        print('Train Accuracy: %.2f%s\tTrain Loss: %.6f' % (train_acc, '%', train_loss))
        print('Val Accuracy:   %.2f%s\tVal Loss:   %.6f' % (val_acc, '%', val_loss))
        if val_acc > self.best_pred:
            self.best_pred = val_acc
        if val_acc > self.config['export_bound']:
            self.export_weights(val_acc)
        if self.config['dump_summary']:
            self.writer.add_scalar('accuracy/train', train_acc, self.curr_epoch)
            self.writer.add_scalar('loss/train', train_loss, self.curr_epoch)
            self.writer.add_scalar('accuracy/val', val_acc, self.curr_epoch)
            self.writer.add_scalar('loss/val', val_loss, self.curr_epoch)
        print('==================================================')
    
    def export_weights(self, accuracy):
        curr_info = '%02d_%.2f' % (self.curr_epoch, accuracy)
        weight_path = os.path.join(self.path, 'weights_%s.pth' % curr_info)
        torch.save(self.model.state_dict(), weight_path)
        print('[Weights]: [%s] state dict exported.' % (curr_info))

    def train(self):
        for epoch in range(1, self.config['epochs']+1):
            print('\n============ train epoch [%2d/%2d] =================' % (epoch, self.config['epochs']))
            self.train_one_epoch(epoch)
        runtime = int(time.time() - self.start_time) // 60
        print('\n[Time]: %.2fmins\n[Best Pred]: %.2f%s' % (runtime, self.best_pred, '%'))

if __name__ == '__main__':
    trainer = Trainer(config=CONFIG)
    trainer.train()
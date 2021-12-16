
import time
import torch
from torchsummary import summary
from torch.utils.data import DataLoader

from net.dataset import get_dataset
from net.model import get_model
from config import get_config

# rep modules to be evaluated
MODEL_IDS = ['vgg-idt', 'vgg-se', 'repvgg-idt', 'repvgg-se', 'birepvgg-idt3',
                'birepvgg-se3', 'repmaf-maf3', 'repmaf-maf4']

class BatchIPS():
    def __init__(self, model_ids, total_iter=20):
        self.model_ids = model_ids
        self.total_iter = total_iter

        # init dataloader
        _, valset = get_dataset(
            dataset='cifar10',
            aug=False
        )
        self.val_loader = DataLoader(
            dataset=valset,
            batch_size=64, 
            shuffle=False, 
            num_workers=2
        )

        # init device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_ips()

    def batch_ips(self):
        # loop through all the modules
        for i in self.model_ids:
            # evaluate 3 times and pick the median
            for j in range(3):
                print('[model_id]:', i)
                print('[iter_num]:', j)
                self.config = get_config(i)
                # # debugging mode
                # self.config['batch_size'] = 2
                # self.config['use_cuda'] = False
                self.model = get_model(
                    model_name=self.config['model'],
                    model_config=self.config['model_config']
                ).to(self.device)
                # summary(self.model, (3, 32, 32))
                self.inference()
                print('[ori]: %d imgs/s' % self.ips)
                self.model, self.rep_flag = convert_model(self.model)
                if self.rep_flag:
                    self.model = self.model.to(self.device)
                    # summary(self.model, (3, 32, 32))
                    self.inference()
                    print('[rep]: %d imgs/s\n' % self.ips)

    # test inference speed
    def inference(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        start = time.time()
        for i in range(self.total_iter):
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.data.max(1, keepdim=True)[1]
        end = time.time()
        self.ips = self.total_iter * len(self.val_loader.dataset) / (end - start)

# convert a re-parameterizable model to inference-time mode
def convert_model(model):
    rep_flag = False
    model = model.cpu()
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
            rep_flag = True
    return model, rep_flag

if __name__ == "__main__":
    infer = BatchIPS(MODEL_IDS)
    infer.batch_ips()

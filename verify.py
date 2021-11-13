import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from copy import deepcopy

from config import get_config
from net.dataset import get_dataset
from net.model import get_model


# Global variables
DEVICE = "cuda"
MODE = sys.argv[1]                  # 1st param, convert 1 or multiple
assert MODE in ["one", "multi"]
if MODE == "one":
    PATHS = [sys.argv[2]]           
else:
    PATH = sys.argv[2]
    PATHS = [PATH + "/" + _ for _ in os.listdir(PATH) if _.endswith(".pth")]

EXP_ID = sys.argv[3]
CONFIG = get_config(EXP_ID)


def convert_repvgg_model(model, to_path=None):
    assert isinstance(model, torch.nn.Module)
    
    tmp_model = deepcopy(model)
    for module in tmp_model.modules():
        if hasattr(module, "br_3x3"):
            module.switch_to_deploy()

    if to_path:
        torch.save(tmp_model.state_dict(), to_path)
    
    return tmp_model


def validate(train_model, inf_model, val_loader):
    train_loss, train_correct = 0, 0
    inf_loss, inf_correct = 0, 0

    train_model.eval()
    inf_model.eval()

    for data, target in val_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        train_output = train_model(data)
        train_loss += F.cross_entropy(train_output, target, reduction='sum').item()
        pred = train_output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        inf_output = inf_model(data)
        inf_loss += F.cross_entropy(inf_output, target, reduction='sum').item()
        pred = inf_output.data.max(1, keepdim=True)[1]
        inf_correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_accuracy = 100. * train_correct / len(val_loader.dataset)
    train_loss /= len(val_loader.dataset)
    print('Training Model Val Accuracy :   %.2f%s\tVal Loss:   %.6f' % (train_accuracy, '%', train_loss))

    inf_accuracy = 100. * inf_correct / len(val_loader.dataset)
    inf_loss /= len(val_loader.dataset)
    print('Inference Model Val Accuracy :   %.2f%s\tVal Loss:   %.6f' % (inf_accuracy, '%', inf_loss))


if __name__ == "__main__":
    # print("folder path  :", PATH)
    print("weight files :", PATHS)
    print("id for config:", EXP_ID)
    print("config info  :", CONFIG)
    print()
    
    # Get dataset
    trainset, valset = get_dataset(CONFIG["dataset"])
    val_loader = DataLoader(
                dataset=valset,
                batch_size=CONFIG['batch_size'], 
                shuffle=False, 
                num_workers=2
    )

    # 4. convert the model
    for i in range(len(PATHS)):
        print("==============================================================")
        print("%d. Load weights:" % i, PATHS[i])

        # load train model
        train_model = get_model(
            model_name = "repvgg_cifar",
            model_config = CONFIG["model_config"]
        ).to(DEVICE)
        train_model.load_state_dict(torch.load(PATHS[i]))
        
        # get inf model
        inf_model = convert_repvgg_model(train_model)
        
        # test on dataset
        print()
        validate(train_model, inf_model, val_loader)
        print("==============================================================")
        print("\n\n")

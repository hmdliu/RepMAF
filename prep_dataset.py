
from net.dataset import get_dataset

datasets = get_dataset(dataset='cifar10', aug=False)
print('Dataset downloaded.')
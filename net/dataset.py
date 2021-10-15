from torchvision import datasets, transforms

def get_dataset(dataset):
    avail_datasets = {
        'cifar10': get_cifar10
    }
    assert dataset in avail_datasets
    return avail_datasets[dataset]()

def get_cifar10():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        # credits: https://github.com/kuangliu/pytorch-cifar/issues/19
    ])
    trainset = datasets.CIFAR10(
        root='./', 
        train=True, 
        download=True, 
        transform=data_transform
    )
    testset = datasets.CIFAR10(
        root='./', 
        train=False, 
        download=True, 
        transform=data_transform
    )
    return trainset, testset
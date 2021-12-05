from torchvision import datasets, transforms
import torch

def get_dataset(dataset, aug):
    avail_datasets = {
        'cifar10': get_cifar10,
        'cifar100': get_cifar100
    }
    assert dataset in avail_datasets
    return avail_datasets[dataset](aug)

# credits: https://github.com/kuangliu/pytorch-cifar
def get_cifar10(aug=False):
    ori_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ### list of augmentations
    aug_transform_list = []
    aug_transform_list.append(transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]))
    aug_transform_list.append(transforms.Compose([
        transforms.ColorJitter(brightness=5, contrast=5, saturation=5, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]))
    aug_transform_list.append(transforms.Compose([
        transforms.RandomAffine(degrees =15, translate=(0.1,0.1), shear=2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]))
    aug_transform_list.append(transforms.Compose([
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]))
    aug_transform_list.append(transforms.Compose([
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]))
    ### concat trainset
    trainset = []
    for augments in aug_transform_list:
        trainset.append(datasets.CIFAR10(
        root='./', 
        train=True, 
        download=True, 
        transform=(augments if aug else ori_transform)))
    trainset = torch.utils.data.ConcatDataset(trainset)

    '''
    trainset = datasets.CIFAR10(
        root='./', 
        train=True, 
        download=True, 
        transform=(aug_transform if aug else ori_transform)
    )
    '''
    testset = datasets.CIFAR10(
        root='./', 
        train=False, 
        download=True, 
        transform=ori_transform
    )
    return trainset, testset

# credits: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
def get_cifar100(aug=False):
    ori_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    trainset = datasets.CIFAR100(
        root='./', 
        train=True, 
        download=True, 
        transform=(aug_transform if aug else ori_transform)
    )
    testset = datasets.CIFAR100(
        root='./', 
        train=False, 
        download=True, 
        transform=ori_transform
    )
    return trainset, testset
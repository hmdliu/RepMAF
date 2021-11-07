
from copy import deepcopy

TEMPLATE = {
    'seed': 0,
    'aug': False,
    'model': 'repvgg_cifar',
    'model_config': {
        'act': 'relu',
        'use_att': False,
        'num_classes': 10,
    },
    'dataset': 'cifar10',
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

def get_config(info):
    config = deepcopy(TEMPLATE)
    date, setting = tuple(info.split('_'))
    if date == '1106':
        config['model_config']['use_att'] = (setting[0] == 't')
    else:
        raise ValueError('Invalid Date: %s' % date)
    return config

if __name__ == '__main__':
    from sys import argv
    test_config = get_config(argv[1])
    print(test_config)
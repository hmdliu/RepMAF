
from copy import deepcopy

TEMPLATE = {
    'seed': 0,
    'aug': True,
    'model': 'repvgg',
    'model_config': {
        'act': 'relu',
        'att': 'idt',
        'att_kwargs': {},
        'num_classes': 10,
        'blocks_seq': [1, 3, 4, 1],
        'planes_seq': [64, 128, 256, 512]
    },
    'dataset': 'cifar10',
    'use_cuda': True,
    'batch_size': 64,
    'epochs': 150,
    'poly': 0.9,
    'optim': {
        'method': 'sgd',
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'dump_summary': True,
    'export_best': False
}

def get_config(info):
    # To be completed
    config = deepcopy(TEMPLATE)
    date, setting = tuple(info.split('_'))

    att_config = {
        # repvgg
        'i': {'act': 'relu', 'att': 'idt'},
        'e': {'act': 'relu', 'att': 'se'},
        't': {'act': 'relu', 'att': 'ses'},
        # birepvgg
        'a': {'act': 'relu', 'att': 'idt', 'fwd_size': (16, 8)},
        'b': {'act': 'relu', 'att': 'idt', 'fwd_size': (16, 16)},
        'c': {'act': 'relu', 'att': 'se', 'fwd_size': (16, 8)},
        'd': {'act': 'relu', 'att': 'se', 'fwd_size': (16, 16)},
        
    }
    config['epochs'] = 150

    config['model'] = 'birepvgg'
    # config['model'] = 'repmaf'
    # config['model'] = 'repvgg'
    # config['model'] = 'vgg'
    config['model_config'] = att_config[setting[0]]

    return config

if __name__ == '__main__':
    from sys import argv
    test_config = get_config(argv[1])
    print(test_config)
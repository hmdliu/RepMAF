
from copy import deepcopy

TEMPLATE = {
    'seed': 0,
    'aug': True,
    'model': 'repvgg_cifar',
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
    'epochs': 30,
    'poly': 0.9,
    'optim': {
        'method': 'sgd',
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'dump_summary': True,
    'export_bound': 100
}

EPOCH_NUM_DICT = {
    'a':  20, 'b':  30, 'c':  50, 'd':  80,
    'e': 100, 'f': 120, 'g': 150, 'h': 200
}

ATT_DICT = {
    'i': {'att': 'idt', 'att_kwargs': {}},
    'p': {'att': 'se', 'att_kwargs': {}},
    'q': {'att': 'sem', 'att_kwargs': {}},
    'r': {'att': 'simam', 'att_kwargs': {}},
}

ACT_DICT = {
    'i': 'idt',
    'r': 'relu',
    's': 'silu',
    't': 'tanh',
    'g': 'gelu',
    'h': 'hardswish',
}

def get_config(info):
    config = deepcopy(TEMPLATE)
    if len(info.split('_')) == 2:
        date, setting = tuple(info.split('_'))
    else:
        date, setting, dataset = tuple(info.split('_'))
        if dataset == '100':
            config['dataset'] = 'cifar100'
            config['model_config']['num_classes'] = 100
    '''
    if date == '1106':
        config['model_config']['use_att'] = (setting[0] == 't')
    elif date == '1107':
        seq_dict = {
            'x': {'blocks_seq': [1, 3, 5], 'planes_seq': [64, 128, 256]},
            'y': {'blocks_seq': [1, 3, 4, 1], 'planes_seq': [64, 128, 256, 512]},
            'z': {'blocks_seq': [1, 3, 4, 1], 'planes_seq': [64, 128, 256, 1024]},
            '1': {'blocks_seq': [1, 3, 2, 1], 'planes_seq': [64, 128, 256, 512]},
            '2': {'blocks_seq': [1, 3, 4, 1], 'planes_seq': [64, 128, 256, 512]},
            '3': {'blocks_seq': [1, 3, 6, 1], 'planes_seq': [64, 128, 256, 512]},
            '4': {'blocks_seq': [1, 3, 8, 1], 'planes_seq': [64, 128, 256, 512]},
            '5': {'blocks_seq': [1, 3, 4, 2], 'planes_seq': [64, 128, 256, 512]},
            '6': {'blocks_seq': [1, 3, 4, 4], 'planes_seq': [64, 128, 256, 512]},
        }
        config['optim'] = get_optim_dict(setting[:2])
        config['epochs'] = EPOCH_NUM_DICT[setting[2]]
        config['model_config']['act'] = ACT_DICT[setting[3]]
        config['model_config'].update(seq_dict[setting[4]])
        config['model_config'].update(ATT_DICT[setting[5]])
    elif date == '1108':
        # default settings
        config['epochs'] = 30
        config['optim'] = get_optim_dict('sj')
        config['model_config']['act'] = 'relu'
        config['model_config']['blocks_seq'] = [1, 3, 4, 1]
        config['model_config']['planes_seq'] = [64, 128, 256, 512]
        # experiments
        config['model_config'].update(ATT_DICT[setting[0]])
        config['model_config']['in_pool'] = (setting[1] == 't')
    else:
        raise ValueError('Invalid Date: %s' % date)
    '''

    seq_dict = {
        'x': {'blocks_seq': [1, 3, 5], 'planes_seq': [64, 128, 256]},
        'y': {'blocks_seq': [1, 3, 4, 1], 'planes_seq': [64, 128, 256, 512]},
        'z': {'blocks_seq': [1, 3, 4, 1], 'planes_seq': [64, 128, 256, 1024]},
        '1': {'blocks_seq': [1, 3, 2, 1], 'planes_seq': [64, 128, 256, 512]},
        '2': {'blocks_seq': [1, 3, 4, 1], 'planes_seq': [64, 128, 256, 512]},
        '3': {'blocks_seq': [1, 3, 6, 1], 'planes_seq': [64, 128, 256, 512]},
        '4': {'blocks_seq': [1, 3, 8, 1], 'planes_seq': [64, 128, 256, 512]},
        '5': {'blocks_seq': [1, 3, 4, 2], 'planes_seq': [64, 128, 256, 512]},
        '6': {'blocks_seq': [1, 3, 4, 4], 'planes_seq': [64, 128, 256, 512]},
    }

    config['optim'] = get_optim_dict(setting[:2])           # optimizer and lr
    config['epochs'] = EPOCH_NUM_DICT[setting[2]]           # number of epochs
    config['model_config']['act'] = ACT_DICT[setting[3]]    # activation function
    config['model_config'].update(seq_dict[setting[4]])     # repvgg blocks and width
    config['model_config'].update(ATT_DICT[setting[5]])     # attention module
    return config

# two-letter str: first optimizer, second lr
def get_optim_dict(info):
    assert len(info) == 2
    lr_dict = {
        'a': 0.001, 'b': 0.003, 'c': 0.005, 'd': 0.007,
        'e': 0.010, 'f': 0.020, 'g': 0.030, 'h': 0.050,
        'i': 0.070, 'j': 0.100, 'k': 0.200, 'l': 0.300,
    }
    optim_dict = {}
    if info[0] == 's':
        optim_dict['method'] = 'sgd'
        optim_dict['momentum'] = 0.9
    else:
        optim_dict['method'] = 'adam'
    optim_dict['lr'] = lr_dict[info[1]]
    return optim_dict

if __name__ == '__main__':
    from sys import argv
    test_config = get_config(argv[1])
    print(test_config)
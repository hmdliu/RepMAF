
from copy import deepcopy

# default setting
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

# MAF block setting
MAF_IDS = {
    'maf1': {'act': 'relu', 'fwd_size': (8, 8), 'att': 'maf', 'att_kwargs': {'version': 1}},
    'maf2': {'act': 'relu', 'fwd_size': (8, 8), 'att': 'maf', 'att_kwargs': {'version': 2}},
    'maf3': {'act': 'relu', 'fwd_size': (16, 8), 'att': 'maf', 'att_kwargs': {'version': 1}},
    'maf4': {'act': 'relu', 'fwd_size': (16, 8), 'att': 'maf', 'att_kwargs': {'version': 2}},
    'maf5': {'act': 'relu', 'fwd_size': (16, 16), 'att': 'maf', 'att_kwargs': {'version': 1}},
    'maf6': {'act': 'relu', 'fwd_size': (16, 16), 'att': 'maf', 'att_kwargs': {'version': 2}}
}
# feature map size
FWD_SIZE = {'1': (8, 8), '2': (16, 8), '3': (16, 16)}

# generate model & training config
def get_config(info):

    # parse exp_id to a config dict
    def parse_id(exp_id):
        if exp_id.isalpha():
            return {'act': 'relu', 'att': exp_id}
        elif exp_model == 'repmaf':
            return MAF_IDS[exp_id]
        elif exp_model == 'birepvgg':
            return {'act': 'relu', 'att': exp_id[:-1], 'fwd_size': FWD_SIZE[exp_id[-1]]}
        else:
            raise ValueError('Invalid ID: %s.' % exp_id)

    # init config & exp setting
    config = deepcopy(TEMPLATE)
    assert info.count('-') > 0
    exp_model, exp_id = tuple(info.split('-')[:2])

    # modify config dict as needed
    assert exp_model in ('vgg', 'repvgg', 'birepvgg', 'repmaf')
    config['model_config'] = parse_id(exp_id)
    config['model'] = exp_model
    # # uncomment to disable data augmentation
    # config['aug'] = False

    return config

if __name__ == '__main__':
    from sys import argv
    test_config = get_config(argv[1])
    print(test_config)
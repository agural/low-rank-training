'''
This file contains configuration settings for the neural network.
'''

import json
from copy import deepcopy
import numpy as np


class ConfigNN():
    def __str__(self):
        ss = []
        for k,v in self.__dict__.items():
            ss.append('%s: %s'%(k,str(v)))
        return '\n'.join(ss)

    def set(self, attrs, value):
        for attr in attrs:
            self.__dict__[attr] = value

    def ser(self):
        d = {k:v for k,v in self.__dict__.items() if k[:2] != '__'}
        s = json.dumps(d)
        return s
    
    def des(self, s):
        d = json.loads(s)
        for k,v in d.items():
            setattr(self, k, v)
    
    def validate_attr(self):
        valid_attr = ['name', 'description', 'seed', 'epochs', 'lr',
            'train', 'validate', 'train_1ep_fp', 'dataset',
            'add_drift', 'pt_init', 'qbits', 'use_bn', 'norm_b',
            'upc1', 'upc2', 'upc3', 'upc4', 'upd1', 'upd2',
            'train_b', 'train_W', 'quant_regen'
        ]
        for attr in self.__dict__.keys():
            if attr[:2] == '__' or attr in valid_attr:
                continue
            print('WARNING: invalid ConfigNN attribute found in %s: %s'%(self.name, attr))

conf_base = ConfigNN()


### Primary Settings ###

# Every exeriment should have a name.
conf_base.name = 'lrt-base'

# Every experiment should provide a description that possibly references another experiment.
conf_base.description = 'Base configuration.'

# Random seed for reproducibility (None for non-deterministic).
conf_base.seed = 0

# Number of passes through dataset. For online training, should set to 1.
conf_base.epochs = 1

# Whether to run with online training or only inference.
conf_base.train = True

# Whether to validate the runs (faster if no validation).
conf_base.validate = True

# Whether to start with an epoch of floating point training.
conf_base.train_1ep_fp = False

# There are several datasets.
# Normal/standard datasets: MNIST, SVHN, CIFAR10.
# Augmented datasets:
#   MNISTATR: Similar to MNIST, but for the other MNISTA* datasets.
#   MNISTAON: MNIST augmented to 500k samples with elastic transforms.
#   MNISTADS: MNISTAON with input distribution shifts every 10k images.
#   MNISTAON2k: 2k-length version of MNISTAON for testing (also 10k, 100k).
conf_base.dataset = 'MNISTAON2k'

# Adds noise to weights after each sample. There are two versions (version, param):
#   version = 'analog': Adds Gaussian noise such that over 1e6 samples, the total
#       standard deviation of added noise is equal to param. Mathematically,
#       Gaussian N(mu = 0, sigma = param / sqrt(1e6)) is added at each step.
#   version = 'digital': Adds bit-wise flips such that over 1e6 samples, approximately
#       param fraction of binary bits have been flipped.
# Set to `None` for no drift.
conf_base.add_drift = None

# Initializes from the PyTorch model file provided.
# Set to `None` for no initialization.
conf_base.pt_init = None


### Detailed Settings ###

# Quantization settings.
conf_base.qbits = {
    'w':-8,
    'b':-16,
    'a':+8,
    'g':-8,
    'wmax':1.0,
    'bmax':8.0,
    'amax':2.0,
    'gmax':1.0,
}

# Use batchnorm after conv2/4.
conf_base.use_bn = True

# Base learning rate.
conf_base.lr = 1e-2

# Weight update settings.
conf_base.norm_b = True

upConv = ('SKS', {
    'lr':1e-2, 'rank':4, 'zerovar':False, 'kappa_th':100,
    'pseudo_batch':10, 'lr_pb_pow':0.5,
    'norm_uv':'post', 'lr_pb_pow_post':0.0,
    'min_density':1e-2, 'discount':1.0,
    'rho_samp':1.0, 'lookahead':False})
upFC = ('SKS', {
    'lr':1e-2, 'rank':4, 'zerovar':False, 'kappa_th':100,
    'pseudo_batch':100, 'lr_pb_pow':0.5,
    'norm_uv':'post', 'lr_pb_pow_post':0.0,
    'min_density':1e-2, 'discount':1.0,
    'rho_samp':1.0, 'lookahead':False})

upConvStandard = ('Standard', {'lr':1e-2, 'norm_uv':'post', 'count_version':2})
upFCStandard = upConvStandard

convs = ['upc1', 'upc2', 'upc3', 'upc4']
fcs = ['upd1', 'upd2']

conf_base.set(convs, upConv)
conf_base.set(fcs, upFC)

conf_base.train_b = True
conf_base.train_W = True
conf_base.quant_regen = True

# Reference to config.
configs = {
    conf_base.name:conf_base,
}

# Another baseline model.
conf = deepcopy(conf_base)
conf.name = 'lrt-000'
conf.description = 'lrt-base with small online dataset.'
conf.dataset = 'MNISTAON10k'
configs[conf.name] = conf


### Hyperparameter Selection ###

# Baseline and standard weight update learning rate.
i = 0
for lr in 1e-5 * 2**np.arange(0, 21, 2):
    conf = deepcopy(configs['lrt-000'])
    conf.name = 'lrt-001-%03d'%i
    conf.description = 'lrt-000 with base update LR %f.'%(lr)
    conf.validate = False
    conf.lr = lr
    conf.norm_b = False
    stand = deepcopy(upConvStandard)
    stand[1]['lr'] = lr
    stand[1]['norm_uv'] = None
    for l in convs + fcs:
        conf.__dict__[l] = stand
    configs[conf.name] = conf

    conf = deepcopy(configs['lrt-000'])
    conf.name = 'lrt-005-%03d'%i
    conf.description = 'lrt-000 with base update LR %f and post-norming.'%(lr)
    conf.validate = False
    conf.lr = lr
    stand = deepcopy(upConvStandard)
    stand[1]['lr'] = lr
    for l in convs + fcs:
        conf.__dict__[l] = stand
    configs[conf.name] = conf
    i += 1

# Learning rate / batch size experiments for SKS.
i = 0
for pbc, pbd in [(1, 10), (3, 30), (10, 100), (30, 300), (100, 1000)]:
    for lr in 1e-3 * 2**np.arange(0, 15, 2):
        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-002-%03d'%i
        conf.description = 'lrt-000 with LR %f; Pseudobatch Conv %d, FC %d.'%(lr, pbc, pbd)
        conf.validate = False
        conf.norm_b = False
        for l in convs:
            conf.__dict__[l][1]['lr'] = lr
            conf.__dict__[l][1]['norm_uv'] = None
            conf.__dict__[l][1]['pseudo_batch'] = pbc
        for l in fcs:
            conf.__dict__[l][1]['lr'] = lr
            conf.__dict__[l][1]['norm_uv'] = None
            conf.__dict__[l][1]['pseudo_batch'] = pbd
        configs[conf.name] = conf

        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-006-%03d'%i
        conf.description = 'lrt-000 with LR %f; Pseudobatch Conv %d, FC %d and post-norming.'%(lr, pbc, pbd)
        conf.validate = False
        for l in convs:
            conf.__dict__[l][1]['lr'] = lr
            conf.__dict__[l][1]['pseudo_batch'] = pbc
        for l in fcs:
            conf.__dict__[l][1]['lr'] = lr
            conf.__dict__[l][1]['pseudo_batch'] = pbd
        configs[conf.name] = conf
        i += 1


### Ablations ###

# Ablations - bitwidths / ranks.
i = 0
for w_bl in [3, 4, 5, 6, 7, 8, 1, 2]:
    for rank in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-010-%03d'%i
        conf.description = 'lrt-000 with %d/16/8/8 w/b/a/g and rank %d.'%(w_bl, rank)
        conf.dataset = 'MNISTAON2k'
        conf.validate = False
        conf.qbits['w'] = -w_bl
        conf.norm_b = False
        for l in convs + fcs:
            conf.__dict__[l][1]['rank'] = rank
            conf.__dict__[l][1]['norm_uv'] = None
        configs[conf.name] = conf

        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-015-%03d'%i
        conf.description = 'lrt-000 with %d/16/8/8 w/b/a/g and rank %d and max-norming.'%(w_bl, rank)
        conf.dataset = 'MNISTAON2k'
        conf.validate = False
        conf.qbits['w'] = -w_bl
        for l in convs + fcs:
            conf.__dict__[l][1]['rank'] = rank
        configs[conf.name] = conf
        i += 1

# Ablations - OK vs SVD.
for seed in range(5):
    v = 'a'
    for conv_zv in [True, False]:
        for fc_zv in [True, False]:
            conf = deepcopy(configs['lrt-000'])
            conf.name = 'lrt-011%s%d'%(v,seed)
            conf.description = 'lrt-000 with conv_zv %s and fc_zv %s.'%(str(conv_zv), str(fc_zv))
            conf.validate = False
            conf.seed = seed
            for l in convs:
                conf.__dict__[l][1]['zerovar'] = conv_zv
            for l in fcs:
                conf.__dict__[l][1]['zerovar'] = fc_zv
            configs[conf.name] = conf

            conf = deepcopy(configs['lrt-000'])
            conf.name = 'lrt-017%s%d'%(v,seed)
            conf.description = 'lrt-000 with conv_zv %s and fc_zv %s (no-norm).'%(str(conv_zv), str(fc_zv))
            conf.validate = False
            conf.seed = seed
            conf.norm_b = False
            for l in convs:
                conf.__dict__[l][1]['norm_uv'] = None
                conf.__dict__[l][1]['zerovar'] = conv_zv
            for l in fcs:
                conf.__dict__[l][1]['norm_uv'] = None
                conf.__dict__[l][1]['zerovar'] = fc_zv
            configs[conf.name] = conf
            v = chr(ord(v) + 1)

# Ablations - Clear various binary flags.
for seed in range(5):
    v = 'a'
    for clear in ['use_bn', 'train_b', 'train_W']:
        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-012%s%d'%(v, seed)
        conf.description = 'lrt-000 with %s False.'%clear
        conf.validate = False
        conf.seed = seed
        conf.__dict__[clear] = False
        configs[conf.name] = conf

        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-018%s%d'%(v, seed)
        conf.description = 'lrt-000 with %s False (no-norm).'%clear
        conf.validate = False
        conf.seed = seed
        conf.norm_b = False
        for l in convs + fcs:
            conf.__dict__[l][1]['norm_uv'] = None
        conf.__dict__[clear] = False
        configs[conf.name] = conf

        v = chr(ord(v) + 1)
    conf = deepcopy(configs['lrt-000'])
    conf.name = 'lrt-012%s%d'%(v, seed)
    conf.description = 'lrt-000 again.'
    conf.seed = seed
    configs[conf.name] = conf

# Ablations - misc SKS-related ablations.
modifications = [
    ({}, {}),
    ({'kappa_th':10}, {'kappa_th':10}),
    ({'kappa_th':1e8}, {'kappa_th':1e8}),
    ({'min_density':1e-1}, {'min_density':1e-1}),
    ({'min_density':0}, {'min_density':0}),
    ({'discount':0.9}, {'discount':0.9}),
    ({'norm_uv':None}, {'norm_uv':None}),
]
for seed in range(5):
    for i, mod in enumerate(modifications):
        vv = chr(ord('a') + i)
        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-013%s%d'%(vv, seed)
        conf.description = 'lrt-000 with conv mods %s; fc mods %s.'%(str(mod[0]), str(mod[1]))
        conf.validate = False
        conf.seed = seed
        for l in convs:
            for k,v in mod[0].items():
                conf.__dict__[l][1][k] = v
        for l in fcs:
            for k,v in mod[1].items():
                conf.__dict__[l][1][k] = v
        configs[conf.name] = conf

        conf = deepcopy(configs['lrt-000'])
        conf.name = 'lrt-019%s%d'%(vv, seed)
        conf.description = 'lrt-000 with conv mods %s; fc mods %s (no-norm).'%(str(mod[0]), str(mod[1]))
        conf.validate = False
        conf.seed = seed
        conf.norm_b = False
        for l in convs:
            conf.__dict__[l][1]['norm_uv'] = None
            for k,v in mod[0].items():
                conf.__dict__[l][1][k] = v
        for l in fcs:
            conf.__dict__[l][1]['norm_uv'] = None
            for k,v in mod[1].items():
                conf.__dict__[l][1][k] = v
        configs[conf.name] = conf


### Main Experiments ###

# Different training levels for weights for different MNIST Aug datasets.
modifications = [
    {'dataset':'MNISTAON100k'},
    {'dataset':'MNISTAON100k', 'drift':('analog', 10.0)},
    {'dataset':'MNISTAON100k', 'drift':('digital', 10.0)},
    {'dataset':'MNISTADS100k'},
]
for i, mod in enumerate(modifications):
    v = i+20
    conf = deepcopy(conf_base)
    conf.name = 'lrt-%03da'%v
    conf.description = 'lrt-base with PT init, inference only, and %s.'%str(mod)
    conf.pt_init = 'MNISTATR_model.pt'
    conf.dataset = mod['dataset']
    conf.add_drift = mod.get('drift')
    conf.train = False
    configs[conf.name] = conf

    conf = deepcopy(configs['lrt-%03da'%v])
    conf.name = 'lrt-%03db'%v
    conf.description = 'lrt-%03da with bias-only/no-norm.'%v
    conf.train = True
    conf.norm_b = False
    conf.train_W = False
    configs[conf.name] = conf
    
    conf = deepcopy(configs['lrt-%03da'%v])
    conf.name = 'lrt-%03dc'%v
    conf.description = 'lrt-%03da with standard SGD/no-norm.'%v
    conf.train = True
    stand = deepcopy(upConvStandard)
    conf.norm_b = False
    stand[1]['norm_uv'] = None
    conf.set(convs, stand)
    conf.set(fcs, stand)
    configs[conf.name] = conf

    conf = deepcopy(configs['lrt-%03da'%v])
    conf.name = 'lrt-%03dd'%v
    conf.description = 'lrt-%03da with SKS/no-norm.'%v
    conf.train = True
    conf.norm_b = False
    for l in convs + fcs:
        conf.__dict__[l][1]['norm_uv'] = None
    configs[conf.name] = conf
    
    conf = deepcopy(configs['lrt-%03da'%v])
    conf.name = 'lrt-%03de'%v
    conf.description = 'lrt-%03da with SKS/max-norm.'%v
    conf.train = True
    configs[conf.name] = conf

# Different training levels for weights for different image datasets.
modifications = [
    {'dataset':'MNIST',   'drift':('digital', 10.0)},
    {'dataset':'SVHN',    'drift':('digital', 10.0)},
    {'dataset':'CIFAR10', 'drift':('digital', 10.0)},
]
for i, mod in enumerate(modifications):
    v = i+25
    conf = deepcopy(conf_base)
    conf.name = 'lrt-%03da'%v
    conf.description = 'lrt-base with inference only, PT init, and %s.'%str(mod)
    conf.pt_init = '%s_model.pt'%mod['dataset']
    conf.dataset = mod['dataset']
    conf.add_drift = mod.get('drift')
    conf.train = False
    configs[conf.name] = conf
        
    conf = deepcopy(configs['lrt-%03da'%v])
    conf.name = 'lrt-%03db'%v
    conf.description = 'lrt-%03da with SKS training + post-norming.'%v
    conf.train = True
    for l in convs + fcs:
        conf.__dict__[l][1]['rank'] = 8
    configs[conf.name] = conf


### Test configs ###
conf = deepcopy(configs['lrt-000'])
conf.name = 'lrt-100'
conf.description = '.'
conf.dataset = 'MNISTAON2k'
conf.validate = False
configs[conf.name] = conf

conf = deepcopy(configs['lrt-000'])
conf.name = 'lrt-101'
conf.description = '.'
stand = ('Standard', {'lr':0.1, 'norm_uv':None, 'count_version':1})
conf.set(convs, stand)
conf.set(fcs, stand)
conf.validate = False
configs[conf.name] = conf

conf = deepcopy(configs['lrt-000'])
conf.name = 'lrt-102'
conf.description = '.'
conf.validate = False
conf.use_bn = False
configs[conf.name] = conf

conf = deepcopy(configs['lrt-000'])
conf.name = 'lrt-103'
conf.description = '.'
conf.dataset = 'MNISTAON2k'
conf.validate = False
conf.lr = 1e-2
conf.norm_b = True
for l in convs + fcs:
    conf.__dict__[l][1]['lr'] = 1e-2
    conf.__dict__[l][1]['norm_uv'] = 'post'
configs[conf.name] = conf

### Validate All Configs ###
for _, config in configs.items():
    config.validate_attr()


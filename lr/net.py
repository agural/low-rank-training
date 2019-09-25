import os, sys, pdb, pickle

import numpy as np
import scipy as sp
from scipy.spatial.distance import cosine

from lr.utils import *
from lr.layers import *

def get_summary(model0, model):
    results = {}
    for v in model0.summary_variables:
        path = v.split('/')
        m0 = model0
        m1 = model
        try:
            for child in path:
                m0 = m0.__dict__[child]
                m1 = m1.__dict__[child]
        except Exception as e:
            continue
        if isinstance(m0, float):
            m0 = np.array([m0])
            m1 = np.array([m1])
        shape = m0.shape
        m0 = m0.flatten()
        m1 = m1.flatten()
        results[v] = {
            'shape': shape,
            'L20': np.linalg.norm(m0),
            'L2' : np.linalg.norm(m1 - m0),
            'cos': cosine(m0 + 1e-8, m1 + 1e-8),
            'mu' : np.mean(m1),
            'std': np.std(m1),
            'hist': np.histogram(m1, bins=20),
            'sample': np.concatenate((m1[:10], m1[-10:])),
        }
    return results

class Net(Module):
    def __init__(self, conf, data_meta):
        super(Net, self).__init__()
        self.conf = conf
        self.qx = FixedQuantize(conf.qbits['a'], clip=data_meta['input_range'][1])
        fi = data_meta['input_shape'][-1]
        fo = data_meta['output_shape'][-1]

        self.conv1 = Conv2D(fi, 128, 3, conf, activation=ReLU, weight_update=conf.upc1)
        
        self.conv2 = Conv2D(128, 128, 3, conf, activation=None, weight_update=conf.upc2)
        self.mp1   = MaxPool2D(2) #NOTE: conv-mp-act is implementable when Z is quantized.
        self.actc2 = ReLU(conf.qbits)
        self.bn2   = StreamBatchNorm(128, conf, update_every_ba=10) if conf.use_bn else Module()
        
        self.conv3 = Conv2D(128, 128, 3, conf, activation=ReLU, weight_update=conf.upc3)
        
        self.conv4 = Conv2D(128, 128, 3, conf, activation=None, weight_update=conf.upc4)
        self.mp2   = MaxPool2D(2)
        self.actc4 = ReLU(conf.qbits)
        self.bn4   = StreamBatchNorm(128, conf, update_every_ba=10) if conf.use_bn else Module()
        
        cur_size   = 128 * (data_meta['input_shape'][1]//4 - 3)**2
        self.fc1   = FC(cur_size, 512, conf, activation=ReLU, weight_update=conf.upd1)
        
        self.fc2   = FC(512, fo, conf, activation=None, weight_update=conf.upd2)
        
        self.loss_fn = SoftMaxCrossEntropyLoss(conf.qbits)
        self.set_path('/Net')
        self.init_summary()
     
    def forward(self, X):
        X = self.qx(X)
        self.x1 = X
        X = self.conv1(X)
        self.x2 = X
        X = self.conv2(X)
        self.x3 = X
        X = self.mp1(X)
        X = self.actc2(X)
        self.x4 = X
        X = self.bn2(X)
        self.x5 = X
        X = self.conv3(X)
        self.x6 = X
        X = self.conv4(X)
        self.x7 = X
        X = self.mp2(X)
        X = self.actc4(X)
        self.x8 = X
        X = self.bn4(X)
        self.x9 = X
        self.shapeA = X.shape
        X = X.reshape(X.shape[0], -1)
        self.x10 = X
        X = self.fc1(X)
        self.x11 = X
        X = self.fc2(X)
        self.x12 = X
        return X
        
    def backward(self):
        Grad = self.loss_fn.backward(1)
        Grad = self.fc2.backward(Grad)
        Grad = self.fc1.backward(Grad)
        Grad = Grad.reshape(self.shapeA)
        Grad = self.bn4.backward(Grad)
        Grad = self.actc4.backward(Grad)
        Grad = self.mp2.backward(Grad)
        Grad = self.conv4.backward(Grad)
        Grad = self.conv3.backward(Grad)
        Grad = self.bn2.backward(Grad)
        Grad = self.actc2.backward(Grad)
        Grad = self.mp1.backward(Grad)
        Grad = self.conv2.backward(Grad)
        Grad = self.conv1.backward(Grad)
        return Grad

    def init_summary(self):
        self.summary_variables = [
            'conv1/W', 'conv1/b', 'conv1/wup',
            'conv2/W', 'conv2/b', 'conv2/wup',
            'conv2/bn/gamma', 'conv2/bn/beta', 'conv2/bn/mu_ba', 'conv2/bn/std_ba',
            'conv3/W', 'conv3/b', 'conv3/wup',
            'conv4/W', 'conv4/b', 'conv4/wup',
            'conv4/bn/gamma', 'conv4/bn/beta', 'conv4/bn/mu_ba', 'conv4/bn/std_ba',
            'fc1/W', 'fc1/b', 'fc1/wup',
            'fc2/W', 'fc2/b', 'fc2/wup',
        ]
        if self.conf.use_bn:
            self.summary_variables += [
                'bn2/gamma', 'bn2/beta', 'bn2/mu_ba', 'bn2/std_ba',
                'bn4/gamma', 'bn4/beta', 'bn4/mu_ba', 'bn4/std_ba',
            ]
     
    def get_update_density(self):
        layers = [x for x in self.__dict__.values()
            if isinstance(x, Module) and hasattr(x, 'num_updates')]
        num_updates = sum([x.num_updates for x in layers])
        num_updatable = sum([x.W.size for x in layers])
        upd_steps = self.fc1.num_update_steps
        if upd_steps == 0:
            return 0
        try:
            update_density = num_updates / num_updatable / upd_steps
        except Exception as e:
            pdb.set_trace()
            print()
        return update_density
     
    def get_worst_case_updates(self):
        layers = [x for x in self.__dict__.values()
            if isinstance(x, Module) and hasattr(x, 'num_updates')]
        wc_updates = [np.max(x.W_updates) for x in layers]
        return max(wc_updates)

    def drift(self, *args, **kwargs):
        self.conv1.drift(*args, **kwargs)
        self.conv2.drift(*args, **kwargs)
        self.conv3.drift(*args, **kwargs)
        self.conv4.drift(*args, **kwargs)
        self.fc1.drift(*args, **kwargs)
        self.fc2.drift(*args, **kwargs)

    def uhist(self):
        return {
            'conv1':self.conv1.uhist,
            'conv2':self.conv2.uhist,
            'conv3':self.conv3.uhist,
            'conv4':self.conv4.uhist,
            'fc1':self.fc1.uhist,
            'fc2':self.fc2.uhist,
        }

    def init_from_pytorch(self, pt_model):
        self.conv1.init_from_pytorch(pt_model.conv1)
        self.conv2.init_from_pytorch(pt_model.conv2)
        if self.conf.use_bn: self.bn2.init_from_pytorch(pt_model.bn2)
        self.conv3.init_from_pytorch(pt_model.conv3)
        self.conv4.init_from_pytorch(pt_model.conv4)
        if self.conf.use_bn: self.bn4.init_from_pytorch(pt_model.bn4)
        self.fc1.init_from_pytorch(pt_model.fc1)
        self.fc2.init_from_pytorch(pt_model.fc2)


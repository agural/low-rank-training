import os, sys, pdb, time, pickle
from profilehooks import profile
from collections import deque, OrderedDict
import inspect

import numpy as np
import scipy as sp
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# General settings
dt = 'f4'

class Timer():
    def __init__(self):
        self.clocks = OrderedDict()
        self.r()
    
    def __call__(self, name=None):
        td = time.time() - self.t0
        if name is None:
            name = 'L' + str(inspect.currentframe().f_back.f_lineno)
            #name = 'L' + str(inspect.getframeinfo(inspect.currentframe().f_back).lineno)
        if name in self.clocks:
            cn, ctd = self.clocks[name]
            self.clocks[name] = (cn+1, ctd+td)
        else:
            self.clocks[name] = (1, td)
        self.r()
    
    def r(self):
        self.t0 = time.time()
    
    def print(self):
        print('='*80)
        print('Timer Results:')
        v1 = 0
        v0 = 1e8
        for k, v in self.clocks.items():
            v1 += v[1]
            v0 = min(v0, v[0])
            print('%10s    ==>    Total: %8.1fus    -    PerCall: %8.1fus'%(
                k, 1e6*v[1], 1e6*v[1]/v[0]))
        print('-'*80)
        print('%10s    ==>    Total: %8.1fus    -    PerCall: %8.1fus'%(
            'ALL', 1e6*v1, 1e6*v1/v0))
        print('='*80)

if __name__ == '__main__':
    t = Timer()
    for i in range(100):
        t.r()
        x = np.random.randn(64, 4096)
        w = np.random.randn(4096, 256)
        t()
        np.dot(x, w)
        t()
    t.print()
    

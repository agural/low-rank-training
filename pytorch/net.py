import os, sys, pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.utils import *
from pytorch.layers import *

device = torch.device('cuda')

qbits = {
    'w':8,
    'b':16,
    'a':8,
    'wmax':1.0,
    'bmax':8.0,
    'amax':2.0,
}

use_bn = True

class NetPT(QModule):
    def __init__(self, data_meta, qbits=qbits, bs=256):
        super(NetPT, self).__init__()
        self.qx    = FixedQuantize(bits=qbits['a'], signed=False, clip=1.0)
        self.qbits = qbits
        fi = data_meta['input_shape'][-1]
        fo = data_meta['output_shape'][-1]
        k = (3,3)
        p = (0,0)
        self.conv1 = QConvNd(2,  fi, 128, k, qbits, padding=p, post='bias', relu=True)
        self.conv2 = QConvNd(2, 128, 128, k, qbits, padding=p, post='bias', relu=True)
        self.bn2   = QStreamBatchnormNd(2, 128, qbits, 1 + np.ceil(10/bs), 20) if use_bn else QModule()
        self.conv3 = QConvNd(2, 128, 128, k, qbits, padding=p, post='bias', relu=True)
        self.conv4 = QConvNd(2, 128, 128, k, qbits, padding=p, post='bias', relu=True)
        self.bn4   = QStreamBatchnormNd(2, 128, qbits, 1 + np.ceil(10/bs), 20) if use_bn else QModule()
        cur_size   = 128 * (data_meta['input_shape'][1] // 4 - 3)**2
        self.fc1   = QLinear(cur_size, 512, qbits, bias=True, relu=True)
        self.fc2   = QLinear(512, fo, qbits, bias=True, relu=False, postact=False)

    def forward(self, x):
        x = x.permute(0,3,1,2) # CHW
        x = self.qx(x)
        self.x1 = x.permute(0,2,3,1)
        x = F.relu(self.conv1(x))
        self.x2 = x.permute(0,2,3,1)
        x = F.relu(self.conv2(x))
        self.x3 = x.permute(0,2,3,1)
        x = F.max_pool2d(x, 2)
        self.x4 = x.permute(0,2,3,1)
        x = self.bn2(x)
        self.x5 = x.permute(0,2,3,1)
        x = F.relu(self.conv3(x))
        self.x6 = x.permute(0,2,3,1)
        x = F.relu(self.conv4(x))
        self.x7 = x.permute(0,2,3,1)
        x = F.max_pool2d(x, 2)
        self.x8 = x.permute(0,2,3,1)
        x = self.bn4(x)
        self.x9 = x.permute(0,2,3,1)
        x = x.permute(0,2,3,1).contiguous() # HWC
        x = x.view(x.size(0), -1)
        self.x10 = x
        x = F.relu(self.fc1(x))
        self.x11 = x
        x = self.fc2(x)
        self.x12 = x
        return x
    
    def force_quantize(self):
        self.conv1.force_quantize()
        self.conv2.force_quantize()
        if use_bn: self.bn2.force_quantize()
        self.conv3.force_quantize()
        self.conv4.force_quantize()
        if use_bn: self.bn4.force_quantize()
        self.fc1.force_quantize()
        self.fc2.force_quantize()


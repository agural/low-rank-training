import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch.utils import flip, _ntuple, HardSigmoid, StochasticDropout, LayerNorm


class ST_round(torch.autograd.Function):
    def __init__(self, gradient_factor=1.0):
        super(ST_round, self).__init__()
        self.gradient_factor = gradient_factor
    def forward(self, x):
        return torch.round(x)
    def backward(self, grad_output):
        return grad_output * self.gradient_factor

class ST_ceil(torch.autograd.Function):
    def __init__(self, gradient_factor=1.0):
        super(ST_ceil, self).__init__()
        self.gradient_factor = gradient_factor
    def forward(self, x):
        return torch.ceil(x)
    def backward(self, grad_output):
        return grad_output * self.gradient_factor

class Quantize(nn.Module):
    '''
    Quantizes the input x to specified bitwidth (bits) and polarity (signed).
    There are four phases of operations, which can be toggled with set_qmode and should be run in order.
        1. 'F': Forward, no quantization - use for standard FP32 training.
        2. 'C': Calibration - run a forward pass in this mode to calibrate an initial setting for self.lt.
        3. 'T': Quantization Threshold Training - train weights/activations and clipping threshold simultaneously.
        4. 'Q': Quantization Training - freeze clipping thresholds and only train quantized weights/activations.
    '''
    def __init__(self, bits, signed, balanced=False, momentum=0.1, percentile=0.5, epsilon=1e-8):
        super(Quantize, self).__init__()
        self.lt = nn.Parameter(torch.zeros([]))
        self.bits = bits
        self.signed = signed
        self.midrise = signed and (bits <= 2)
        self.balanced = balanced
        self.mom = momentum
        self.pct = percentile
        self.eps = epsilon
        self.qmode = 'F'
        self.hist = []
        self.diag = {'last_in':None, 'last_out':None}
    
    def set_qmode(self, qmode):
        self.qmode = qmode
    
    def forward(self, x):
        if self.qmode != 'F':
            self.hist.append(float(self.lt.data.clone().detach().cpu().numpy()))
        if self.qmode in ['C']: # Calibrate
            values = x.clone().detach().cpu().numpy().flatten()
            lim_l = np.percentile(values, self.pct)
            lim_h = np.percentile(values, 100 - self.pct)
            lt_cur = np.log2(np.max(np.abs([lim_l, lim_h])) + self.eps)
            lt_cur = 3
            self.lt.data = (1 - self.mom) * self.lt + self.mom * lt_cur
        if self.qmode in ['F', 'C']: # Forward pass w/ no quantization
            x = x + 0 * self.lt
        if self.qmode in ['T', 'Q']: # Forward pass w/ quantization
            t = torch.exp(ST_ceil(self.qmode == 'T')(self.lt) * np.log(2)) # Only train the threshold self.lt in mode 'T'
            qmax = 2**(self.bits - 1) if self.signed else 2**self.bits
            s = x * qmax / t
            rounded = ST_round()(s - 0.5) + 0.5 if self.midrise else ST_round()(s)
            q = torch.clamp(rounded, -qmax + self.balanced + self.midrise*0.5 if self.signed else 0, qmax - 1 + self.midrise * 0.5) * t / qmax
            if not self.training:
                self.diag = {'last_in':x, 'last_out':q}
            x = q
        return x

class FixedQuantize(nn.Module):
    '''
    Quantizes the input x to specified bitwidth (bits) and polarity (signed).
    There are four phases of operations, which can be toggled with set_qmode and should be run in order.
        1. 'F': Forward, no quantization - use for standard FP32 training.
        2. 'C': Calibration - does nothing, since this is a fixed quantizer.
        3. 'T': Quantization Threshold Training - train weights/activations and clipping threshold simultaneously.
        4. 'Q': Quantization Training - freeze clipping thresholds and only train quantized weights/activations.
    '''
    def __init__(self, bits, signed, clip=1.0):
        super(FixedQuantize, self).__init__()
        self.register_buffer('lt', torch.tensor(np.log2(clip) - 1e-8))
        self.clip = clip
        self.bits = bits
        self.signed = signed
        self.qmode = 'F'
        self.hist = []
        self.diag = {'last_in':None, 'last_out':None}
    
    def set_qmode(self, qmode):
        self.qmode = qmode
    
    def forward(self, x):
        if self.qmode != 'F':
            self.hist.append(float(self.lt.data.clone().detach().cpu().numpy()))
        if self.qmode in ['T', 'Q']: # Forward pass w/ quantization
            t = torch.exp(ST_ceil(self.qmode == 'T')(self.lt) * np.log(2)) # Only train the threshold self.lt in mode 'T'
            qmax = 2**(self.bits - 1) if self.signed else 2**self.bits
            s = x * qmax / t
            rounded = ST_round()(s)
            q = torch.clamp(rounded, -qmax if self.signed else 0, qmax - 1) * t / qmax
            if not self.training:
                self.diag = {'last_in':x, 'last_out':q}
            x = q
        else:
            x = torch.clamp(x, -self.clip, self.clip)
        return x

class QModule(nn.Module):
    def __init__(self):
        super(QModule, self).__init__()
        self.qmode = 'F'
    def set_qmode(self, qmode):
        def f(t):
            if 'set_qmode' in dir(t): t.qmode = qmode
        return self.apply(f)
    def forward(self, x):
        return x

class QStreamBatchnormNd(QModule):
    def __init__(self, N, channels, qbits, update_every_ba=10, update_every_mv=1000, momentum=0.05, eps=1e-5):
        super(QStreamBatchnormNd, self).__init__()
        self.N = N
        self.channels     = channels
        self.update_every = int(max(1, update_every_ba))
        self.mom_ba  = 1.0 - 1.0 / update_every_ba
        self.mom_mv  = 1.0 - 1.0 / (update_every_mv / update_every_ba)
        self.eps     = 1e-8

        self.gamma   = nn.Parameter(torch.Tensor(channels))
        self.beta    = nn.Parameter(torch.Tensor(channels))
        self.register_buffer('mu',  torch.zeros(channels))
        self.register_buffer('std', torch.zeros(channels))
        self.register_buffer('mean_ba', torch.zeros(channels))
        self.register_buffer('msq_ba',  torch.zeros(channels))
        self.register_buffer('mean_mv', torch.zeros(channels))
        self.register_buffer('var_mv',  torch.ones(channels))
        self.register_buffer('step', torch.Tensor([0]))

        self.qgamma = FixedQuantize(qbits['b'], signed=True, clip=qbits['bmax'])
        self.qbeta  = FixedQuantize(qbits['b'], signed=True, clip=qbits['bmax'])
        self.qa     = FixedQuantize(qbits['a'], signed=True, clip=qbits['amax'])

        # Initialization
        nn.init.constant_(self.beta, 0.0)
        nn.init.constant_(self.gamma, 1.0)

    def forward(self, X):
        self.step += 1
        Xflat = X.permute(0,2,3,1).contiguous()
        init_shape = Xflat.shape
        Xflat = Xflat.view(-1, self.channels)
        Xflatd = Xflat.detach()
        if self.training:
            mean = Xflatd.mean(dim=0)
            msq  = (Xflatd**2).mean(dim=0)
            self.mean_ba = self.mom_ba * self.mean_ba + (1 - self.mom_ba) * mean
            self.msq_ba  = self.mom_ba * self.msq_ba + (1 - self.mom_ba) * msq
            bias_correct = 1 - self.mom_ba**self.step
            mu_ba = self.mean_ba / bias_correct
            var_ba = F.relu(self.msq_ba / bias_correct - mu_ba**2)
            std_ba = torch.sqrt(var_ba + self.eps)
            if self.step % self.update_every == 0:
                self.mean_mv = self.mom_mv * self.mean_mv + (1 - self.mom_mv) * mu_ba
                self.var_mv  = self.mom_mv * self.var_mv + (1 - self.mom_mv) * var_ba
        else:
            mu_ba = self.mean_mv
            std_ba = torch.sqrt(self.var_mv + self.eps)
        self.mu = mu_ba
        self.std = std_ba
        Y = self.qgamma(self.gamma) * (Xflat - self.mu) / self.std + self.qbeta(self.beta)
        return self.qa(Y.view(init_shape).permute(0,3,1,2).contiguous())

    def force_quantize(self):
        self.gamma.data = self.qgamma(self.gamma.data)
        self.beta.data  = self.qbeta(self.beta.data)

class QConvNd(QModule):
    def __init__(self, N, in_channels, out_channels, kernel_size, qbits, stride=1, padding=0, dilation=1, transposed=False, groups=1, post='bias', relu=False):
        super(QConvNd, self).__init__()
        if  in_channels % groups != 0: raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0: raise ValueError('out_channels must be divisible by groups')
        self.N              = N
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = _ntuple(N)(kernel_size)
        self.stride         = _ntuple(N)(stride)
        self.padding        = _ntuple(N)(padding)
        self.dilation       = _ntuple(N)(dilation)
        self.transposed     = transposed
        self.output_padding = _ntuple(N)(0)
        self.groups         = groups
        self.post           = post
        self.relu           = relu
        self.convNd         = F.conv1d if N==1 else F.conv3d if N==3 else F.conv2d
        if transposed: self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else: self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if post == 'bias': self.bias = nn.Parameter(torch.Tensor(out_channels))
        else: self.register_parameter('bias', None)
        if post == 'bn': self.bn = QStreamBatchnormNd(2, 128, qbits=qbits[1:])

        in_size = in_channels * np.prod(self.kernel_size)
        nlvar = 2 # Assumes ReLU will follow at some point.
        self.wmult = np.float32(2**(np.floor(np.log2(np.sqrt(3 * nlvar / in_size)))))
        
        # Quantization
        self.q_w = FixedQuantize(bits=qbits['w'], signed=True, clip=qbits['wmax'])
        if post == 'bias': self.q_b = FixedQuantize(bits=qbits['b'], signed=True, clip=qbits['bmax'])
        self.q_a = FixedQuantize(bits=qbits['a'], signed=False, clip=qbits['amax']) # Assume ReLU will follow.(not relu))
        
        # Initialization
        n = self.in_channels
        nn.init.uniform_(self.weight, -1, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        self.wq = self.wmult * self.q_w(self.weight)
        if self.post == 'bias':
            bq = self.q_b(self.bias).view(1,-1,1,1)
        else:
            bq = 0
        Z = self.convNd(x, self.wq, None, self.stride, self.padding, self.dilation, self.groups)
        Z = self.q_b(Z) + bq
        Z = self.q_b(Z)
        if self.post == 'bn': Z = self.bn(Z)
        self.Z = Z
        return self.q_a(self.Z)

    def force_quantize(self):
        self.weight.data = self.q_w(self.weight.data)
        if self.post == 'bias': self.bias.data   = self.q_b(self.bias.data)

    def get_weight_as_matrix(self):
        return self.weight.view([self.weight.shape[0], -1])

class QLinear(QModule):
    def __init__(self, in_features, out_features, qbits, bias=True, relu=False, postact=True):
        super(QLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias: self.bias = nn.Parameter(torch.Tensor(out_features))
        else: self.register_parameter('bias', None)
        self.relu = relu

        in_size = in_features
        nlvar = 2 if relu else 1
        self.wmult = np.float32(2**(np.floor(np.log2(np.sqrt(3 * nlvar / in_size)))))
        
        # Quantization
        self.q_w = FixedQuantize(bits=qbits['w'], signed=True, clip=qbits['wmax'])
        self.q_b = FixedQuantize(bits=qbits['b'], signed=True, clip=qbits['bmax'])
        self.q_a = FixedQuantize(bits=qbits['a'], signed=(not relu), clip=qbits['amax']) if postact else QModule()
        
        # Initialization
        nn.init.uniform_(self.weight, -1, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        wq = self.wmult * self.q_w(self.weight)
        bq = self.q_b(self.bias).view(1,-1)
        Z = F.linear(x, wq, None)
        Z = self.q_b(Z) + bq
        Z = self.q_b(Z)
        return self.q_a(Z)

    def force_quantize(self):
        self.weight.data = self.q_w(self.weight.data)
        self.bias.data   = self.q_b(self.bias.data)


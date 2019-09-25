import collections
from itertools import repeat
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

class HardSigmoid(nn.Module):
    def __init__(self, minx=-2.5, maxx=2.5, miny=0, maxy=1):
        super(HardSigmoid, self).__init__()
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
    def forward(self, x):
        return ((torch.clamp(x, min=self.minx, max=self.maxx) - self.minx) * (self.maxy - self.miny) / (self.maxx - self.minx) + self.miny)

class StochasticDropout(nn.Module):
    '''
    Similar to / inspired by Stochastic Delta: https://arxiv.org/pdf/1808.03578.pdf
    '''
    def __init__(self, sigma):
        super(StochasticDropout, self).__init__()
        self.sigma = sigma
    def forward(self, x):
        if self.training: sdo = torch.from_numpy(np.exp(np.random.normal(loc=-self.sigma**2/2, scale=self.sigma, size=list(x.size()))).astype('f'))
        else: sdo = torch.FloatTensor([1])
        if x.is_cuda: sdo = sdo.cuda()
        return x * sdo

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def get_quant_nodes(mod, path='Net'):
    qlist = []
    if mod.__class__.__name__ == 'Quantize':
        qlist.append((path, mod))
    else:
        for name, child in mod.named_children():
            qlist += get_quant_nodes(child, path + '/' + name)
    return qlist

def get_quant_threshs(mod):
    qlist = get_quant_nodes(mod)
    return map(lambda qm: float(qm[1].lt.data.cpu().numpy()), qlist)

class MaxNormOpt(optim.Optimizer):
    """Implements a Gradient MaxNorm Optimizer.
    
		Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
    """

    def __init__(self, params, lr=1.0, weight_decay=0.0, eta=0.999, eps=1e-4, qbits=4):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.eta = eta
        self.eps = eps
        self.qbits = qbits
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(MaxNormOpt, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('POpt does not support sparse gradients')
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_stat'] = []
                    state['gvar'] = 1.0
                    state['gmax'] = 0.0
                state['step'] += 1
                
                if state['step'] % 1000 == 0:
                    grad_np = grad.cpu().numpy()
                    m1 = np.mean(grad_np)
                    m2 = np.std(grad_np)
                    m3 = np.mean(grad_np**3)
                    m3 = np.sign(m3) * np.abs(m3)**(1/3.0)
                    m4 = np.mean(grad_np**4)**(1/4.0)
                    gmin = np.min(grad_np)
                    gmax = np.max(grad_np)
                    state['grad_stat'].append((m1, m2, m3, m4, gmin, gmax))
                
                gmax = torch.max(torch.abs(grad)) + self.eps
                state['gmax'] = self.eta * state['gmax'] + (1 - self.eta) * gmax
                state['gnorm'] = max(gmax, state['gmax'] / (1 - self.eta**state['step']))

                ### Quantization
                grad = grad * group['lr'] / state['gnorm']
                qbits = self.qbits
                scale = 2**(1-qbits)
                grad = torch.clamp(torch.round(grad / scale), -2**(qbits-1), 2**(qbits-1)-1) * scale

                #p.data.add_(-group['lr'], grad)
                p.data.add_(-1.0, grad)
        return loss


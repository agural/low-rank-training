import os, sys, pdb, pickle
from profilehooks import profile

import time, math, random
import numpy as np
import scipy as sp
from scipy.spatial.distance import cosine

from lr.sks import SKS
from lr.utils import *


def im2col(X, kernel, strides=(1,1), padding=(0,0)):
    '''
    Views X as the matrix-version of a strded image.
    https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    
    @arg X: Input batch of images B x Hi x Wi x Fi.
    @arg kernel: Tuple of kernel dimensions kR x kC.
    @arg strides: Tuple of strides sR x sC.
    @arg padding: Tuple of paddings pR x pC.

    @return: X viewed in matrix form B x Ho x Wo x (kH*kW*Fi).
    '''
    kR, kC = kernel
    sR, sC = strides
    pR, pC = padding
    B, Hi, Wi, Fi = X.shape
    sB, sH, sW, sF = X.strides
    Ho = int((Hi + 2*pR - kR)/sR) + 1
    Wo = int((Wi + 2*pC - kC)/sC) + 1
    out_shape = B, Ho, Wo, kR, kC, Fi
    out_strides = sB, sR*sH, sC*sW, sH, sW, sF
    Xpad = np.pad(X, ((0,0),(pR,pR),(pC,pC),(0,0)), mode='constant')
    Xcol = np.lib.stride_tricks.as_strided(Xpad, shape=out_shape, strides=out_strides)
    Xcol = Xcol.reshape(B, Ho, Wo, kR * kC * Fi)
    return Xcol

class MaxNorm():
    def __init__(self, beta=0.999, eps=1e-4):
        self.step = 0
        self.beta = beta
        self.eps = eps
        self.x_max = eps

    def __call__(self, x):
        self.step += 1
        x_max = np.max(np.abs(x)) + self.eps
        self.x_max = self.beta * self.x_max + (1 - self.beta) * x_max
        x_max_tilde = self.x_max / (1 - self.beta**self.step)
        x_normed = x / max(x_max, x_max_tilde)
        return x_normed

class Module():
    def __init__(self, *args, name=None, **kwargs):
        self.name = name or self.__class__.__name__
        self.mode = {'train':True, 'quant':False, 'qcal':False}
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def recursive_apply(self, fn, *args, **kwargs):
        for modn, mod in self.__dict__.items():
            if isinstance(mod, Module):
                getattr(mod, fn)(*args, **kwargs)
    
    def set_path(self, parent):
        self.path = parent
        for modn, mod in self.__dict__.items():
            if isinstance(mod, Module):
                mod.set_path(parent + '/' + modn)

    def forward(self, X):
        return X
    
    def backward(self, Grad):
        return Grad
    
    def update(self, lr):
        if hasattr(self, 'my_update'):
            self.my_update(lr)
        self.recursive_apply('update', lr)
    
    def get_all(self, class_, path_):
        insts = []
        if isinstance(self, class_):
            insts.append((path_, self))
        for modn, mod in self.__dict__.items():
            if isinstance(mod, Module):
                insts += mod.get_all(class_, path_ + '/' + modn)
        return insts
    
    def set_mode(self, **kwargs):
        self.mode.update(kwargs)
        self.recursive_apply('set_mode', **kwargs)

class FixedQuantize(Module):
    def __init__(self, bits, clip=1.0):
        super(FixedQuantize, self).__init__()
        self.signed = bits < 0
        self.bits = abs(bits)
        self.midrise = 0.5 if self.bits <= 2 else 0
        self.n = -2**(self.bits-1) if self.signed else 0
        self.p = 2**(self.bits-1) - 1 if self.signed else 2**(self.bits) - 1
        self.s = 2**(np.ceil(np.log2(clip) - 1e-8) + self.signed - self.bits)
        self.Aq = {}
        self.step = 0
        self.rel_error = 0
   
    def forward(self, X, fid=0):
        if not self.mode['quant']: return X
        Af = X / self.s
        Aq = np.round(Af - self.midrise)
        Q = self.s * (np.clip(Aq, self.n, self.p) + self.midrise)
        if self.step % 100 == 0:
            relE = np.sum((Af - Aq)**2) / Af.size / (np.std(Af) + 1e-6)
            self.rel_error = 0.9 * self.rel_error + 0.1 * relE
        self.Aq[fid] = Aq
        return Q
    
    def backward(self, Grad, fid=0):
        if not self.mode['quant']: return Grad
        Aq = self.Aq[fid]
        return Grad * ((Aq >= self.n) & (Aq <= self.p))

    def my_update(self, lr):
        self.step += 1

class ReLU(Module):
    def __init__(self, qbits):
        super(ReLU, self).__init__()
        self.qa = FixedQuantize(qbits['a'], clip=qbits['amax'])
        self.qg = FixedQuantize(qbits['g'], clip=qbits['gmax'])

    def forward(self, X):
        self.A = self.qa(np.maximum(X, 0))
        return self.A

    def backward(self, Grad):
        Grad = self.qa.backward(Grad)
        Grad *= (self.A > 0)
        return self.qg(Grad)

class SoftMaxCrossEntropyLoss(Module):
    def __init__(self, qbits):
        super(SoftMaxCrossEntropyLoss, self).__init__()
        self.qg = FixedQuantize(qbits['g'], clip=qbits['gmax'])
        self.eps = np.exp(-100).astype(dt)

    def forward(self, X, Yt):
        self.batch_size = X.shape[0]
        self.Yt = Yt.astype(dt)
        self.X = X
        exp = np.exp(X - np.max(X, 1, keepdims=True)) + self.eps
        self.Yp = exp / np.sum(exp, 1, keepdims=True)
        self.L = -np.sum(self.Yt * np.log(self.Yp)) / self.batch_size
        return self.L
    
    def backward(self, Grad=1.0):
        Grad = self.qg((Grad * (self.Yp - self.Yt)).astype(dt))
        return Grad

class MaxPool2D(Module):
    def __init__(self, kernel):
        super(MaxPool2D, self).__init__()
        self.kernel_size = (kernel, kernel)
    
    def forward(self, X):
        self.A = X
        Xcol = im2col(X, kernel=self.kernel_size, strides=self.kernel_size)
        Xcol = Xcol.reshape(Xcol.shape[:3] + (self.kernel_size[0] * self.kernel_size[1], X.shape[-1]))
        max_pos = np.argmax(Xcol, axis=3)
        self.idx = list(np.ogrid[[slice(Xcol.shape[ax]) for ax in range(Xcol.ndim) if ax != 3]])
        self.idx.insert(3, max_pos)
        self.idx = tuple(self.idx)
        Z = Xcol[self.idx]
        return Z
    
    def backward(self, Grad):
        dZ = np.zeros(Grad.shape[:3] + (self.kernel_size[0] * self.kernel_size[1], Grad.shape[-1]))
        dZ[self.idx] = Grad
        dZ = dZ.reshape(Grad.shape[:3] + self.kernel_size + (Grad.shape[-1],))
        dZ = np.transpose(dZ, (0,1,3,2,4,5))
        dZ = dZ.reshape(self.A.shape)
        return dZ

class Dropout(Module):
    def __init__(self, keep_prob, qbits):
        super(Dropout, self).__init__()
        self.qa = FixedQuantize(qbits['a'], clip=qbits['amax'])
        self.p = keep_prob
    
    def forward(self, X):
        if self.mode['train']:
            self.mask = np.random.binomial(2, self.p, X.shape).astype(dt)
            X *= self.mask / self.p
            X = self.qa(X)
        return X
     
    def backward(self, Grad):
        if self.mode['train']:
            Grad = self.qa.backward(Grad)
            Grad *= self.mask / self.p
        return Grad

class StreamBatchNorm(Module):
    def __init__(self, channels, conf, update_every_ba=10, update_every_mv=100):
        super(StreamBatchNorm, self).__init__()
        self.conf = conf
        qbits = conf.qbits
        self.qgamma = FixedQuantize(qbits['b'], clip=qbits['bmax'])
        self.qbeta  = FixedQuantize(qbits['b'], clip=qbits['bmax'])
        self.qmean  = FixedQuantize(qbits['b'], clip=qbits['bmax'])
        self.qmsq   = FixedQuantize(qbits['b'], clip=qbits['bmax']**2)
        self.qa = FixedQuantize(-abs(qbits['a']), clip=qbits['amax'])
        self.qg = FixedQuantize(qbits['g'], clip=qbits['gmax'])

        self.channels = channels
        self.update_every = update_every_ba
        self.mom_ba = np.float32(1.0 - 1.0 / update_every_ba)
        self.mom_mv = np.float32(1.0 - 1.0 / (update_every_mv/update_every_ba))
        self.eps = np.float32(1e-8)

        self.step = 0
        self.mean_ba = np.zeros((1,channels), dtype=dt)
        self.msq_ba  = np.zeros((1,channels), dtype=dt)
        self.mu_ba   = np.zeros((1,channels), dtype=dt)
        self.std_ba  = np.zeros((1,channels), dtype=dt)
        self.mean_mv = np.zeros((1,channels), dtype=dt)
        self.var_mv  =  np.ones((1,channels), dtype=dt)
        self.beta    = np.zeros((1,channels), dtype=dt)
        self.gamma   =  np.ones((1,channels), dtype=dt)

        self.MN_dg = MaxNorm()
        self.MN_db = MaxNorm()

    def init_from_pytorch(self, pt_layer):
        self.step    = pt_layer.step.detach().cpu().numpy()[0]
        self.mean_ba = pt_layer.mean_ba.detach().cpu().numpy().reshape(1, -1)
        self.msq_ba  = pt_layer.msq_ba.detach().cpu().numpy().reshape(1, -1)
        self.mean_mv = pt_layer.mean_mv.detach().cpu().numpy().reshape(1, -1)
        self.var_mv  = pt_layer.var_mv.detach().cpu().numpy().reshape(1, -1)
        self.beta    = pt_layer.beta.detach().cpu().numpy().reshape(1, -1)
        self.gamma   = pt_layer.gamma.detach().cpu().numpy().reshape(1, -1)

    def forward(self, X):
        self.step += 1
        self.X = X.reshape((-1, self.channels))
        if self.mode['train']:
            mean = np.mean(self.X, axis=0)
            msq  = np.mean(self.X**2, axis=0)
            self.mean_ba = self.mom_ba * self.mean_ba + (1 - self.mom_ba) * mean
            self.msq_ba  = self.mom_ba * self.msq_ba + (1 - self.mom_ba) * msq
            
            # NOTE: In an online setting, only mean_ba and msq_ba need to be stored
            # since mu_ba, std_ba can be calculated on the fly and mean_mv, var_mv
            # will not be used online. Therefore we only quantize these two.
            # Dynamic range can be an issue, so we use max-quantization.
            self.mean_ba = self.qmean(self.mean_ba)
            self.msq_ba  = self.qmsq(self.msq_ba)
            
            bias_correct = 1 - self.mom_ba**self.step
            self.mu_ba = self.mean_ba / bias_correct
            self.var_ba = np.clip(self.msq_ba / bias_correct - self.mu_ba**2, self.eps, 1/self.eps)
            
            # NOTE: Using +eps doesn't work too well because in backprop, it causes
            # multiplication of the delta by ~1/eps, which ruins the dynamic
            # range for the qg quantizer.
            self.std_ba = np.sqrt(self.var_ba + self.eps)

            if self.step % self.update_every == 0:
                self.mean_mv = self.mom_mv * self.mean_mv + (1 - self.mom_mv) * self.mu_ba
                self.var_mv  = self.mom_mv * self.var_mv + (1 - self.mom_mv) * self.var_ba
            mu_ba = self.mu_ba
            std_ba = self.std_ba
        else:
            mu_ba = self.mean_mv
            std_ba = np.sqrt(self.var_mv + self.eps)
        self.mu = mu_ba
        self.std = std_ba
        Y = self.qgamma(self.gamma) * (self.X - self.mu) / self.std + self.qbeta(self.beta)
        return self.qa(Y.reshape(X.shape))
    
    def backward(self, Grad):
        self.dY = self.qa.backward(Grad).reshape((-1, self.channels))
        std_ba = self.std_ba
        dX = self.dY * self.gamma / self.std_ba
        Grad = self.qg(dX.reshape(Grad.shape))
        return Grad
    
    def my_update(self, lr):
        dg = np.sum(self.dY * ((self.X - self.mu_ba) / self.std_ba), 0)
        dg = self.qgamma.backward(dg)
        if self.conf.norm_b:
            self.dg_normed = self.MN_dg(dg)
        else:
            self.dg_normed = dg
        self.gamma -= self.qgamma.backward(lr * self.dg_normed)
        self.gamma = self.qgamma(self.gamma, fid=5)

        db = np.sum(self.dY, 0)
        db = self.qbeta.backward(db)
        if self.conf.norm_b:
            self.db_normed = self.MN_db(db)
        else:
            self.db_normed = db
        self.beta -= self.qbeta.backward(lr * self.db_normed)
        self.beta = self.qbeta(self.beta, fid=5)

class WeightModule(Module):
    def __init__(self, in_size, out_size, conf, activation=ReLU, nlvar=2, weight_update=('SKS', {})):
        super(WeightModule, self).__init__()
        self.conf = conf
        self.qbits = conf.qbits
        self.qw = FixedQuantize(self.qbits['w'], clip=self.qbits['wmax'])
        self.qb = FixedQuantize(self.qbits['b'], clip=self.qbits['bmax'])
        self.qg = Module() # NOTE: Gradient quantization handled in activation function.

        self.W = np.random.uniform(-1,1, size=(in_size, out_size)).astype(dt)
        self.b = np.zeros((1, out_size), dtype=dt)
        self.wup = np.zeros_like(self.W, dtype=dt)
        self.wmult = np.float32(2**(np.floor(np.log2(np.sqrt(3 * nlvar / in_size)))))
        if activation is None:
            self.act = Module()
        else:
            self.act = activation(self.qbits)

        self.step = 0
        self.MN_db = MaxNorm()
        self.weight_update = weight_update
        self.num_updates = 0
        self.num_update_steps = 0
        self.W_updates = np.zeros((in_size, out_size), dtype='i4')
        if weight_update[0] == 'Standard':
            init = {'lr':1e-4, 'norm_uv':'post', 'count_version':2}
            if sorted(init.keys()) != sorted(weight_update[1].keys()):
                print('WARNING: SKS kwargs key mismatch.')
            self.wc = { k: np.float(v) if type(v) == float else v for k,v in weight_update[1].items() }
            self.MN_fA = MaxNorm()
            self.MN_dZ = MaxNorm()

        if weight_update[0] == 'SKS':
            init = {'lr':1e-4, 'rank':10, 'zerovar':False, 'kappa_th':10,
                    'pseudo_batch':10, 'lr_pb_pow':1.0,
                    'norm_uv':'post', 'lr_pb_pow_post':0.5,
                    'min_density':1e-2, 'discount':0.9,
                    'rho_samp':1.0, 'lookahead':False,}
            if sorted(init.keys()) != sorted(weight_update[1].keys()):
                print('WARNING: SKS kwargs key mismatch.')
            self.wc = { k: np.float(v) if type(v) == float else v for k,v in weight_update[1].items() }
            self.MN_U = MaxNorm()
            self.MN_V = MaxNorm()
            self.uv = SKS(in_size, out_size, self.wc['rank'], zerovar=self.wc['zerovar'],
                kappa_th=self.wc['kappa_th'], qb=self.qbits['b'])
            self.update_lr_multiplier = 0
        self.dhist = []
        self.uhist = []

    def get_W_SKS(self):
        if 'lookahead' in self.wc and self.wc['lookahead']:
            U, V = self.uv.grab()
            Wlookahead = self.get_lr_SKS() * np.dot(U, V.T)
            Wf = self.W + 2 * Wlookahead # Multiply by 2 to match https://arxiv.org/abs/1907.08610
        else:
            Wf = self.W
        return self.wmult * self.qw(Wf, fid=0)

    def get_lr_SKS(self):
        lr_mult = np.power(self.update_lr_multiplier, self.wc['lr_pb_pow'])
        if self.wc['norm_uv'] == 'post':
            lr_mult *= np.power(self.update_lr_multiplier, self.wc['lr_pb_pow_post'])
        lr = self.wc['lr'] * lr_mult
        return lr

    def drift(self, drift_params, downsample):
        n_samp = 1e6 / downsample
        wmax = self.conf.qbits['wmax']
        if drift_params[0] == 'analog':
            sigma = drift_params[1] / np.sqrt(n_samp)
            self.W += sigma * np.random.randn(*self.W.shape)
        if drift_params[0] == 'digital':
            pflip = drift_params[1] / n_samp
            w_bl = abs(self.conf.qbits['w'])
            w_qn = 2**(w_bl - 1)
            Wint = ((self.W + wmax) * w_qn / wmax).astype('i4')
            bit_flips = np.random.binomial(2, pflip, size=(*self.W.shape, w_bl))
            int_flips = np.dot(bit_flips, 2**np.arange(w_bl))
            Wnew = np.bitwise_xor(Wint, int_flips) * wmax / w_qn - wmax
            self.W = Wnew
        self.W = np.clip(self.W, -wmax, wmax).astype(dt)
    
    def update_b(self, lr, db):
        if self.conf.norm_b:
            db_normed = self.MN_db(db)
        else:
            db_normed = db
        self.b -= self.qb(self.qb.backward(lr * db_normed, fid=0), fid=5)
        if self.conf.quant_regen:
            self.b = self.qb(self.b, fid=8)
    
    def update_W_standard_A(self, a, d):
        if self.wc['norm_uv'] == 'post':
            fA = self.MN_fA(a)
            dZ = self.MN_dZ(d)
        else:
            fA = a
            dZ = d
        self.wup = self.wc['lr'] * np.dot(fA.T, dZ)
        self.wup = self.qw.backward(self.wup, fid=0)
        self.wup = self.qw(self.wup, fid=5)
        self.W -= self.wup
        self.W = self.qw(self.W, fid=8)
        return fA, dZ
     
    def update_W_standard_B(self, nonzero):
        nonzero_sum = np.sum(nonzero)
        if self.mode['quant']:
            self.W_updates += nonzero
            self.num_updates += nonzero_sum
     
    def update_W_SKS_A(self, a, d):
        try:
            if self.wc['norm_uv'] == 'pre':
                Ax = self.MN_U(a)
                Dx = self.MN_V(d)
            else:
                Ax = a
                Dx = d
            self.uv.update(Ax, Dx)
        except Exception as e:
            print('\n\nError: %s\n'%e)
            pdb.set_trace()
    
    def update_W_SKS_B(self):
        self.update_lr_multiplier *= self.wc['discount']
        self.update_lr_multiplier += self.wc['pseudo_batch']
        
        U, V = self.uv.grab()
        if self.wc['norm_uv'] == 'post':
            U_normed = self.MN_U(U)
            V_normed = self.MN_V(V)
            dWraw = np.dot(U_normed, V_normed.T)
        else:
            dWraw = np.dot(U, V.T)

        #NOTE: We need to adjust the learning rate based on the sparser updates.
        # We estimate this as the product of three components:
        #   1) original learning rate = lr
        #   2) effective batch size = update_lr_multiplier
        #   3) ratio of grad norm sum to sum of grad norms = sqrt(update_lr_multiplier)
        #      (3) is applied if grad norm on U, V instead of A, dZ
        self.wup = self.get_lr_SKS() * dWraw
        self.wup = self.qw.backward(self.wup, fid=0)
        self.wup = self.qw(self.wup, fid=5)
        nonzero = np.abs(self.wup) > 1e-12
        nonzero_sum = np.sum(nonzero)
        if nonzero_sum / self.wup.size > self.wc['min_density']:
            cos = cosine(self.uv.Mtrue.flatten(), self.wup.flatten())
            self.uhist.append((self.update_lr_multiplier, cos))
            self.W -= self.wup
            if self.conf.quant_regen:
                self.W = self.qw(self.W, fid=8)
            self.uv.reset()
            self.update_lr_multiplier = 0
            if self.mode['quant']:
                self.W_updates += nonzero
                self.num_updates += nonzero_sum
        else:
            self.uv.Cx *= self.wc['discount']

class Conv2D(WeightModule):
    def __init__(self, in_channels, out_channels, kernel_size,
            conf, activation=ReLU, weight_update=('SKS', {})):
        self.kernel = (kernel_size, kernel_size)
        in_size = kernel_size**2 * in_channels
        out_size = out_channels
        nlvar = 2 # Assumes ReLU will follow at some point.
        super(Conv2D, self).__init__(in_size, out_size, conf, activation, nlvar, weight_update)
    
        if weight_update[0] == 'SKS':
            self.pixel_step = 0

    def init_from_pytorch(self, pt_layer):
        W = pt_layer.weight.detach().cpu().numpy()
        self.W = np.transpose(W, (2,3,1,0)).reshape(-1, W.shape[0])
        self.b = pt_layer.bias.detach().cpu().numpy().reshape(1, -1)

    def forward(self, X):
        self.in_shape = X.shape
        self.A = im2col(X, self.kernel)
        self.out_shape = self.A.shape
        self.A = self.A.reshape(-1, self.A.shape[-1])
        self.wq = self.get_W_SKS()
        self.Zcol = np.dot(self.A, self.wq)
        self.Zcol = self.qb(self.Zcol.reshape(self.out_shape[:-1] + (self.W.shape[1],)), fid=2)
        self.Zcol = self.qb(self.Zcol + self.qb(self.b, fid=0), fid=1)
        return self.act(self.Zcol)

    def backward(self, Grad):
        self.dZ = self.act.backward(Grad)
        self.dZ = self.qb.backward(self.dZ, fid=1)
        B, Ho, Wo, _ = self.dZ.shape
        self.dZ_col = self.qb.backward(self.dZ, fid=2).reshape(-1, self.dZ.shape[-1])
        dX_col = np.dot(self.dZ_col, self.wq.T)
        dX_col = dX_col.reshape(B, Ho, Wo, self.kernel[0], self.kernel[1], self.in_shape[-1])
        dX = np.zeros(self.in_shape, dtype=dt)
        for i in range(self.kernel[0]):
            for j in range(self.kernel[1]):
                dX[:,i:i+Ho,j:j+Wo,:] += dX_col[:,:,:,i,j,:]
        return self.qg(dX)
     
    def my_update(self, lr):
        self.step += 1
        if self.mode['quant']:
            self.num_update_steps += 1

        a = self.A
        d = self.dZ.reshape(-1, self.dZ.shape[-1])
        
        if self.conf.train_b:
            self.update_b(lr, np.sum(d, 0))
    
        if self.conf.train_W:
            d = self.wmult * self.dZ_col
            self.dhist.append((a,d))
            self.dhist = self.dhist[-10:]

            # Standard SGD.
            if self.weight_update[0] == 'Standard':
                fA_normed, dZ_normed = self.update_W_standard_A(a, d)
                #NOTE: Two possible methods of finding number nonzero:
                #   (1) Non-zero elements when aggregating dot products across all pixels.
                #   (2) Non-zero elements due to updating at each pixel.
                # (1) is typically an underestimate in memory-constrained settings so we go with (2).
                if self.wc['count_version'] == 1:
                    nonzero = np.abs(self.wup) > 1e-12
                else:
                    fax = (np.abs(fA_normed) > 1e-12).astype('f4') # FP32 is faster than Int32
                    dzx = (np.abs(dZ_normed) > 1e-12).astype('f4')
                    nonzero = np.dot(fax.T, dzx).astype('i')
                self.update_W_standard_B(nonzero)

            # Low-rank SGD.
            if self.weight_update[0] == 'SKS':
                if self.wc['rho_samp'] < 1.0:
                    m = int(self.wc['rho_samp'] * d.shape[0])
                    samps = np.argsort(np.var(d, 1))[-m:] #NOTE: This may not be implementable.
                    samps = np.sort(samps)
                else:
                    samps = np.arange(d.shape[0])
                for i in samps:
                    self.pixel_step += 1
                    self.update_W_SKS_A(a[i], d[i])
                    if self.pixel_step % (self.wc['pseudo_batch'] * len(samps) + 13) == 0:
                        self.update_W_SKS_B()

class FC(WeightModule):
    def __init__(self, in_size, out_size, conf, activation=ReLU, weight_update=('SKS', {})):
        nlvar = 2 if activation == ReLU else 1
        super(FC, self).__init__(in_size, out_size, conf, activation, nlvar, weight_update)

    def init_from_pytorch(self, pt_layer):
        W = pt_layer.weight.detach().cpu().numpy()
        self.W = W.T
        self.b = pt_layer.bias.detach().cpu().numpy().reshape(1, -1)
    
    def forward(self, X):
        self.A = X
        self.wq = self.get_W_SKS()
        self.Zcol = self.qb(np.dot(X, self.wq), fid=2)
        self.Z = self.qb(self.Zcol + self.qb(self.b, fid=0), fid=1)
        return self.act(self.Z)
    
    def backward(self, Grad):
        self.dZ = self.qb.backward(self.act.backward(Grad), fid=1)
        return self.qg(np.dot(self.qb.backward(self.dZ, fid=2), self.wq.T))
   
    def my_update(self, lr):
        self.step += 1
        if self.mode['quant']:
            self.num_update_steps += 1
        
        if self.conf.train_b:
            self.update_b(lr, self.dZ)

        if self.conf.train_W:
            self.dZ = self.wmult * self.qb.backward(self.dZ, fid=2)
            self.dhist.append((self.A,self.dZ))
            self.dhist = self.dhist[-10:]

            # Standard SGD.
            if self.weight_update[0] == 'Standard':
                self.update_W_standard_A(self.A, self.dZ)
                nonzero = np.abs(self.wup) > 1e-12
                self.update_W_standard_B(nonzero)

            # Low-rank SGD.
            if self.weight_update[0] == 'SKS':
                self.update_W_SKS_A(self.A[0], self.dZ[0])
                if self.step % self.wc['pseudo_batch'] == 0:
                    self.update_W_SKS_B()
 

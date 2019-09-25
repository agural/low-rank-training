import os, sys, pdb, time, pickle
from profilehooks import profile
from collections import deque

import numpy as np
import scipy as sp
from scipy.spatial.distance import cosine
#import matplotlib.pyplot as plt

from numba import jit

from lr.utils import *

EPS = np.float32(1e-12)
dt = np.float32

@jit(nopython=True)
def apply_updates(Uhat, Vhat, Mult, Cx, Us, Vs, zerovar=0, kappa_th=10, qb=np.float32(16), rbits=None):
    N, nr = Us.shape
    _, nc = Vs.shape
    r = len(Cx)
    q = r + 1
    au = np.zeros(q, dtype=dt)
    av = np.zeros(q, dtype=dt)
    one = np.float32(1)

    # Quantization params
    qh = np.float32(2**(qb-1))
    qn = -qh
    qp = qh - one
    for n in range(N):
        u = Us[n]
        v = Vs[n]
        u = np.copy(u)
        v = np.copy(v)

        # Numerically stable Gram-Schmidt.
        Uhat *= Mult[0]
        Vhat *= Mult[1]
        Mult[0] = one
        Mult[1] = one
        for i in range(r):
            au[i] = np.dot(Uhat[i], u)
            av[i] = np.dot(Vhat[i], v)
            u -= Uhat[i] * au[i]
            v -= Vhat[i] * av[i]

        au[-1] = np.sqrt(np.dot(u,u)) + EPS
        av[-1] = np.sqrt(np.dot(v,v)) + EPS
        if Cx[0] > kappa_th * au[-1] * av[-1]: continue
        C = np.outer(au, av)
        for i in range(r): C[i,i] += Cx[i]
        Uhat[-1] = u / au[-1]
        Vhat[-1] = v / av[-1]
        Mult[0] = np.max(np.abs(Uhat)) + EPS
        Mult[1] = np.max(np.abs(Vhat)) + EPS
        Uhat = np.floor(Uhat * qp / Mult[0] + np.float32(0.5))
        Uhat = np.minimum(np.maximum(Uhat, qn), qp) / qp
        Vhat = np.floor(Vhat * qp / Mult[1] + np.float32(0.5))
        Vhat = np.minimum(np.maximum(Vhat, qn), qp) / qp

        # Find the SVD.
        Cu, Cd, Cv = np.linalg.svd(C)
        CdQT = np.eye(q, dtype=dt)[:r]
        Cx = Cd[:r]
        if not zerovar and Cd[-1] > EPS:
            # Approximate small singular values by mixing in Z matrix.
            s1 = Cd[r]
            m = r
            for k in range(1,r+1):
                m -= 1
                if k * Cd[m] > s1 + Cd[m]:
                    k -= 1
                    m += 1
                    break
                s1 += Cd[m]

            # https://math.stackexchange.com/a/525587
            sk = np.float32(s1 / k)
            v = np.sqrt(one - Cd[m:] / sk)#.astype(dt)
            v[0] -= one
            rnd = rbits[:len(v),n]
            Z = np.outer(rnd * v, v[1:] / v[0])
            for i in range(1, len(v)): Z[i,i-1] += rnd[i]#one

            # Create full diagonal approximation, represented in QR form as CdQ * CdR.
            CdQT[m:,m:] = Z.T
            Cx[m:] = sk

        # Update Uhat, Vhat.
        Uhat[:-1] = np.dot(np.dot(CdQT, Cu.T), Uhat) * Mult[0]
        Vhat[:-1] = np.dot(np.dot(CdQT, Cv), Vhat) * Mult[1]

        # Quantize Uhat, Vhat
        Mult[0] = np.max(np.abs(Uhat[:-1])) + EPS
        Mult[1] = np.max(np.abs(Vhat[:-1])) + EPS
        Uhat = np.floor(Uhat * qp / Mult[0] + np.float32(0.5))
        Uhat = np.minimum(np.maximum(Uhat, qn), qp) / qp
        Vhat = np.floor(Vhat * qp / Mult[1] + np.float32(0.5))
        Vhat = np.minimum(np.maximum(Vhat, qn), qp) / qp
    return Uhat, Vhat, Mult, Cx


class SKS():
    ''' Streaming version of the Optimal Kronecker Sum Algorithm. '''
    def __init__(self, nr, nc, rank, zerovar=False, kappa_th=10, qb=16):
        self.nr = nr
        self.nc = nc
        self.rank = rank
        self.zerovar = zerovar
        self.kappa_th = kappa_th
        self.qb = np.float32(np.abs(qb))
        self.uvs = deque()
        self.reset()

    def reset(self):
        r = self.rank
        self.Uhat = np.zeros((r + 1, self.nr), dtype=dt)
        self.Vhat = np.zeros((r + 1, self.nc), dtype=dt)
        self.Mult = np.ones(2, dtype=dt)
        self.Cx = np.zeros(r, dtype=dt)
        self.Mtrue = np.zeros((self.nr, self.nc), dtype=dt)

    def update(self, u, v):
        #NOTE: u and v are assumed to have the proper data type dt.
        self.uvs.append((u, v))

    def aggregate(self):
        N = len(self.uvs)
        self.Us = np.zeros((N, self.nr), dtype=dt)
        self.Vs = np.zeros((N, self.nc), dtype=dt)
        for n in range(N):
            u, v = self.uvs.popleft()
            self.Us[n] = u
            self.Vs[n] = v
        self.Mtrue += np.dot(self.Us.T, self.Vs)

    def grab(self, transpose=False):
        self.aggregate()
        N = self.Us.shape[0]
        rbits = np.sign(np.random.random(size=(self.rank + 1, N)).astype(dt) - np.float32(0.5))
        self.Uhat, self.Vhat, self.Mult, self.Cx = apply_updates(
            self.Uhat, self.Vhat, self.Mult, self.Cx,
            self.Us, self.Vs, zerovar=self.zerovar,
            kappa_th=self.kappa_th,
            qb=self.qb, rbits=rbits)
        rtC = np.sqrt(self.Cx).reshape((-1,1))
        U = self.Mult[0] * self.Uhat[:-1] * rtC
        V = self.Mult[1] * self.Vhat[:-1] * rtC
        return U.T, V if transpose else V.T

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    eps = 1e-16
    grab_every = 100
    nr = 4096
    nc = 256
    r = 4

    M = np.zeros((nr, nc))
    uv = SKS(nr, nc, r, zerovar=0)

    for i in range(grab_every):
        np.random.seed(i)
        a = np.maximum(np.random.randn(nr) * 10**np.random.randn(), 0)
        d = np.random.randn(nc) * 10**np.random.randn()
        a = a.astype('f4')
        d = d.astype('f4')
        M += np.outer(a, d)
        uv.update(a, d)
    U, V = uv.grab()

    uv.reset()
    tt = 0
    for i in range(grab_every):
        np.random.seed(i)
        a = np.maximum(np.random.randn(nr) * 10**np.random.randn(), 0)
        d = np.random.randn(nc) * 10**np.random.randn()
        a = a.astype('f4')
        d = d.astype('f4')
        M += np.outer(a, d)
        t0 = time.time()
        uv.update(a, d)
        tt += time.time() - t0
    t0 = time.time()
    U, V = uv.grab()
    td = time.time() - t0

    Muv = np.dot(U, V.T)
    l2mag = np.linalg.norm(M)
    l2err = np.linalg.norm(M - Muv)
    cosx = cosine(M.flatten() + eps, Muv.flatten() + eps)
    print('Timing - Per Update: %5.1fus - Per Grab: %5.1fus - Total: %8.1fus'%(1e6 * tt/grab_every, 1e6 * td, 1e6 * (tt+td)))
    print('Relative L2 Error: %8.3f%% | Cosine Distance: %+.6f'%(100*l2err/l2mag, cosx))


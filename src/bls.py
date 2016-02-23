from __future__ import division, print_function
from time import time

import numpy as np
import matplotlib.pyplot as pl
from scipy.ndimage import median_filter as MF

from k2ps.blsf import bls

def fold(time, period, origo=0.0, shift=0.0):
    return ((time - origo)/period + shift) % 1.


class BLS(object):
    def __init__(self, time, flux, error, **kwargs):
        self.time = time
        self.flux = flux
        self.error = error

        self.fmin  = kwargs.get('fmin', 0.1)
        self.nf    = kwargs.get('nf',  1000)
        self.df    = kwargs.get('df',  1e-6)
        self.nbin  = kwargs.get('nbin', 1000)
        self.qmin  = kwargs.get('qmin', 0.01)
        self.qmax  = kwargs.get('qmax', 0.10)
        self.fmode = kwargs.get('fmode', 'log')
        self.pmul  = kwargs.get('pmul', np.ones(self.nf))
        self.pmean = kwargs.get('pmean', 'running_median')

        if 'period_range' in kwargs.keys():
            self.period_range = np.array(kwargs.get('period_range'))
            self.freq_range = np.flipud(1/self.period_range)
            self.fmin = self.freq_range[0]
            self.df = np.diff(self.freq_range) / self.nf

        else:
            self.freq_range   = np.array([self.fmin, self.fmin+self.nf*self.df])
            self.period_range = np.flipud(1/self.freq_range)
            
        if 'q_range' in kwargs.keys():
            self.qmin, self.qmax = kwargs.get('q_range')

        if self.fmode == 'linear':
            self.freqs = np.linspace(*self.freq_range, num=self.nf)
        else:
            self.freqs = np.logspace(*np.log10(self.freq_range), num=self.nf)
                

    def __call__(self):
        self.result = BLSResult(*bls.eebls(self.time, self.flux, self.error, self.freqs,
                                          self.nbin, self.qmin, self.qmax, self.pmul))
        self.result._p = self.period
        if self.result.in2 < self.result.in1:
            self.result.in2 += self.nbin
        return self.result
        

    @property
    def frequency(self):
        return self.freqs

    @property
    def period(self):
        return 1./self.frequency

    @property
    def sde(self):
        if self.result:
            return self.result.sde
        else:
            return np.zeros(self.nf)

    @property
    def phase(self):
        """Return the time folded and normalised using the best identified period"""
        return fold(self.time, self.result.bper, self.tc, 0.5) - 0.5

    @property
    def t1(self):
        """Returns the start-of-transit epoch"""
        return self.result.in1/self.nbin*self.result.bper + self.time[0]

    @property
    def t2(self):
        """Returns the end-of-transit epoch"""
        return self.result.in2/self.nbin*self.result.bper + self.time[0]

    @property
    def tc(self):
        """Returns the mid-transit epoch"""
        return 0.5*(self.result.in1+self.result.in2)/self.nbin*self.result.bper + self.time[0]
    
    @property
    def p1(self):
        """Returns the start-of-transit phase"""
        return fold(self.t1, self.result.bper, self.tc, 0.5) - 0.5

    @property
    def p2(self):
        """Returns the end-of-transit phase"""
        return fold(self.t2, self.result.bper, self.tc, 0.5) - 0.5

    @property
    def duration(self):
        """Returns the approximate transit duration"""
        return self.t2 - self.t1


class BLSResult(object):
    def __init__(self, p, bper, bpow, depth, qtran, in1, in2, pmean='running_median'):
        self.pmean = pmean
        self.p = p 
        self.bpow = bpow
        self.bsde = self.sde.max()
        self._depth = depth
        self._qtran = qtran
        self._in1 = in1
        self._in2 = in2

    def __str__(self):
        return 'Power {pw:8.6f}   sde {sde:6.3f}   Period {pr:6.3f}   Freq {fr:6.3f}   Depth {df:6.3f}   qtran {qt:5.3f}'.format(pr=self.bper, sde=self.bsde, fr=1/self.bper, pw=self.bpow, df=self.depth, qt=self.qtran)

    @property
    def sde(self):
        if self.pmean == 'mean':
            return (self.p - self.p.mean()) / self.p.std()
        elif self.pmean == 'running_median':
            power = self.p - MF(self.p, 200)
            return power / power.std()
        
    @property
    def bper(self):
        i = np.argmax(self.sde)
        return self._p[i]

    @property
    def depth(self):
        i = np.argmax(self.sde)
        return self._depth[i]

    @property
    def in1(self):
        i = np.argmax(self.sde)
        return self._in1[i]

    @property
    def in2(self):
        i = np.argmax(self.sde)
        return self._in2[i]

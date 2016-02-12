from __future__ import division

import math as mt
import numpy as np
import pandas as pd
import seaborn as sb
import pyfits as pf
import matplotlib.pyplot as plt
from IPython.display import clear_output

from glob import glob
from copy import copy
from os.path import join, basename, abspath
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import scoreatpercentile

from pytransit import MandelAgol as MA
from pytransit import Gimenez as GM
import scipy.optimize as so 
from exotk.utils.likelihood import ll_normal_es
from scipy.optimize import minimize

from mpi4py import MPI
from pybls import BLS
from pytransit import MandelAgol as MA
from pytransit.orbits_f import orbits as of
from exotk.utils.orbits import as_from_rhop, i_from_baew
from exotk.utils.misc_f import utilities as uf
from exotk.utils.misc import fold

from numpy.core.records import array as rarr
from numpy.lib.recfunctions import stack_arrays, merge_arrays

from numpy import (array, zeros, ones, ones_like, isfinite, median, nan, inf, 
                   sqrt, floor, diff, unique, concatenate, sin, pi, nanmin, nanmax)


from acor import acor

from matplotlib.pyplot import setp, subplots

def nanmedian(s):
    m = np.isfinite(s)
    return np.median(s[m])

## Array type definitions
## ----------------------
str_to_dt = lambda s: [tuple(t.strip().split()) for t in s.split(',')]
dt_lcinfo    = str_to_dt('epic u8, flux_median f8, flux_std f8, lnlike_constant f8, type a8,'
                         'acor_raw f8, acor_corr f8, acor_trp f8, acor_trt f8')
dt_blsresult = str_to_dt('sde f8, bls_zero_epoch f8, bls_period f8, bls_duration f8, bls_depth f8,'
                         'bls_radius_ratio f8, ntr u4')
dt_trfresult = str_to_dt('lnlike_transit f8, trf_zero_epoch f8, trf_period f8, trf_duration f8, trf_depth f8,'
                         'trf_radius_ratio f8, trf_semi_major_axis f8, trf_impact_parameter f8')
dt_varresult = str_to_dt('lnlike_sine f8, sine_amplitude f8')
dt_oeresult  = str_to_dt('lnlike_oe f8, oe_diff_k f8, oe_diff_a f8, oe_diff_b f8')
dt_poresult  = str_to_dt('po_lnlike_med f8, po_lnlike_std f8, po_lnlike_max f8, po_lnlike_min f8')


class TransitSearch(object):
    """K2 transit search and candidate vetting

    Overview
    --------
           The main idea is that we carry out a basic BLS search, continue with several model fits,
       and fit all the (possibly sensible) statistics to a random forest classifier.

       Transit search
         - BLS

       Transit fitting
         - basic transit fitting
         - even-odd transit fitting
         - calculation of per-orbit average log likelihoods
         - (Secondary eclipse search, not implemented)

       Sine fitting
         - fits a sine curve to the data to test for variability
    """

    def __init__(self, infile, inject=False):        
        d = pf.getdata(infile,1)
        m = (d.quality == 0) & (~(d.mflags_1 & 2**3).astype(np.bool)) & isfinite(d.flux_1)

        self.tm = MA(supersampling=12, nthr=1) 
        self.em = MA(supersampling=10, nldc=0, nthr=1)

        self.d = d
        self.epic   = int(basename(infile).split('_')[1])
        self.time   = d.time[m]
        tmin, tmax = np.nanmin(self.time), np.nanmax(self.time)
        self.flux   = (d.flux_1[m] 
                       - d.trend_t_1[m] + median(d.trend_t_1[m]) 
                       - d.trend_p_1[m] + median(d.trend_p_1[m]))
        self.mflux   = nanmedian(d.flux_1[m])
        self.flux   /= self.mflux
        self.flux_e  = d.error_1[m] / abs(self.mflux)

        self.flux_r  = d.flux_1[m] / self.mflux
        self.trend_t = d.trend_t_1[m] / self.mflux
        self.trend_p = d.trend_p_1[m] / self.mflux

        self.period_range = (0.01,0.98*(tmax-tmin))
        self.nbin = 800
        self.qmin = 0.001
        self.qmax = 0.2
        self.nf   = 15000
        
        self.bls =  BLS(self.time, self.flux, self.flux_e, period_range=self.period_range, 
                        nbin=self.nbin, qmin=self.qmin, qmax=self.qmax, nf=self.nf)
        
        try:
            ar,ac,ap,at = acor(self.flux_r)[0], acor(self.flux)[0], acor(self.trend_p)[0], acor(self.trend_t)[0]
        except RuntimeError:
            ar,ac,ap,at = nan,nan,nan,nan
        self.lcinfo = array((self.epic, self.mflux, self.flux.std(), nan, nan, ar, ac, ap, at), dtype=dt_lcinfo)

        self._rbls = None
        self._rtrf = None
        self._rvar = None
        self._rtoe = None
        self._rpol = None
        
        self._pv_bls = None
        self._pv_trf = None
        
        
    def create_transit_arrays(self):
        p   = self._rbls['bls_period']
        tc  = self._rbls['bls_zero_epoch']
        dur = max(0.15, self._rbls['bls_duration'])
        
        tid_arr  = np.round((self.time - tc) / p).astype(np.int)
        tid_arr -= tid_arr.min()
        tids     = unique(tid_arr)

        phase = p*(fold(self.time, p, tc, shift=0.5) - 0.5)
        pmask = abs(phase) < 5*dur

        self.times   = [self.time[tid_arr==tt]   for tt in unique(tids)]
        self.fluxes  = [self.flux[tid_arr==tt]   for tt in unique(tids)]
        self.flux_es = [self.flux_e[tid_arr==tt] for tt in unique(tids)]
        
        self.time_even   = concatenate(self.times[0::2])
        self.time_odd    = concatenate(self.times[1::2])
        self.flux_even   = concatenate(self.fluxes[0::2])
        self.flux_odd    = concatenate(self.fluxes[1::2])
        self.flux_e_even = concatenate(self.flux_es[0::2])
        self.flux_e_odd  = concatenate(self.flux_es[1::2])

        
    @property
    def result(self):
        return merge_arrays([self.lcinfo, self._rbls, self._rtrf,
                             self._rvar,  self._rtoe, self._rpol], flatten=True)
        
        
    def __call__(self):
        """Runs a BLS search, fits a transit, and fits an EB model"""
        self.run_bls()
        self.fit_transit()
        self.fit_variability()
        self.fit_even_odd()
        self.per_orbit_likelihoods()

    
    def run_bls(self):
        b = self.bls
        r = self.bls()
        self._rbls = array((r.bsde, b.tc, r.bper, b.t2-b.t1, r.depth, sqrt(r.depth),
                            floor(diff(self.time[[0,-1]])[0]/r.bper)), dt_blsresult)
        self._pv_bls = [r.bper, b.tc, sqrt(r.depth), 5, 0.1]
        self.create_transit_arrays()
        self.lcinfo['lnlike_constant'] = ll_normal_es(self.flux, ones_like(self.flux), self.flux_e)
    
    
    def fit_transit(self):
        def minfun(pv):
            if any(pv<=0) or (pv[3] <= 1) or (pv[4] > 1) or not (0.75<pv[0]<80): return inf
            return -ll_normal_es(self.flux, self.transit_model(pv), self.flux_e)
        
        mr = minimize(minfun, self._pv_bls, method='powell')
        lnlike, x = -mr.fun, mr.x
        self._rtrf = array((lnlike, x[1], x[0], of.duration_circular(x[0],x[3],mt.acos(x[4]/x[3])),
                            x[2]**2, x[2], x[3], x[4]), dt_trfresult)
        self._pv_trf = mr.x.copy()


    def per_orbit_likelihoods(self):
        pv = self._pv_trf
        lnl = array([ll_normal_es(f, self.transit_model(pv, t), e) / t.size
               for t,f,e in zip(self.times, self.fluxes, self.flux_es)])
        self._tr_lnlike_po = lnl
        self._tr_lnlike_med = lmed = median(lnl)
        self._tr_lnlike_std = lstd = lnl.std()
        self._tr_lnlike_max = lmax = (lnl.max() - lmed) / lstd
        self._tr_lnlike_min = lmin = (lnl.min() - lmed) / lstd
        self._rpol = array((lmed, lstd, lmax, lmin), dt_poresult)
        

    def fit_even_odd(self):
        def minfun(pv, time, flux, flux_e):
            if any(pv<=0) or (pv[3] <= 1) or (pv[4] > 1): return inf
            return -ll_normal_es(flux, self.transit_model(pv, time), flux_e)
        
        mr_even = minimize(minfun, self._pv_bls,
                           args=(self.time_even, self.flux_even, self.flux_e_even), method='powell')
        mr_odd  = minimize(minfun, self._pv_bls,
                           args=(self.time_odd, self.flux_odd, self.flux_e_odd), method='powell')
        pvd = abs(mr_even.x - mr_odd.x)
        self._rtoe = array((-mr_even.fun-mr_odd.fun, pvd[2], pvd[3], pvd[4]), dt_oeresult)

    
    def fit_variability(self):
        def minfun(pv, period, zero_epoch):
            if any(pv<0): return inf
            return -ll_normal_es(self.flux, self.sine_model(pv, 2*period, zero_epoch), self.flux_e)
        
        mr = minimize(minfun, [self.flux.std()],
                      args=(self._rbls['bls_period'],self._rbls['bls_zero_epoch']), method='powell')
        self._rvar = array((-mr.fun, mr.x), dt_varresult)

        
    ## Models
    ## ------
    def sine_model(self, pv, period, zero_epoch):
        return 1.+pv[0]*sin(2*pi/period*(self.time-zero_epoch) - 0.5*pi)
    
    def transit_model(self, pv, time=None):
        time = self.time if time is None else time
        _i = mt.acos(pv[4]/pv[3])
        return self.tm.evaluate(time, pv[2], [0.4, 0.1], pv[1], pv[0], pv[3], _i)
    
    def eclipse_model(self, shift, time=None):
        time = self.time if time is None else time
        pv = self._pv_trf
        _i = mt.acos(pv[4]/pv[3])
        return self.em.evaluate(time, pv[2], [], pv[1]+(0.5+shift)*pv[0], pv[0], pv[3], _i)


    def eclipse_likelihood(self, f, shift, time=None):
        return ll_normal_es(self.flux, (1-f)+f*self.eclipse_model(shift, time), self.flux_e)

    ## Plotting
    ## --------
    def plot_lc_time(self, ax=None):
        if not ax:
            fig,ax = subplots(1,1)
        ax.plot(self.time, self.flux_r, lw=1)
        ax.plot(self.time, self.trend_t+2*(np.percentile(self.flux_r, [99])[0]-1), lw=1)
        ax.plot(self.time, self.trend_p+4*(np.percentile(self.flux_r, [99])[0]-1), lw=1)
        ax.plot(self.time, self.flux+1.1*(self.flux_r.min()-1), lw=1)
        [ax.axvline(self.bls.tc+i*self._rbls['bls_period'], alpha=0.25, ls='--', lw=1) for i in range(15)]
        setp(ax,xlim=self.time[[0,-1]], xlabel='Time', ylabel='Normalised flux')
        return ax
    
    def plot_lc(self, ax=None, nbin=None):
        if not ax:
            fig,ax = subplots(1,1)
        nbin = nbin or self.nbin
        
        r = rarr(self.result)
        
        if self._pv_trf is None:
            bp,bf,be = uf.bin(self.bls.phase, self.flux, nbin)
            ax.plot(bp*r.bls_period, bf, drawstyle='steps-mid', lw=1)
        else:
            phase = fold(self.time, r.trf_period, r.trf_zero_epoch, shift=0.5) - 0.5
            bp,bf,be = uf.bin(phase, self.flux, nbin)
            ms = isfinite(bf)
            ax.plot(bp[ms]*r.trf_period, bf[ms], lw=1, marker='.', drawstyle='steps-mid')
            
            flux_m = self.transit_model(self._pv_trf)
            bp,bf,be = uf.bin(phase, flux_m, nbin)
            ms = isfinite(bf)
            ax.plot(bp[ms]*r.trf_period, bf[ms], 'k--', drawstyle='steps-mid')
            
        if self._rvar:
            flux_s = self.sine_model([self._rvar['sine_amplitude']], 2*r.bls_period, r.bls_zero_epoch)
            bp,bf,be = uf.bin(phase, flux_s, nbin)
            ax.plot(bp*r.bls_period, bf, 'k:', drawstyle='steps-mid')
            
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
        [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in [-0.5*r.bls_duration, 0.5*r.bls_duration]];
        setp(ax,xlim=r.bls_period*bp[[0,-1]], xlabel='Phase [d]', ylabel='Normalised flux')
        return ax

    
    def plot_even_odd_lc(self, ax=None, nbin=None):
        if ax is None:
            fig,ax = subplots(1,2, sharey=True, sharex=True)

        try:
            nbin = nbin or self.nbin
            res  = rarr(self.result)
            period, zero_epoch, duration = res.trf_period, res.trf_zero_epoch, res.trf_duration
            hdur = array([-0.5,0.5]) * duration

            for time,flux_o in ((self.time_even,self.flux_even),
                                (self.time_odd,self.flux_odd)):
        
                phase = fold(time, period, zero_epoch, shift=0.5, normalize=False) - 0.5*period
                flux_m = self.transit_model(self._pv_trf, time)
                bpd,bfd,bed = uf.bin(phase, flux_o, nbin)
                bpm,bfm,bem = uf.bin(phase, flux_m, nbin)
                pmask = abs(bpd) < 1.5*duration
                omask = pmask & isfinite(bfd)
                mmask = pmask & isfinite(bfm)
                ax[0].plot(bpd[omask], bfd[omask], marker='o')
                ax[1].plot(bpm[mmask], bfm[mmask], marker='o')

            [a.axvline(0, alpha=0.25, ls='--', lw=1) for a in ax]
            [[a.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur] for a in ax]
            setp(ax,xlim=3*hdur, xlabel='Phase [d]', ylim=(0.9998*nanmin(bfd[pmask]), 1.0002*nanmax(bfd[pmask])))
            setp(ax[0], ylabel='Normalised flux')
            setp(ax[1].get_yticklabels(), visible=False)
        except ValueError:
            pass
        return ax


    def plot_eclipse(self, ax=None):
        if ax is None:
            fig,ax = subplots(1,1)

        shifts = np.linspace(-0.2,0.2, 500)
        ll0 = self.eclipse_likelihood(0.0, 0.0)
        ll1 = array([self.eclipse_likelihood(0.01, shift) for shift in shifts])

        ax.plot(0.5+shifts, ll1-ll0)
        setp(ax, xlim=(0.3,0.7), ylabel='ln likelihood', xlabel='Phase shift')
        return ax


    def plot_lnlike(self, ax=None):
        if ax is None:
            fig,ax = subplots(1,1)
        ax.plot(self._tr_lnlike_po, marker='.', markersize=20)
        ax.axhline(self._tr_lnlike_med, ls='--', alpha=0.6)
        [ax.axhline(self._tr_lnlike_med+s*self._tr_lnlike_std, ls=':', alpha=a)
         for s,a in zip((-1,1,-2,2,-3,3), (0.6,0.6,0.4,0.4,0.2,0.2))]
        setp(ax, xlim=(0,len(self._tr_lnlike_po)-1), ylabel='ln likelihood', xlabel='Orbit number')
        return ax

    
    def plot_sde(self, ax=None):
        if not ax:
            fig,ax = subplots(1,1)
        r = rarr(self.result)
            
        ax.plot(self.bls.period, self.bls.result.sde, drawstyle='steps-mid')
        ax.axvline(r.bls_period, alpha=0.25, ls='--', lw=1)
        ax.text(0.97,0.87, 'Best period: {:4.2f} days'.format(float(r.bls_period)),
                ha='right', transform=ax.transAxes)
        setp(ax,xlim=self.bls.period[[-1,0]], xlabel='Period [d]', ylabel='SDE')
        return ax

from __future__ import division

import math as mt
import numpy as np
import pandas as pd
import seaborn as sb
import pyfits as pf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functools import wraps

from glob import glob
from copy import copy
from os.path import join, basename, abspath
from collections import namedtuple
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import scoreatpercentile

from pyfits.card import Undefined
from pytransit import MandelAgol as MA
from pytransit import Gimenez as GM
import scipy.optimize as so 
from exotk.utils.likelihood import ll_normal_es
from scipy.optimize import minimize

from mpi4py import MPI
from pytransit.orbits_f import orbits as of
from exotk.utils.orbits import as_from_rhop, i_from_baew
from exotk.utils.misc_f import utilities as uf
from exotk.utils.misc import fold

from numpy.core.records import array as rarr
from numpy.lib.recfunctions import stack_arrays, merge_arrays

from numpy import (array, zeros, ones, ones_like, isfinite, median, nan, inf, abs,
                   sqrt, floor, diff, unique, concatenate, sin, pi, nanmin, nanmax,
                   log, exp, argsort)

from acor import acor

from matplotlib.pyplot import setp, subplots
from bls import BLS

def nanmedian(s):
    m = np.isfinite(s)
    return np.median(s[m])

from scipy.constants import G
from exotk.utils.orbits import d_s

def rho_from_pas(period,a):
    return 1e-3*(3*pi)/G * a**3 * (period*d_s)**-2

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
dt_ecresult  = str_to_dt('ec_lnlratio f8, shift f8')

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

    def __init__(self, infile, inject=False, **kwargs):        
        self.d = d = pf.getdata(infile,1)
        m = (d.quality == 0) & (~(d.mflags_1 & 2**3).astype(np.bool)) & isfinite(d.flux_1)
        # m = isfinite(d.flux_1) & (~(d.mflags_1 & 2**3).astype(np.bool))
        self.badcads = kwargs.get('bad_cads',np.array([]))
        self.cadence = d.cadence

        for cad in self.badcads:
            m[self.cadence==cad] = 0

        self.Kp = pf.getval(infile,'kepmag')
        self.Kp = self.Kp if not isinstance(self.Kp, Undefined) else nan

        self.tm = MA(supersampling=12, nthr=1) 
        self.em = MA(supersampling=10, nldc=0, nthr=1)

        self.epic   = int(basename(infile).split('_')[1])
        self.time   = d.time[m]
        self.flux   = (d.flux_1[m] 
                       - d.trend_t_1[m] + nanmedian(d.trend_t_1[m]) 
                       - d.trend_p_1[m] + nanmedian(d.trend_p_1[m]))
        self.mflux   = nanmedian(self.flux)
        self.flux   /= self.mflux
        self.flux_e  = d.error_1[m] / abs(self.mflux)

        self.flux_r  = d.flux_1[m] / self.mflux
        self.trend_t = d.trend_t_1[m] / self.mflux
        self.trend_p = d.trend_p_1[m] / self.mflux

        self.period_range = kwargs.get('period_range', (0.7,0.98*(self.time.max()-self.time.min())))
        self.nbin = kwargs.get('nbin',900)
        self.qmin = kwargs.get('qmin',0.002)
        self.qmax = kwargs.get('qmax',0.115)
        self.nf   = kwargs.get('nfreq',10000)

        
        self.bls =  BLS(self.time, self.flux, self.flux_e, period_range=self.period_range, 
                        nbin=self.nbin, qmin=self.qmin, qmax=self.qmax, nf=self.nf, pmean='running_median')

        def ndist(p=0.302):
            return 1.-2*abs(((self.bls.period-p)%p)/p-0.5)

        def cmask(s=0.05):
            return 1.-np.exp(-ndist()/s)

        self.bls.pmul = cmask()

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
        self._recl = None

        ## Transit fit pv [k u t0 p a i]
        self._pv_bls = None
        self._pv_trf = None
        
        self.period = None
        self.zero_epoch = None
        self.duration = None


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
                             self._rvar,  self._rtoe, self._rpol,
                             self._recl], flatten=True)
        
        
    def __call__(self):
        """Runs a BLS search, fits a transit, and fits an EB model"""
        self.run_bls()
        self.fit_transit()
        self.fit_variability()
        self.fit_even_odd()
        self.per_orbit_likelihoods()
        self.test_eclipse()
    

    def run_bls(self):
        b = self.bls
        r = self.bls()
        self._rbls = array((b.bsde, b.tc, b.bper, b.duration, b.depth, sqrt(b.depth),
                            floor(diff(self.time[[0,-1]])[0]/b.bper)), dt_blsresult)
        self._pv_bls = [b.bper, b.tc, sqrt(b.depth), as_from_rhop(2.5, b.bper), 0.1]
        self.create_transit_arrays()
        self.lcinfo['lnlike_constant'] = ll_normal_es(self.flux, ones_like(self.flux), self.flux_e)
        self.period = b.bper
        self.zero_epoch = b.tc
        self.duration = b.duration

    
    def fit_transit(self):
        def minfun(pv):
            if any(pv<=0) or (pv[3] <= 1) or (pv[4] > 1) or not (0.75<pv[0]<80) or abs(pv[0]-pbls) > 1.: return inf
            return -ll_normal_es(self.flux, self.transit_model(pv), self.flux_e)
        
        pbls = self._pv_bls[0]
        mr = minimize(minfun, self._pv_bls, method='powell')
        lnlike, x = -mr.fun, mr.x
        self._rtrf = array((lnlike, x[1], x[0], of.duration_circular(x[0],x[3],mt.acos(x[4]/x[3])),
                            x[2]**2, x[2], x[3], x[4]), dt_trfresult)
        self._pv_trf = mr.x.copy()
        self.period = x[0]
        self.zero_epoch = x[1]
        self._rtrf['trf_duration']


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
            dummy = []
            for j in range(4):
                dummy.append(-ll_normal_es(self.flux, self.sine_model(pv, j*2*period, zero_epoch), self.flux_e))
            return np.nanmin(dummy)#-ll_normal_es(self.flux, self.sine_model(pv, 2*period, zero_epoch), self.flux_e)
        
        mr = minimize(minfun, [self.flux.std()],
                      args=(self._rbls['bls_period'],self._rbls['bls_zero_epoch']), method='powell')
        self._rvar = array((-mr.fun, mr.x), dt_varresult)


    def test_eclipse(self):
        self.ec_shifts  = shifts = np.linspace(-0.2,0.2, 500)
        self.ec_ll0     = ll0 = self.eclipse_likelihood(0.0, 0.0)
        self.ec_ll1     = ll1 = array([self.eclipse_likelihood(0.01, shift) for shift in shifts])
        ll1_max = ll1.max()
        self.ec_lnratio = lnlratio = log(exp(ll1-ll1_max).mean())+ll1_max - ll0
        self._recl = array((self.ec_lnratio, self.ec_shifts[np.argmax(self.ec_lnratio)]), dt_ecresult)
        
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
    def bplot(plotf):
        @wraps(plotf)
        def wrapper(self, ax=None, *args, **kwargs):
            if ax is None:
                fig, ax = subplots(1,1)

            try:
                plotf(self, ax, **kwargs)
            except ValueError:
                pass
            return ax
        return wrapper

    @bplot
    def plot_lc_time(self, ax=None):
        ax.plot(self.time, self.flux_r, lw=1)
        ax.plot(self.time, self.trend_t+2*(np.percentile(self.flux_r, [99])[0]-1), lw=1)
        ax.plot(self.time, self.trend_p+4*(np.percentile(self.flux_r, [99])[0]-1), lw=1)
        ax.plot(self.time, self.flux+1.1*(self.flux_r.min()-1), lw=1)
        [ax.axvline(self.bls.tc+i*self._rbls['bls_period'], alpha=0.25, ls='--', lw=1) for i in range(35)]
        setp(ax,xlim=self.time[[0,-1]], xlabel='Time', ylabel='Normalised flux')


    @bplot
    def plot_lc(self, ax=None, nbin=None):
        nbin = nbin or self.nbin        
        r = rarr(self.result)
        period, t0, trdur = self._rbls['bls_period'], self._rbls['bls_zero_epoch'], self._rbls['bls_duration'] #r.trf_period, r.trf_zero_epoch, r.trf_duration
        phase = period*(fold(self.time, period, t0, shift=0.5) - 0.5)

        bp,bfo,beo = uf.bin(phase, self.flux, nbin)
        bp,bfm,bem = uf.bin(phase, self.transit_model(self._pv_trf), nbin)
        mo,mm = isfinite(bfo), isfinite(bfm)
        ax.plot(bp[mo], bfo[mo], '.')
        ax.plot(bp[mm], bfm[mm], 'k--', drawstyle='steps-mid')
        if self._rvar:
            flux_s = self.sine_model([self._rvar['sine_amplitude']], 2*period, t0)
            bp,bf,be = uf.bin(phase, flux_s, nbin)
            ax.plot(bp, bf, 'k:', drawstyle='steps-mid')
            
        setp(ax,xlim=bp[[0,-1]], xlabel='Phase [d]', ylabel='Normalised flux')
        setp(ax.get_yticklabels(), visible=False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)


    @bplot
    def plot_even_odd_lc(self, ax=None, nbin=None):
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
 

    @bplot
    def plot_transits(self, ax=None):
        offset = 1.1*scoreatpercentile([f.ptp() for f in self.fluxes], 95)
        twodur = 24*2*self.duration
        for i,(time,flux) in enumerate(zip(self.times, self.fluxes)[:10]):
            phase = 24*(fold(time, self.period, self.zero_epoch, 0.5, normalize=False) - 0.5*self.period)
            sids  = argsort(phase)
            phase, flux = phase[sids], flux[sids]
            pmask = abs(phase) < 2.5*twodur
            if any(pmask):
                ax.plot(phase[pmask], flux[pmask]+i*offset, marker='o')

        setp(ax, xlim=(-twodur,twodur), xlabel='Phase [h]', yticks=[])


    @bplot
    def plot_transit_fit(self, ax=None):
        res  = rarr(self.result)
        period, zero_epoch, duration = res.trf_period, res.trf_zero_epoch, res.trf_duration
        hdur = 24*duration*array([-0.5,0.5])

        flux_m = self.transit_model(self._pv_trf)
        phase = 24*(fold(self.time, period, zero_epoch, 0.5, normalize=False) - 0.5*period)
        sids = argsort(phase)
        phase = phase[sids]
        pmask = abs(phase) < 2*24*duration
        flux_m = flux_m[sids]
        flux_o = self.flux[sids]
        ax.plot(phase[pmask], flux_o[pmask], '.')
        ax.plot(phase[pmask], flux_m[pmask], 'k')
        ax.text(2.5*hdur[0], flux_m.min(), '{:6.4f}'.format(flux_m.min()), size=7, va='center', bbox=dict(color='white'))
        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
        [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur]
        setp(ax, xlim=3*hdur, xlabel='Phase [h]', ylabel='Normalised flux')
        setp(ax.get_yticklabels(), visible=False)


    @bplot
    def plot_fit_and_eo(self, ax=None, nbin=None):
        nbin = nbin or self.nbin
        res  = rarr(self.result)
        period, zero_epoch, duration = res.trf_period, res.trf_zero_epoch, res.trf_duration
        hdur = 24*duration*array([-0.5,0.5])

        self.plot_transit_fit(ax[0])

        for time,flux_o in ((self.time_even,self.flux_even),
                            (self.time_odd,self.flux_odd)):

            phase = 24*(fold(time, period, zero_epoch, shift=0.5, normalize=False) - 0.5*period)
            bpd,bfd,bed = uf.bin(phase, flux_o, nbin)
            pmask = abs(bpd) < 2*24*duration
            omask = pmask & isfinite(bfd)
            ax[1].plot(bpd[omask], bfd[omask], marker='o', ms=2)

        [a.axvline(0, alpha=0.25, ls='--', lw=1) for a in ax]
        [[a.axvline(24*hd, alpha=0.25, ls='-', lw=1) for hd in hdur] for a in ax]
        setp(ax[1],xlim=3*hdur, xlabel='Phase [h]')
        setp(ax[1].get_yticklabels(), visible=False)
        ax[1].get_yaxis().get_major_formatter().set_useOffset(False)



    @bplot
    def plot_eclipse(self, ax=None):
        shifts = np.linspace(-0.2,0.2, 500)
        ll0 = self.eclipse_likelihood(0.0, 0.0)
        ll1 = array([self.eclipse_likelihood(0.01, shift) for shift in shifts])
        ll1_max = ll1.max()
        lnlratio = log(exp(ll1-ll1_max).mean())+ll1_max - ll0

        ax.plot(0.5+shifts, ll1-ll1.min())
        ax.text(0.02, 0.9, 'ln (P$_e$/P$_f$) = {:3.2f}'.format(lnlratio), size=7, transform=ax.transAxes)
        setp(ax, xlim=(0.3,0.7), xticks=(0.35,0.5,0.65), ylabel='$\Delta$ ln likelihood', xlabel='Phase shift')
        setp(ax.get_yticklabels(), visible=False)
        ax.set_title('Secondary eclipse')
        for t,l in zip(ax.get_yticks(),ax.get_yticklabels()):
            if l.get_text():
                ax_ec.text(0.31, t, l.get_text())

    @bplot
    def plot_lnlike(self, ax=None):
        ax.plot(self._tr_lnlike_po, marker='.', markersize=10)
        ax.axhline(self._tr_lnlike_med, ls='--', alpha=0.6)
        [ax.axhline(self._tr_lnlike_med+s*self._tr_lnlike_std, ls=':', alpha=a)
         for s,a in zip((-1,1,-2,2,-3,3), (0.6,0.6,0.4,0.4,0.2,0.2))]
        setp(ax, xlim=(0,len(self._tr_lnlike_po)-1), ylabel='ln likelihood', xlabel='Orbit number')
        setp(ax.get_yticklabels(), visible=False)


    @bplot
    def plot_sde(self, ax=None):
        r = rarr(self.result)
        ax.plot(self.bls.period, self.bls.sde, drawstyle='steps-mid')
        ax.axvline(r.bls_period, alpha=0.25, ls='--', lw=1)
        setp(ax,xlim=self.bls.period[[-1,0]], xlabel='Period [d]', ylabel='SDE', ylim=(-3,11))
        [ax.axhline(i, c='k', ls='--', alpha=0.5) for i in [0,5,10]]
        [ax.text(self.bls.period.max()-1,i-0.5,i, va='top', ha='right', size=7) for i in [5,10]]
        ax.text(0.5, 0.88, 'BLS search', va='top', ha='center', size=8, transform=ax.transAxes)
        setp(ax.get_yticklabels(), visible=False)


    @bplot
    def plot_info(self, ax):
        res  = rarr(self.result)
        t0,p,tdur,tdep,rrat = res.trf_zero_epoch[0], res.trf_period[0], res.trf_duration[0], res.trf_depth[0], 0
        a = res.trf_semi_major_axis[0]
        ax.text(0.0,1.0, 'EPIC {:9d}'.format(self.epic), size=12, weight='bold', va='top', transform=ax.transAxes)
        ax.text(0.0,0.83, ('SDE\n'
                          'Kp\n'
                          'Zero epoch\n'
                          'Period [d]\n'
                          'Transit depth\n'
                          'Radius ratio\n'
                          'Transit duration [h]\n'
                          'Impact parameter\n'
                          'Stellar density'), size=9, va='top')
        ax.text(0.97,0.83, ('{:9.3f}\n{:9.3f}\n{:9.3f}\n{:9.3f}\n{:9.5f}\n'
                           '{:9.4f}\n{:9.3f}\n{:9.3f}\n{:0.3f}').format(res.sde[0],self.Kp,t0,p,tdep,sqrt(tdep),24*tdur,
                                                                        res.trf_impact_parameter[0], rho_from_pas(p,a)),
                size=9, va='top', ha='right')
        sb.despine(ax=ax, left=True, bottom=True)
        setp(ax, xticks=[], yticks=[])
 

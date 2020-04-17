from collections import namedtuple
from functools import wraps
from pathlib import Path
from typing import Iterable

import astropy.units as u
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from celerite import GP
from celerite.terms import SHOTerm
from matplotlib.pyplot import subplots, setp
from numpy import linspace, array, argmax, ones, floor, log, pi, sin, argsort
from pytransit import BaseLPF, QuadraticModelCL
from pytransit.lpf.loglikelihood import CeleriteLogLikelihood
from pytransit.orbits import epoch
from pytransit.utils.misc import fold
from scipy.optimize import minimize

TransitFitResult = namedtuple('TransitFitResult', 'parameters time phase obs model')


class BLSResult:
    def __init__(self, bls, ts):
        self.bls = bls
        self.ts = ts
        i = argmax(bls.depth_snr)
        self.periods = self.bls.period
        self.depth_snr = self.bls.depth_snr
        self.period = bls.period[i].value
        self.depth = bls.depth[i]

        t0 = bls.transit_time[i].value
        ep = epoch(self.ts.time.min(), t0, self.period)
        self.t0 = t0 + ep * self.period


class SearchLPF(BaseLPF):
    def _init_lnlikelihood(self):
        self._add_lnlikelihood_model(CeleriteLogLikelihood(self))


class TransitSearch:
    def __init__(self, pmin: float = 0.25, pmax: float = 15., nper: int = 10000, excluded_ranges: Iterable = (),
                 nsamples: int = 1, exptime: float = 0.0):

        self.pmin = pmin
        self.pmax = pmax
        self.nper = nper
        self.excluded_ranges = excluded_ranges
        self.nsamples = nsamples
        self.exptime = exptime

        self.periods = linspace(pmin, pmax, nper)
        self.durations = array([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) / 24

        self.lpfs = {}

        self.time = None
        self.phase = None
        self.flux = None
        self.ferr = None
        self.bls = None

        self.period = None
        self.zero_epoch = None
        self.duration = None

    def run(self):
        print("Running BLS")
        self.run_bls()
        print("Fitting all transits")
        self.tf_both = self.fit_transit(mode='all')
        print("Fitting odd transits")
        self.tf_odd = self.fit_transit(mode='odd')
        print("Fitting even transits")
        self.tf_even = self.fit_transit(mode='even')

    def read_data(self, filename: Path):
        time, flux, ferr = self._reader(filename)
        self._setup_data(time, flux, ferr)

    def run_bls(self):
        self.bls_result = BLSResult(self.bls.power(self.periods, self.durations, objective='snr'), self)
        self.phase = fold(self.time, self.bls_result.period, self.bls_result.t0, 0.5) * self.bls_result.period

    def _setup_data(self, time, flux, ferr):
        self.time = time
        self.flux = flux
        self.ferr = ferr
        self.bls = BoxLeastSquares(self.time * u.day, self.flux, self.ferr)

    def _reader(self, filename: Path):
        raise NotImplementedError

    def fit_transit(self, niter: int = 1000, npop: int = 30, mode: str = 'all', initialize_only: bool = False):
        epochs = epoch(self.time, self.bls_result.t0, self.bls_result.period)
        if mode == 'all':
            mask = ones(self.time.size, bool)
        elif mode == 'even':
            mask = epochs % 2 == 0
        elif mode == 'odd':
            mask = epochs % 2 == 1
        else:
            raise NotImplementedError

        time = self.time[mask]
        flux = self.flux[mask]

        tref = floor(time.min())
        tm = QuadraticModelCL()
        lpf = BaseLPF('transit_fit', ['Kepler'], times=time, fluxes=flux, tm=tm,
                      nsamples=self.nsamples, exptimes=self.exptime, tref=tref)
        self.lpfs[f"transit_{mode}"] = lpf

        lpf.set_prior('tc', 'NP', self.bls_result.t0, 0.01)
        lpf.set_prior('p', 'NP', self.bls_result.period, 0.01)
        lpf.set_prior('k2', 'NP', 0.1 * self.bls_result.depth, 2 * self.bls_result.depth)

        if initialize_only:
            return
        else:
            lpf.optimize_global(niter=niter, npop=npop, use_tqdm=False, plot_convergence=False)
            lpf.sample_mcmc(500, repeats=2, use_tqdm=False)
            df = lpf.posterior_samples(derived_parameters=True)
            df = pd.DataFrame((df.median(), df.std()), index='med err'.split())
            pv = lpf.posterior_samples(derived_parameters=False).median().values
            phase = fold(time, pv[1], pv[0], 0.5) * pv[1] - 0.5 * pv[1]
            model = lpf.flux_model(pv)
            return TransitFitResult(df, time, phase, flux, model)

    def fit_gp_periodicity(self, period: float = None):
        period = period if period is not None else self.bls_result.period

        gp = GP(SHOTerm(log(self.flux.var()), log(10), log(2 * pi / period)), mean=1.)
        gp.freeze_parameter('kernel:log_omega0')
        gp.compute(self.time, yerr=self.ferr)

        def minfun(pv):
            gp.set_parameter_vector(pv)
            gp.compute(self.time, yerr=self.ferr)
            return -gp.log_likelihood(self.flux)

        res = minimize(minfun, gp.get_parameter_vector(), jac=False, method='powell')
        self.gp_result = res
        self.gp_periodicity = res.x

    def sine_model(self, pv, period, zero_epoch):
        return 1. + pv[0] * sin(2 * pi / period * (self.time - zero_epoch) - 0.5 * pi)

    # def eclipse_model(self, shift, time=None):
    #     time = self.time if time is None else time
    #     pv = self._pv_trf
    #     _i = mt.acos(pv[4] / pv[3])
    #     return self.em.evaluate(time, pv[2], [], pv[1] + (0.5 + shift) * pv[0], pv[0], pv[3], _i)
    #
    # def eclipse_likelihood(self, f, shift, time=None):
    #     return lnlike_normal_s(self.flux, (1 - f) + f * self.eclipse_model(shift, time), self.flux_e)

    def bplot(plotf):
        @wraps(plotf)
        def wrapper(self, ax=None, *args, **kwargs):
            if ax is None:
                fig, ax = subplots(1, 1)
            else:
                fig, ax = None, ax

            try:
                plotf(self, ax, **kwargs)
            except ValueError:
                pass
            return fig

        return wrapper

    @bplot
    def plot_snr(self, ax=None):
        r = self.bls_result
        ax.plot(r.periods, r.depth_snr, drawstyle='steps-mid')
        ax.axvline(r.period, alpha=0.5, ls='--', lw=2)
        setp(ax, xlabel='Period [d]', ylabel='Depth SNR')
        ax.autoscale(axis='x', tight=True)

    @bplot
    def plot_transit_fit(self, ax=None):
        p = self.tf_both.parameters
        zero_epoch, period, duration = p[['tc', 'p', 't14']].iloc[0].copy()

        if duration >= (0.25 / 24.):
            hdur = duration * array([-0.5, 0.5])
        else:
            hdur = 24 * array([-0.25, 0.25])
            duration = 0.5

        flux_m = self.tf_both.model
        phase = self.tf_both.phase
        sids = argsort(phase)
        phase = phase[sids]
        pmask = abs(phase) < 2 * 24 * duration
        flux_m = flux_m[sids]
        flux_o = self.tf_both.obs[sids]
        ax.plot(phase[pmask], flux_o[pmask], '.')
        ax.plot(phase[pmask], flux_m[pmask], 'k')
        ax.text(2.5 * hdur[0], flux_m.min(), '{:6.4f}'.format(flux_m.min()), size=7, va='center',
                bbox=dict(color='white'))
        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
        [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur]
        fluxrange = flux_o.max() - flux_o.min()
        setp(ax, xlim=3 * hdur, ylim=[flux_o.min() - 0.05 * fluxrange, flux_o.max() + 0.05 * fluxrange],
             xlabel='Phase [h]', ylabel='Normalised flux')
        setp(ax.get_yticklabels(), visible=False)


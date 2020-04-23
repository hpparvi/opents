from collections import namedtuple
from functools import wraps
from pathlib import Path
from typing import Iterable
from logging import info, warning, error

import astropy.units as u
import pandas as pd
from astropy.stats import sigma_clipped_stats
from astropy.timeseries import BoxLeastSquares, LombScargle
from celerite import GP
from celerite.terms import SHOTerm
from matplotlib.pyplot import subplots, setp, errorbar
from numpy import linspace, array, argmax, ones, floor, log, pi, sin, argsort, unique, median, zeros, atleast_2d, \
    ndarray, squeeze, percentile, inf, asarray
from pytransit import BaseLPF, QuadraticModelCL, QuadraticModel
from pytransit.lpf.loglikelihood import CeleriteLogLikelihood
from pytransit.orbits import epoch
from pytransit.param import LParameter, UniformPrior as UP
from pytransit.utils.misc import fold
from pytransit.lpf.tesslpf import downsample_time

from scipy.optimize import minimize

from typing import Optional
from numba import njit


@njit(fastmath=True)
def sine_model(time, period, phase, amplitudes):
    npv = period.size
    npt = time.size
    nsn = amplitudes.shape[1]

    bl = zeros((npv, npt))
    for i in range(npv):
        for j in range(nsn):
            bl[i, :] += amplitudes[i, j] * sin(2 * pi * (time - phase[i] * period[i]) / (period[i] / (j + 1)))
    return bl


class SineBaseline:
    def __init__(self, lpf, name: str = 'sinbl', n: int = 1, lcids=None):
        self.name = name
        self.lpf = lpf
        self.n = n

        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising LinearModelBaseline.')

        self.init_data(lcids)
        self.init_parameters()

    def init_data(self, lcids=None):
        self.time = self.lpf.timea - self.lpf._tref

    def init_parameters(self):
        """Baseline parameter initialisation.
        """
        fptp = self.lpf.ofluxa.ptp()
        bls = []
        bls.append(LParameter(f'c_sin', f'sin phase', '', UP(0.0, 1.0), bounds=(0, 1)))
        for i in range(self.n):
            bls.append(LParameter(f'a_sin_{i}', f'sin {i} amplitude', '', UP(0, fptp), bounds=(0, inf)))
        self.lpf.ps.thaw()
        self.lpf.ps.add_global_block(self.name, bls)
        self.lpf.ps.freeze()
        self.pv_slice = self.lpf.ps.blocks[-1].slice
        self.pv_start = self.lpf.ps.blocks[-1].start
        setattr(self.lpf, f"_sl_{self.name}", self.pv_slice)
        setattr(self.lpf, f"_start_{self.name}", self.pv_start)

    def __call__(self, pvp, bl: Optional[ndarray] = None):
        pvp = atleast_2d(pvp)
        if bl is None:
            bl = ones((pvp.shape[0], self.time.size))
        else:
            bl = atleast_2d(bl)

        bl += sine_model(self.time,
                         period=pvp[:, 1],
                         phase=pvp[:, self.pv_start],
                         amplitudes=pvp[:, self.pv_start + 1:])
        return squeeze(bl)



class BLSResult:
    def __init__(self, bls, ts):
        self.bls = bls
        self.ts = ts
        i = argmax(bls.depth_snr)
        self.periods = self.bls.period
        self.depth_snr = self.bls.depth_snr
        self.snr = self.depth_snr[i]
        self.period = bls.period[i].value
        self.depth = bls.depth[i]
        self.duration = bls.duration[i].value

        t0 = bls.transit_time[i].value
        ep = epoch(self.ts.time.min(), t0, self.period)
        self.t0 = t0 + ep * self.period


class LombScargleResult:
    def __init__(self, periods, powers):
        self.periods = asarray(periods)
        self.powers = asarray(powers)
        imax = argmax(powers)
        self.period = self.periods[imax]
        self.power = self.powers[imax]

    def __repr__(self):
        return f"LombScargleResult({self.periods.__repr__()}, {self.powers.__repr__()})"

    def __str__(self):
        return f"Lomb-Scargle Result\n    Period {self.period:>8.4f}\n    Power  {self.power:>8.4f}"


TransitFitResult = namedtuple('TransitFitResult', 'parameters time epoch phase obs model transit lnldiff')


class SearchLPF(BaseLPF):
    # def _init_lnlikelihood(self):
    #    self._add_lnlikelihood_model(CeleriteLogLikelihood(self))

    def _init_baseline(self):
        self._add_baseline_model(SineBaseline(self, n=1))


class TransitSearch:
    def __init__(self, pmin: float = 0.25, pmax: float = 15., nper: int = 10000, snr_limit: float = 3,
                 nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):

        self.name: Optional[str] = None
        self.pmin: float = pmin
        self.pmax: float = pmax
        self.nper: int  = nper
        self.nsamples: int = nsamples
        self.exptime: float = exptime
        self.use_tqdm: bool = use_tqdm
        self.use_opencl: bool = use_opencl
        self.snr_limit: float = snr_limit

        self.bls = None
        self.transit_fits = {}
        self.transit_fit_results = {}

        self.time: Optional[ndarray] = None
        self.flux: Optional[ndarray] = None
        self.ferr: Optional[ndarray] = None
        self.phase: Optional[ndarray] = None

        self.period: Optional[float] = None        # Orbital period
        self.zero_epoch: Optional[float] = None    # Zero epoch
        self.duration: Optional[float] = None      # Transit duration in days
        self.snr: Optional[float] = None           # Transit signal-to-noise ratio

    def save_fits(self, savedir: Path):
        raise NotImplementedError

    @property
    def basename(self):
        raise NotImplementedError

    def run(self):
        info("Running BLS")
        self.run_bls()
        info("Running Lomb-Scargle")
        self.run_lomb_scargle()
        info("Fitting all transits")
        self.tf_both = self.fit_transit(mode='all')
        info("Fitting odd transits")
        self.tf_odd = self.fit_transit(mode='odd')
        info("Fitting even transits")
        self.tf_even = self.fit_transit(mode='even')

    def update_ephemeris(self, zero_epoch, period, duration):
        self.zero_epoch = zero_epoch
        self.period = period
        self.duration = duration
        self.phase = fold(self.time, self.period, self.zero_epoch, 0.5) * self.period

    def read_data(self, filename: Path):
        time, flux, ferr = self._reader(filename)
        self._setup_data(time, flux, ferr)

    def run_bls(self):
        periods = linspace(self.pmin, self.pmax, self.nper)
        durations = array([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) / 24
        self.bls = BoxLeastSquares(self.time * u.day, self.flux, self.ferr)
        self.bls_result = r = BLSResult(self.bls.power(periods, durations, objective='snr'), self)
        self.update_ephemeris(r.t0, r.period, r.duration)
        self.snr = self.bls_result.snr

    def run_lomb_scargle(self):
        self.ls = LombScargle(self.time, self.flux)
        freq = linspace(1 / self.pmax, 1 / self.pmin, 1000)
        power = self.ls.power(freq)
        self.ls_result = LombScargleResult(1 / freq, power)

    def _setup_data(self, time, flux, ferr):
        self.time = time
        self.flux = flux
        self.ferr = ferr

    def _reader(self, filename: Path):
        raise NotImplementedError

    def fit_transit(self, npop: int = 30, de_niter: int = 1000, mcmc_niter: int = 100, mcmc_repeats: int = 2,
                    mode: str = 'all', initialize_only: bool = False):
        epochs = epoch(self.time, self.bls_result.t0, self.bls_result.period)
        if mode == 'all':
            mask = ones(self.time.size, bool)
        elif mode == 'even':
            mask = epochs % 2 == 0
        elif mode == 'odd':
            mask = epochs % 2 == 1
        else:
            raise NotImplementedError

        epochs = epochs[mask]
        time = self.time[mask]
        flux = self.flux[mask]

        tref = floor(time.min())
        tm = QuadraticModelCL() if self.use_opencl else QuadraticModel(interpolate=False)
        lpf = SearchLPF('transit_fit', ['Kepler'], times=time, fluxes=flux, tm=tm,
                        nsamples=self.nsamples, exptimes=self.exptime, tref=tref)
        self.transit_fits[mode] = lpf

        if mode == 'all':
            lpf.set_prior('tc', 'NP', self.zero_epoch, 0.01)
            lpf.set_prior('p', 'NP', self.period, 0.01)
            lpf.set_prior('k2', 'UP', 0.5 * self.bls_result.depth, 2 * self.bls_result.depth)
        else:
            priorp = self.transit_fit_results['all'].parameters
            lpf.set_prior('tc', 'NP', priorp.tc.med, priorp.tc.err)
            lpf.set_prior('p', 'NP', priorp.p.med, priorp.p.err)
            lpf.set_prior('k2', 'UP', 0.5 * priorp.k2.med, 2 * priorp.k2.med)
            lpf.set_prior('q1_Kepler', 'NP', priorp.q1_Kepler.med, priorp.q1_Kepler.err)
            lpf.set_prior('q2_Kepler', 'NP', priorp.q2_Kepler.med, priorp.q2_Kepler.err)

        if initialize_only:
            return
        else:
            lpf.optimize_global(niter=de_niter, npop=npop, use_tqdm=self.use_tqdm, plot_convergence=False)
            lpf.sample_mcmc(mcmc_niter, repeats=mcmc_repeats, use_tqdm=self.use_tqdm)
            df = lpf.posterior_samples(derived_parameters=True)
            df = pd.DataFrame((df.median(), df.std()), index='med err'.split())
            pv = lpf.posterior_samples(derived_parameters=False).median().values
            phase = fold(time, pv[1], pv[0], 0.5) * pv[1] - 0.5 * pv[1]
            model = lpf.flux_model(pv)
            transit = lpf.transit_model(pv)

            # Calculate the per-orbit log likelihood differences
            # --------------------------------------------------
            ues = unique(epochs)
            lnl = zeros(ues.size)
            err = 10 ** pv[7]

            def lnlike_normal(o, m, e):
                npt = o.size
                return -npt * log(e) - 0.5 * npt * log(2. * pi) - 0.5 * sum((o - m) ** 2 / e ** 2)

            for i, e in enumerate(ues):
                m = epochs == e
                lnl[i] = lnlike_normal(flux[m], model[m], err) - lnlike_normal(flux[m], 1.0, err)

            self.transit_fit_results[mode] = res = TransitFitResult(df, time, epochs, phase, flux, model, transit,
                                                                    (ues, lnl))
            if mode == 'all':
                self.t0 = res.parameters.tc[0]
                self.period = res.parameters.p[0]
                self.duration = res.parameters.t14[0]
            return res

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
    def plot_flux_vs_time(self, ax=None):
        tref = 2457000
        transits = self.t0 + unique(epoch(self.time, self.t0, self.period)) * self.period
        [ax.axvline(t - tref, ls='--', alpha=0.5, lw=1) for t in transits]

        fsap = self._flux_sap
        fpdc = self._flux_pdcsap
        fpdc = fpdc + 1 - fpdc.min() + 3 * fsap.std()

        ax.plot(self.time - tref, fpdc, label='PDC')
        tb, fb, eb = downsample_time(self.time - tref, fpdc, 1 / 24)
        ax.plot(tb, fb, 'k', lw=1)

        ax.plot(self.time - tref, fsap, label='SAP')
        tb, fb, eb = downsample_time(self.time - tref, fsap, 1 / 24)
        ax.plot(tb, fb, 'k', lw=1)

        ax.legend(loc='best')
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim((ax.get_ylim()[0], median(fpdc) + 5 * sigma_clipped_stats(fpdc)[2]))
        setp(ax, xlabel=f'Time - {tref} [BJD]', ylabel='Normalized flux')

    @bplot
    def plot_bls_snr(self, ax=None):
        r = self.bls_result
        ax.plot(r.periods, r.depth_snr, drawstyle='steps-mid')
        ax.axvline(r.period, alpha=0.5, ls='--', lw=2)
        setp(ax, xlabel='Period [d]', ylabel='Depth SNR')
        ax.autoscale(axis='x', tight=True)
        ax.text(0.98, 0.95, f"Period {self.period:5.2f} d\nSNR {self.bls_result.snr:5.2f}", ha='right', va='top',
         transform=ax.transAxes, bbox=dict(facecolor='w'))

    @bplot
    def plot_ls_power(self, ax=None):
        r = self.ls_result
        ax.semilogx(r.periods, r.powers, drawstyle='steps-mid')
        ax.autoscale(axis='x', tight=True)

        faps = [0.1, 0.01, 0.001]
        fals = self.ls.false_alarm_level(faps)

        axis_to_data = ax.transAxes + ax.transData.inverted()
        fap_label_x = axis_to_data.transform((0.98, 0.99))[0]

        [ax.axhline(l) for l in fals]
        [ax.text(fap_label_x, fal, f"FAP = {100 * fap:>4.1f}%", va='center', ha='right', bbox=dict(facecolor='w')) for
         fal, fap in zip(fals, faps)]

        axis_to_data = ax.transAxes + ax.transData.inverted()
        period_label_y = axis_to_data.transform((0.98, 0.97))[1]
        ax.axvline(r.period, c='k', ls='--', alpha=0.25)
        ax.text(r.period, period_label_y, f"{r.period: .2f} d", va='top', ha='center', bbox=dict(facecolor='w'))

        setp(ax, xlabel='Period [d]', ylabel='LS Power')

    @bplot
    def plot_transit_fit(self, ax=None, full_phase: bool = False, mode='all', nbins: int = 20, alpha=0.2):
        model = self.transit_fit_results[mode]
        p = model.parameters
        zero_epoch, period, duration = p[['tc', 'p', 't14']].iloc[0].copy()
        hdur = duration * array([-0.5, 0.5])

        flux_m = model.model
        phase = model.phase
        sids = argsort(phase)
        phase = phase[sids]

        if full_phase:
            pmask = ones(phase.size, bool)
        else:
            pmask = abs(phase) < 1.5 * duration

        flux_m = flux_m[sids]
        flux_o = model.obs[sids]
        ax.plot(phase[pmask], flux_o[pmask], '.', alpha=alpha)
        ax.plot(phase[pmask], flux_m[pmask], 'k')

        pb, fb, eb = downsample_time(phase[pmask], flux_o[pmask], phase[pmask].ptp() / nbins)
        ax.errorbar(pb, fb, eb, fmt='ok')

        ax.text(2.5 * hdur[0], flux_m.min(), '{:6.4f}'.format(flux_m.min()), size=7, va='center',
                bbox=dict(color='white'))
        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
        [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur]

        fr = percentile(flux_o, [1, 99])
        ax.autoscale(axis='x', tight='true')
        setp(ax, ylim=fr, xlabel='Phase [h]', ylabel='Normalised flux')
        setp(ax.get_yticklabels(), visible=False)

    @bplot
    def plot_folded_and_binned_lc(self, ax=None, nbins: int = 100):
        p = self.tf_both.parameters
        zero_epoch, period, duration = p[['tc', 'p', 't14']].iloc[0].copy()
        phase = self.tf_both.phase
        sids = argsort(phase)
        phase = phase[sids]
        flux_o = self.tf_both.obs[sids]
        flux_m = self.tf_both.model[sids]

        pb, fb, eb = downsample_time(phase, flux_o, period / nbins)
        _, fob, _ = downsample_time(phase, flux_m, period / nbins)

        ax.errorbar(pb, fb, eb)
        ax.plot(pb, fob, 'k')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Phase [d]')

    @bplot
    def plot_even_odd(self, axs=None, nbins: int = 20, alpha=0.2):
        hdur = self.duration * array([-0.5, 0.5])

        for i, ms in enumerate(('even', 'odd')):
            m = self.transit_fit_results[ms]
            phase = m.phase
            sids = argsort(phase)
            phase = phase[sids]
            pmask = abs(phase) < 1.5 * self.duration

            flux_m = m.transit[sids]
            flux_o = m.obs[sids]

            pb, fb, eb = downsample_time(phase[pmask], flux_o[pmask], phase[pmask].ptp() / nbins)
            axs[0].errorbar(24 * pb, fb, eb, fmt='o-', label=ms)
            axs[1].plot(phase[pmask], flux_m[pmask], label=ms)

        for ax in axs:
            ax.legend(loc='best')
            ax.autoscale(axis='x', tight='true')

        setp(axs[0], ylabel='Normalized flux')
        setp(axs, xlabel='Phase [h]')

    @bplot
    def plot_per_orbit_delta_lnlike(self, ax=None):
        dlnl = self.transit_fit_results['all'].lnldiff
        ax.plot(dlnl[0], dlnl[1])
        for marker, model in zip('ox', ('even', 'odd')):
            dlnl = self.transit_fit_results[model].lnldiff
            ax.plot(dlnl[0], dlnl[1], ls='', marker=marker, label=model)
        ax.axhline(0, c='k', ls='--', alpha=0.25, lw=1)
        ax.legend(loc='best')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Epoch', ylabel="$\Delta$ log likelihood")

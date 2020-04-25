from collections import namedtuple
from functools import wraps
from pathlib import Path
from typing import Iterable, Callable
from logging import info, warning, error

import astropy.units as u
import pandas as pd
from astropy.io.fits import HDUList, PrimaryHDU, Card
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.table import Table
from astropy.timeseries import BoxLeastSquares, LombScargle
from celerite import GP
from celerite.terms import SHOTerm
from matplotlib.axis import Axis
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplots, setp, errorbar, figure, subplot
from numpy import linspace, array, argmax, ones, floor, log, pi, sin, argsort, unique, median, zeros, atleast_2d, \
    ndarray, squeeze, percentile, inf, asarray, sqrt, arcsin, exp, isfinite
from pytransit import BaseLPF, QuadraticModelCL, QuadraticModel
from pytransit.lpf.loglikelihood import CeleriteLogLikelihood
from pytransit.orbits import epoch
from pytransit.orbits.orbits_py import impact_parameter_ec
from pytransit.param import LParameter, UniformPrior as UP
from pytransit.utils.misc import fold
from pytransit.lpf.tesslpf import downsample_time
from scipy.interpolate import interp1d

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
        self.planet: int = 1
        self.pmin: float = pmin
        self.pmax: float = pmax
        self.nper: int  = nper
        self.nsamples: int = nsamples
        self.exptime: float = exptime
        self.use_tqdm: bool = use_tqdm
        self.use_opencl: bool = use_opencl
        self.snr_limit: float = snr_limit

        self.bls = None
        self.bls_result = None
        self.ls = None
        self.ls_result = None
        self.transit_fits = {}
        self.transit_fit_results = {}
        self.gp_result = None
        self.gp_periodicity = None

        self.time: Optional[ndarray] = None
        self.flux: Optional[ndarray] = None
        self.ferr: Optional[ndarray] = None
        self.phase: Optional[ndarray] = None

        self.teff: Optional[float] = None
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
        info("Running Celerite")
        self.fit_gp_periodicity()
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
        name, time, flux, ferr = self._reader(filename)
        self.name = name
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
        tm = QuadraticModelCL(klims=(0.01, 0.60)) if self.use_opencl else QuadraticModel(interpolate=False)
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
            lpf.set_prior('k2', 'UP', max(0.01**2, 0.5 * priorp.k2.med), min(0.6**2, 2 * priorp.k2.med))
            lpf.set_prior('q1_Kepler', 'NP', priorp.q1_Kepler.med, priorp.q1_Kepler.err)
            lpf.set_prior('q2_Kepler', 'NP', priorp.q2_Kepler.med, priorp.q2_Kepler.err)

        if self.teff is not None:
            ldcs = Table.read(Path(__file__).parent / "data/ldc_table.fits").to_pandas()
            ip = interp1d(ldcs.teff, ldcs[['q1', 'q2']].T)
            q1, q2 = ip(self.teff)
            lpf.set_prior('q1_Kepler', 'NP', q1, 1e-5)
            lpf.set_prior('q2_Kepler', 'NP', q2, 1e-5)

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
        period = period if period is not None else self.period

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

    # FITS output
    # ===========
    def _create_fits(self):
        hdul = HDUList(PrimaryHDU())
        h = hdul[0].header
        h.append(Card('name', self.name))
        self._cf_pre_hook(hdul)
        self._cf_add_setup_info(hdul)
        self._cf_add_stellar_info(hdul)
        self._cf_add_summary_statistics(hdul)
        self._cf_add_bls_results(hdul)
        self._cf_add_ls_results(hdul)
        self._cf_add_celerite_results(hdul)
        self._cf_add_transit_fit_results(hdul, 'all', ' Transit fit results  ')
        self._cf_add_transit_fit_results(hdul, 'even', ' Even tr. fit results  ')
        self._cf_add_transit_fit_results(hdul, 'odd', ' Odd tr. fit results  ')
        self._cf_post_hook(hdul)
        return hdul

    def _cf_pre_hook(self, hdul: HDUList):
        pass

    def _cf_add_setup_info(self, hdul: HDUList):
        h = hdul[0].header
        h.append(Card('COMMENT', '======================'))
        h.append(Card('COMMENT', ' Transit search setup '))
        h.append(Card('COMMENT', '======================'))
        h.append(Card('pmin', self.pmin, 'Minimum search period [d]'), bottom=True)
        h.append(Card('pmax', self.pmax, 'Maximum search period [d]'), bottom=True)
        h.append(Card('dper', (self.pmax - self.pmin) / self.nper, 'Period grid step size [d]'), bottom=True)
        h.append(Card('nper', self.nper, 'Period grid size'), bottom=True)
        h.append(Card('dmin', 0, 'Minimum search duration [d]'), bottom=True)
        h.append(Card('dmax', 0, 'Maximum search duration [d]'), bottom=True)
        h.append(Card('ddur', 0, 'Duration grid step size [d]'), bottom=True)
        h.append(Card('ndur', 0, 'Duration grid size'), bottom=True)

    def _cf_add_stellar_info(self, hdul: HDUList):
        pass

    def _cf_add_summary_statistics(self, hdul: HDUList):
        h = hdul[0].header
        h.append(Card('COMMENT', '======================'))
        h.append(Card('COMMENT', '  Summary statistics  '))
        h.append(Card('COMMENT', '======================'))
        h.append(Card('fstd', self.flux.std(), "Flux standard deviation"), bottom=True)
        h.append(Card('fmadstd', mad_std(self.flux), "Flux MAD standard deviation"), bottom=True)
        ps = [0.1, 1, 5, 95, 99, 99.9]
        pvs = percentile(self.flux, ps)
        pks = [f"fpc{int(10 * p):03d}" for p in ps]
        for p, pk, pv in zip(ps, pks, pvs):
            h.append(Card(pk, pv, f"{p:4.1f} normalized flux percentile"), bottom=True)

    def _cf_add_bls_results(self, hdul: HDUList):
        if self.bls_result is not None:
            h = hdul[0].header
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', '     BLS results      '))
            h.append(Card('COMMENT', '======================'))
            h.append(Card('bls_snr', self.snr, 'BLS depth signal to noise ratio'), bottom=True)
            h.append(Card('period', self.period, 'Orbital period [d]'), bottom=True)
            h.append(Card('epoch', self.zero_epoch, 'Zero epoch [BJD]'), bottom=True)
            h.append(Card('duration', self.duration, 'Transit duration [d]'), bottom=True)
            h.append(Card('depth', self.bls_result.depth, 'Transit depth'), bottom=True)

    def _cf_add_ls_results(self, hdul: HDUList):
        if self.ls_result is not None:
            h = hdul[0].header
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', ' Lomb-Scargle results '))
            h.append(Card('COMMENT', '======================'))
            h.append(Card('lsper', self.ls_result.period, 'Lomb-Scargle period [d]'), bottom=True)
            h.append(Card('lspow', self.ls_result.power, 'Lomb-Scargle power'), bottom=True)
            h.append(Card('lsfap', self.ls.false_alarm_probability(self.ls_result.power),
                          'Lomb-Scargle false alarm probability'), bottom=True)

    def _cf_add_celerite_results(self, hdul: HDUList):
        if self.gp_periodicity is not None:
            h = hdul[0].header
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', ' Celerite periodicity '))
            h.append(Card('COMMENT', '======================'))
            h.append(Card('GPPLS', self.gp_periodicity[0], 'GP SHOT term log S0'), bottom=True)
            h.append(Card('GPPLQ', self.gp_periodicity[1], 'GP SHOT term log Q'), bottom=True)
            h.append(Card('GPPLO', log(2 * pi / self.period), 'GP SHOT term log omega'), bottom=True)

    def _cf_add_transit_fit_results(self, hdul: HDUList, run: str, title: str):
        if run in self.transit_fit_results:
            p = self.transit_fit_results[run].parameters
            c = run[0]
            h = hdul[0].header
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', title))
            h.append(Card('COMMENT', '======================'))
            h.append(Card(f'TF{c}_T0', p.tc.med, 'Transit centre [BJD]'), bottom=True)
            h.append(Card(f'TF{c}_T0E', p.tc.err, 'Transit centre uncertainty [d]'), bottom=True)
            h.append(Card(f'TF{c}_PR', p.p.med, 'Orbital period [d]'), bottom=True)
            h.append(Card(f'TF{c}_PRE', p.p.err, 'Orbital period uncertainty [d]'), bottom=True)
            h.append(Card(f'TF{c}_RHO', p.rho.med, 'Stellar density [g/cm^3]'), bottom=True)
            h.append(Card(f'TF{c}_RHOE', p.rho.err, 'Stellar density uncertainty [g/cm^3]'), bottom=True)
            h.append(Card(f'TF{c}_B', p.b.med, 'Impact parameter'), bottom=True)
            h.append(Card(f'TF{c}_BE', p.b.err, 'Impact parameter uncertainty'), bottom=True)
            h.append(Card(f'TF{c}_AR', p.k2.med, 'Area ratio'), bottom=True)
            h.append(Card(f'TF{c}_ARE', p.k2.err, 'Area ratio uncertainty'), bottom=True)
            h.append(Card(f'TF{c}_SC', p.c_sin.med, 'Sine phase'), bottom=True)
            h.append(Card(f'TF{c}_SCE', p.c_sin.err, 'Sine phase uncertainty'), bottom=True)
            h.append(Card(f'TF{c}_SA', p.a_sin_0.med, 'Sine amplitude'), bottom=True)
            h.append(Card(f'TF{c}_SAE', p.a_sin_0.err, 'Sine amplitude uncertainty'), bottom=True)
            h.append(Card(f'TF{c}_RR', p.k.med, 'Radius ratio'), bottom=True)
            h.append(Card(f'TF{c}_RRE', p.k.err, 'Radius ratio uncertainty'), bottom=True)
            h.append(Card(f'TF{c}_A', p.a.med, 'Semi-major axis'), bottom=True)
            h.append(Card(f'TF{c}_AE', p.a.err, 'Semi-major axis uncertainty'), bottom=True)
            h.append(Card(f'TF{c}_T14', p.t14.med, 'Transit duration T14 [d]'), bottom=True)
            h.append(Card(f'TF{c}_T14E', p.t14.err, 'Transit duration T14 uncertainty [d]'), bottom=True)
            if isfinite(p.t23.med) and isfinite(p.t23.err):
                h.append(Card(f'TF{c}_T23', p.t23.med, 'Transit duration T23 [d]'), bottom=True)
                h.append(Card(f'TF{c}_T23E', p.t23.err, 'Transit duration T23 uncertainty [d]'), bottom=True)
                h.append(Card(f'TF{c}_TDR', p.t23.med / p.t14.med, 'T23 to T14 ratio'), bottom=True)
            else:
                h.append(Card(f'TF{c}_T23', 0, 'Transit duration T23 [d]'), bottom=True)
                h.append(Card(f'TF{c}_T23E', 0, 'Transit duration T23 uncertainty [d]'), bottom=True)
                h.append(Card(f'TF{c}_TDR', 0, 'T23 to T14 ratio'), bottom=True)
            h.append(Card(f'TF{c}_WN', 10 ** p.wn_loge_0.med, 'White noise std'), bottom=True)
            h.append(Card(f'TF{c}_GRAZ', p.b.med + p.k.med > 1., 'Is the transit grazing'), bottom=True)

            ep, ll = self.transit_fit_results[run].lnldiff
            lm = ll.max()
            h.append(Card(f'TF{c}_DLLA', log(exp(ll - lm).mean()) + lm, 'Mean per-orbit delta log likelihood'), bottom=True)
            if run == 'all':
                m = ep % 2 == 0
                lm = ll[m].max()
                h.append(Card(f'TFA_DLLO', log(exp(ll[m] - lm).mean()) + lm, 'Mean per-orbit delta log likelihood (odd)'),
                         bottom=True)
                m = ep % 2 != 0
                lm = ll[m].max()
                h.append(Card(f'TFA_DLLE', log(exp(ll[m] - lm).mean()) + lm, 'Mean per-orbit delta log likelihood (even)'),
                         bottom=True)

    def _cf_post_hook(self, hdul: HDUList):
        pass

    # Plotting
    # ========
    def bplot(plotf: Callable):
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

    def plot_report(self):
        fig = figure(figsize=(16, 16))
        gs = GridSpec(7, 4, figure=fig, height_ratios=(0.7, 1, 1, 1, 1, 1, 0.1))
        ax_header = subplot(gs[0, :])
        ax_flux = subplot(gs[1:3, :])
        ax_snr = subplot(gs[3, :2])
        ax_ls = subplot(gs[4, :2])
        ax_transit = subplot(gs[4, 2:])
        ax_folded = subplot(gs[3, 2:])
        ax_even_odd = subplot(gs[5, 2]), subplot(gs[5, 3])
        ax_per_orbit_lnlike = subplot(gs[5, :2])
        ax_footer = subplot(gs[-1, :])

        self.plot_header(ax_header)
        self.plot_flux_vs_time(ax_flux)
        self.plot_bls_snr(ax_snr)
        self.plot_ls_power(ax_ls)

        self.plot_transit_fit(ax_transit, nbins=40)
        self.plot_folded_and_binned_lc(ax_folded)
        self.plot_even_odd(ax_even_odd)
        self.plot_per_orbit_delta_lnlike(ax_per_orbit_lnlike)
        ax_footer.axhline(lw=20)

        ax_snr.set_title('BLS periodogram')
        ax_ls.set_title('Lomb-Scargle periodigram')
        ax_transit.set_title('Phase-folded transit')
        ax_folded.set_title('Phase-folded orbit')
        ax_per_orbit_lnlike.set_title('Per-orbit $\Delta$ log likelihood')
        ax_even_odd[0].set_title('Observed even and odd transits')
        ax_even_odd[1].set_title('Modelled even and odd transits')
        setp(ax_footer, frame_on=False, xticks=[], yticks=[])
        setp(ax_even_odd[1], yticks=[], ylim=ax_even_odd[0].get_ylim())
        fig.tight_layout()
        return fig

    def plot_header(self, ax):
        ax.axhline(1.0, lw=20)
        ax.text(0.01, 0.77, self.name.replace('_', ' '), size=33, va='top', weight='bold')
        ax.text(0.01, 0.25,
                       f"SNR {self.snr:5.2f} | Period  {self.period:5.2f} d | Zero epoch {self.t0:10.2f} | Depth {self.bls_result.depth:5.4f} | Duration {24 * self.duration:>4.2f} h",
                       va='center', size=18, linespacing=1.75, family='monospace')
        ax.axhline(0.0, lw=8)
        setp(ax, frame_on=False, xticks=[], yticks=[])

    @bplot
    def plot_flux_vs_time(self, ax = None):
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
    def plot_bls_snr(self, ax = None):
        r = self.bls_result
        ax.semilogx(r.periods, r.depth_snr, drawstyle='steps-mid')
        ax.axvline(r.period, alpha=0.5, ls='--', lw=2)
        setp(ax, xlabel='Period [d]', ylabel='Depth SNR')
        ax.autoscale(axis='x', tight=True)
        ax.text(0.81, 0.93, f"Period  {self.period:5.2f} d\nSNR     {self.bls_result.snr:5.2f}", va='top',
                transform=ax.transAxes, bbox=dict(facecolor='w'), family='monospace', )

    @bplot
    def plot_ls_power(self, ax=None):
        r = self.ls_result
        ax.semilogx(r.periods, r.powers, drawstyle='steps-mid')
        ax.autoscale(axis='x', tight=True)
        fap = self.ls.false_alarm_probability(r.power)
        ax.text(0.81, 0.93, f"Period {r.period:5.2f} d\nFAP    {fap:5.2f}", va='top', family='monospace',
                transform=ax.transAxes, bbox=dict(facecolor='w'))
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

        ax.text(2.5 * hdur[0], flux_m.min(), f'$\Delta$F {1 - flux_m.min():6.4f}', size=10, va='center',
                bbox=dict(color='white'))
        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
        [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur]

        ylim = fb.min() - 2 * eb.max(), fb.max() + 2 * eb.max()
        ax.autoscale(axis='x', tight='true')
        setp(ax, ylim=ylim, xlabel='Phase [h]', ylabel='Normalised flux')

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
        setp(ax, xlabel='Phase [d]', ylabel='Normalized flux')

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
            ax.legend(loc='upper right')
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
        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Epoch', ylabel="$\Delta$ log likelihood")

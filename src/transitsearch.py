#  OpenTS: Open exoplanet transit search pipeline.
#  Copyright (C) 2015-2020  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging

from pathlib import Path

from astropy.io.fits import HDUList, PrimaryHDU, Card
from astropy.stats import mad_std
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.pyplot import setp, figure, subplot
from matplotlib.transforms import offset_copy
from numpy import log, pi, argsort, unique, ndarray, percentile, array, concatenate, isfinite
from pytransit.orbits import epoch
from pytransit.utils.misc import fold
from pytransit.lpf.tesslpf import downsample_time

from typing import Optional, List, Union, Dict

from tqdm.auto import tqdm

from .celeritestep import CeleriteM32Step, CeleriteSHOTStep
from .plots import bplot
from .blsstep import BLSStep
from .lsstep import LombScargleStep
from .pvarstep import PVarStep
from .tfstep import TransitFitStep


class TSData:
    def __init__(self, time: ndarray, flux: ndarray, ferr: ndarray):
        self._steps: List = ['init']
        self._time: List = [time]
        self._flux: List = [flux]
        self._ferr: List = [ferr]

    def update(self, step: str, time: ndarray, flux: ndarray, ferr: ndarray):
        self._steps.append(step)
        self._time.append(time)
        self._flux.append(flux)
        self._ferr.append(ferr)

    @property
    def step(self) -> ndarray:
        return self._steps[-1]

    @property
    def time(self) -> ndarray:
        return self._time[-1]

    @property
    def flux(self) -> ndarray:
        return self._flux[-1]

    @property
    def ferr(self) -> ndarray:
        return self._ferr[-1]


class TransitSearch:
    def __init__(self, pmin: float = 0.25, pmax: Optional[float] = None, nper: int = 10000, bic_limit: float = 5, min_transits: int = 3,
                 nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):

        self.name: Optional[str] = ''
        self.planet: int = 1
        self.min_transits: int = min_transits
        self.pmin: float = pmin
        self.pmax: Optional[float] = pmax
        self.nper: int  = nper
        self.nsamples: int = nsamples
        self.exptime: float = exptime
        self.use_tqdm: bool = use_tqdm
        self.use_opencl: bool = use_opencl
        self.bic_limit: float = bic_limit

        self.bls = None
        self.bls_result = None
        self.ls = None
        self.ls_result = None
        self.transit_fits = {}
        self.transit_fit_results = {}
        self.transit_fit_masks = {}
        self.gp_result = None
        self.gp_periodicity = None

        self.masked_periods = []
        self.logger = None
        self._data = None
        self._steps = []

        self.mag: Optional[float] = None           # Host star magnitude (TESS magnitude, Kepler Magnitude, etc.)
        self.teff: Optional[float] = None          # Host star effective temperature
        self.period: Optional[float] = None        # Orbital period
        self.zero_epoch: Optional[float] = None    # Zero epoch
        self.duration: Optional[float] = None      # Transit duration in days
        self.depth: Optional[float] = None         # Transit depth
        self.snr: Optional[float] = None           # Transit signal-to-noise ratio
        self.dbic: Optional[float] = None          # Delta BIC

        self.initialize_steps()
        self.set_plot_parameters()

    @staticmethod
    def can_read_input(source: Path) -> bool:
        raise NotImplementedError

    @staticmethod
    def gather_data(source: Path, target: Optional[int] = None) -> Dict:
        raise NotImplementedError

    def register_step(self, step):
        self._steps.append(step)
        return step

    def set_plot_parameters(self):
        self._tf_plot_nbins: int = 40

    def initialize_steps(self):
        """Initialize the pipeline steps.

        Returns
        -------
        None
        """
        self.ls      = self.register_step(LombScargleStep(self))
        self.pvar    = self.register_step(PVarStep(self))
        self.cvar    = self.register_step(CeleriteM32Step(self))
        self.bls     = self.register_step(BLSStep(self))
        self.tf_all  = self.register_step(TransitFitStep(self, 'all', ' Transit fit results  ', use_tqdm=self.use_tqdm))
        self.tf_even = self.register_step(TransitFitStep(self, 'even', ' Even tr. fit results  ', use_tqdm=self.use_tqdm))
        self.tf_odd  = self.register_step(TransitFitStep(self, 'odd', ' Odd tr. fit results  ', use_tqdm=self.use_tqdm))

    def update_data(self, step: str, time: ndarray, flux: ndarray, ferr: ndarray):
        self._data.update(step, time, flux, ferr)

    def read_data(self, filename: Path):
        name, time, flux, ferr = self._reader(filename)
        self._data = TSData(time, flux, ferr)
        self.name = name
        self.pmax = self.pmax or 0.98 * (time.ptp() / self.min_transits)

    def _reader(self, filename: Path):
        raise NotImplementedError

    @property
    def time(self) -> ndarray:
        return self._data.time

    @property
    def flux(self) -> ndarray:
        return self._data.flux

    @property
    def ferr(self) -> ndarray:
        return self._data.ferr

    @property
    def phase(self) -> Optional[ndarray]:
        if self.zero_epoch is not None:
            return fold(self.time, self.period, self.zero_epoch, 0.5) * self.period
        else:
            return None

    def update_ephemeris(self, zero_epoch, period, duration, depth: Optional[float] = None):
        self.zero_epoch = zero_epoch
        self.period = period
        self.duration = duration
        self.depth = depth if depth is not None else self.depth

    @property
    def basename(self):
        raise NotImplementedError

    def run(self):
        """Run all the steps in the pipeline.

        Returns
        -------
        None
        """
        for step in tqdm(self._steps, desc=f"{self.name} {self.planet:>2d}", leave=False, disable=not self.use_tqdm, position=1):
            step()

    def next_planet(self):
        """Remove the current best-fit planet signal and prepare to search for another.

        Returns
        -------
        None
        """
        self.planet += 1
        mask = self.transit_fits['all'].mask
        flux = self.flux.copy()
        flux[mask] /= self.transit_fits['all'].ftra
        self.update_data(f'Removed planet {self.planet - 1}', self.time, flux, self.ferr)

        # Period masking
        # --------------
        # We mask the current period +- 0.5 days from the next search. This is necessary in the cases
        # where we have a poorly-fit EB. Removing the model can still leave a signal that is strong enough
        # for the pipeline to find it again and again and again...
        self.masked_periods.append(self.period)

    # FITS output
    # ===========
    def save_fits(self, savedir: Union[Path, str]):
        """Save the search results into a FITS file.

        Parameters
        ----------
        savedir: Path or str
            Directory to save the results in.

        Returns
        -------
        None
        """
        hdul = self._create_fits()
        hdul.writeto(Path(savedir) / f"{self.name}_{self.planet}.fits", overwrite=True)

    def _create_fits(self):
        hdul = HDUList(PrimaryHDU())
        h = hdul[0].header
        h.append(Card('name', self.name))
        self._cf_pre_hook(hdul)
        self._cf_add_setup_info(hdul)
        self._cf_post_setup_hook(hdul)
        self._cf_add_summary_statistics(hdul)
        self._cf_add_pipeline_steps(hdul)
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

    def _cf_post_setup_hook(self, hdul: HDUList):
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

    def _cf_add_pipeline_steps(self, hdul: HDUList):
        for step in self._steps:
            step.add_to_fits(hdul)

    def _cf_post_hook(self, hdul: HDUList):
        pass

    # Plotting
    # ========
    def plot_report(self):
        def sb(*nargs, **kwargs) -> Axes:
            return subplot(*nargs, **kwargs)

        lmargin = 0.06
        rmargin = 0.01

        headheight = 0.11
        headpad = 0.04
        footheight = 0.03
        footpad = 0.04

        fig = figure(figsize=(16, 19))
        gs = GridSpec(8, 4, figure=fig, wspace=0.4, hspace=0.5,
                      left=lmargin, right=1 - rmargin, bottom=footheight + footpad, top=1 - headheight - headpad)

        header = fig.add_axes((0, 1 - headheight, 1, headheight), frame_on=False, xticks=[], yticks=[])
        footer = fig.add_axes((0, 0, 1, footheight), frame_on=False, xticks=[], yticks=[])

        # Flux over time
        # --------------
        aflux = sb(gs[0:2, :])
        self.plot_flux_vs_time(aflux)

        # Periodograms
        # ------------
        gs_periodograms = gs[2:4, :2].subgridspec(2, 1, hspace=0.05)
        abls = sb(gs_periodograms[0])
        als = sb(gs_periodograms[1], sharex=abls)

        self.bls.plot_snr(abls)
        abls.text(0.02, 1, 'Periodograms', va='center', ha='left', transform=abls.transAxes, size=11,
                  bbox=dict(facecolor='w'))
        tra = offset_copy(abls.transAxes, fig=fig, x=-10, y=10, units='points')
        abls.text(1, 0, 'BLS', va='bottom', ha='right', transform=tra, size=11, bbox=dict(facecolor='w'))

        self.ls.plot_power(als)
        tra = offset_copy(als.transAxes, fig=fig, x=-10, y=10, units='points')
        als.text(1, 0, 'Lomb-Scargle', va='bottom', ha='right', transform=tra, size=11, bbox=dict(facecolor='w'))

        # Per-orbit log likelihood difference
        # -----------------------------------
        gs_dll = gs[4:6, :2].subgridspec(2, 1, hspace=0.05)
        adll = sb(gs_dll[0])
        adllc = sb(gs_dll[1], sharex=adll)

        self.plot_dll(adll)
        self.plot_cumulative_dll(adllc)

        adll.text(0.02, 1, 'Per-orbit $\Delta$ log likelihood', va='center', transform=adll.transAxes, size=11,
                  bbox=dict(facecolor='w'))
        setp(adll.get_xticklabels(), visible=False)
        setp(adll, xlabel='')

        # Periodic variability
        # --------------------
        apvp = sb(gs[6, :2])
        apvt = sb(gs[6, 2:])

        self.pvar.plot_over_phase(apvp)
        apvp.text(0.02, 1, 'Periodic variability', va='center', transform=apvp.transAxes, size=11,
                  bbox=dict(facecolor='w'))

        self.pvar.plot_over_time(apvt)
        apvt.text(0.02, 1, 'Periodic variability', va='center', transform=apvt.transAxes, size=11,
                  bbox=dict(facecolor='w'))

        # Transit and orbit
        # -----------------
        gs_to = gs[2:5, 2:].subgridspec(2, 1)
        atr = sb(gs_to[0])
        aor = sb(gs_to[1], sharey=atr)

        self.tf_all.plot_transit_fit(atr, nbins=self._tf_plot_nbins)
        self.plot_folded_orbit(aor)

        atr.text(0.02, 1, 'Phase-folded transit', va='center', transform=atr.transAxes, size=11,
                 bbox=dict(facecolor='w'))
        aor.text(0.02, 1, 'Phase-folded orbit', va='center', transform=aor.transAxes, size=11, bbox=dict(facecolor='w'))

        # Even ad odd transit fits
        # ------------------------
        gs3 = gs[5, 2:].subgridspec(1, 2, wspace=0.05)
        ato = sb(gs3[0])
        atm = sb(gs3[1], sharey=ato)

        self.plot_even_odd((ato, atm))
        ato.text(0.02, 1, 'Even and odd transits', va='center', transform=ato.transAxes, size=11,
                 bbox=dict(facecolor='w'))
        setp(atm.get_yticklabels(), visible=False)

        # Parameter posteriors
        # --------------------
        gs_posteriors = gs[7,:].subgridspec(3, 5, hspace=0, wspace=0.01)
        aposteriors = array([[subplot(gs_posteriors[j, i]) for i in range(5)] for j in range(3)])
        self.plot_posteriors(aposteriors)
        aposteriors[0,0].text(0.02, 1, 'Parameter posteriors', va='center', transform=aposteriors[0,0].transAxes, size=11,
                  bbox=dict(facecolor='w'))

        fig.lines.extend([
            Line2D([0, 1], [1, 1], lw=10, transform=fig.transFigure, figure=fig),
            Line2D([0, 1], [1 - headheight, 1 - headheight], lw=10, transform=fig.transFigure, figure=fig),
            Line2D([0, 1], [footheight, footheight], lw=5, transform=fig.transFigure, figure=fig),
            Line2D([0, 1], [0, 0], lw=10, transform=fig.transFigure, figure=fig)
        ])

        footer.text(0.5, 0.5, 'Open Exoplanet Transit Search Pipeline', ha='center', va='center', size='large')
        self.plot_header(header)

        fig.align_ylabels([aflux, abls, als, adll, adllc, apvp])
        fig.align_ylabels([atr, aor, ato, apvt])
        return fig

    def plot_header(self, ax):
        ax.text(0.01, 0.85, f"{self.name.replace('_', ' ')} - Candidate {self.planet}", size=33, va='top', weight='bold')
        mags = f"{self.mag:3.1f}" if self.mag else '   '
        teffs = f"{self.teff:5.0f}" if self.teff else '     '
        s = (f"$\Delta$BIC {self.dbic:6.1f} | SNR {self.bls.snr:6.1f}   | Period    {self.period:5.2f} d | T$_0$ {self.zero_epoch:10.2f} | Depth {self.depth:5.4f} | Duration {24 * self.duration:>4.2f} h\n"
             f"Mag     {mags} | TEff {teffs} K | LS Period {self.ls.period:5.2f} d | LS FAP {self.ls.fap:5.4f} | Period limits  {self.pmin:5.2f} – {self.pmax:5.2f} d")
        ax.text(0.01, 0.55, s, va='top', size=19.2, linespacing=1.5, family='monospace')

    @bplot
    def plot_flux_vs_time(self, ax = None):
        transits = self.zero_epoch + unique(epoch(self.time, self.zero_epoch, self.period)) * self.period
        [ax.axvline(t, ls='--', alpha=0.5, lw=1) for t in transits]
        ax.plot(self.time, self.flux)
        tb, fb, eb = downsample_time(self.time, self.flux, 1 / 24)
        ax.plot(tb, fb, 'k', lw=1)
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel=f'Time [BJD]', ylabel='Normalized flux')

    @bplot
    def plot_dll(self, ax=None):
        ax.plot(self.transit_fits['all'].dll_epochs, self.transit_fits['all'].dll_values)
        for marker, model in zip('ox', ('even', 'odd')):
            tf = self.transit_fits[model]
            ax.plot(tf.dll_epochs, tf.dll_values, ls='', marker=marker, label=model)
        #ax.axhline(0, c='k', ls='--', alpha=0.25, lw=1)
        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Epoch', ylabel="$\Delta$ log likelihood")

    @bplot
    def plot_cumulative_dll(self, ax=None):
        ax.plot(self.tf_all.dll_epochs, self.tf_all.dll_values.cumsum())
        ax.plot(self.tf_even.dll_epochs, self.tf_even.dll_values.cumsum(), ls='--', alpha=0.5)
        ax.plot(self.tf_odd.dll_epochs, self.tf_odd.dll_values.cumsum(), ls='--', alpha=0.5)
        setp(ax, xlabel='Epoch', ylabel='Cumulative $\Delta$ log L')
        ax.autoscale(axis='x', tight=True)

    @bplot
    def plot_folded_orbit(self, ax=None, nbins: int = 100):
        phase = self.phase - 0.5 * self.period
        sids = argsort(phase)
        pb, fb, eb = downsample_time(phase[sids], self.flux[sids], phase.ptp() / nbins)
        ax.errorbar(pb, fb, eb, fmt='k.')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Phase [d]', ylabel='Normalized flux')

    @bplot
    def plot_even_odd(self, axs=None, nbins: int = 20, alpha=0.2):
        for i, ms in enumerate(('even', 'odd')):
            m = self.transit_fits[ms]
            sids = argsort(m.phase)
            phase = m.phase[sids]
            pmask = abs(phase) < 1.5 * self.duration
            phase = phase[pmask]
            fmod = m.fmod[sids][pmask]
            fobs = m.fobs[sids][pmask]

            pb, fb, eb = downsample_time(phase, fobs, phase.ptp() / nbins)
            mask = isfinite(pb)
            pb, fb, eb = pb[mask], fb[mask], eb[mask]
            axs[0].errorbar(24 * pb, fb, eb, fmt='o-', label=ms)
            axs[1].plot(phase, fmod, label=ms)

        axs[1].legend(loc='upper right')
        for ax in axs:
            ax.autoscale(axis='x', tight='true')

        setp(axs[0], ylabel='Normalized flux')
        setp(axs, xlabel='Phase [h]')

    def plot_posteriors(self, axs):
        df = self.tf_all.lpf.posterior_samples()
        dfe = self.tf_even.lpf.posterior_samples()
        dfo = self.tf_odd.lpf.posterior_samples()

        for d in (df,dfe,dfo):
            d['t14'] *= 24

        def plotp(data, ax, c='C0'):
            p = percentile(data, [50, 16, 84, 2.5, 97.5])
            ax.axvspan(*p[3:5], alpha=0.5, fc=c)
            ax.axvspan(*p[1:3], alpha=0.5, fc=c)
            ax.axvline(p[0], c='k')

        for i, l in enumerate('b rho k t14'.split()):
            for j,d in enumerate((df, dfe, dfo)):
                plotp(d[l], axs[j,i])
            try:
                setp(axs[:,i], xlim=percentile(concatenate([df[l].values, dfe[l].values, dfo[l].values]), [1, 99]))
            except ValueError:
                pass

        plotp(df['ble'], axs[0, 4])
        plotp(df['blo'], axs[0, 4], c='C1')
        plotp(dfe['ble'], axs[1, 4])
        plotp(dfo['blo'], axs[2, 4], c='C1')
        av = concatenate([df['ble'].values, df['blo'].values, dfo['blo'].values, dfe['ble'].values])
        try:
            setp(axs[:,4], xlim=percentile(av, [1, 99]))
        except ValueError:
            pass

        setp(axs, yticks=[])
        setp(axs[:-1, :], xticks=[])
        for ax, label in zip(axs[-1], 'Impact parameter, Stellar density [g/cm$^3$], Radius ratio, Duration [h], Eve-Odd baseline'.split(', ')):
            ax.set_xlabel(label)
        for ax, label in zip(axs[:, 0], 'all even odd'.split()):
            ax.set_ylabel(label)
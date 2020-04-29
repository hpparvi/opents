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

from pathlib import Path
from logging import info

from astropy.io.fits import HDUList, PrimaryHDU, Card
from astropy.stats import mad_std
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import setp, figure, subplot
from numpy import log, pi, argsort, unique, ndarray, percentile, array
from pytransit.orbits import epoch
from pytransit.utils.misc import fold
from pytransit.lpf.tesslpf import downsample_time

from typing import Optional, List, Union

from .celeritestep import CeleriteStep
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
    def __init__(self, pmin: float = 0.25, pmax: float = 15., nper: int = 10000, bic_limit: float = 5,
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

        self._data = None
        self._steps = []

        self.teff: Optional[float] = None          # Host star effective temperature
        self.period: Optional[float] = None        # Orbital period
        self.zero_epoch: Optional[float] = None    # Zero epoch
        self.duration: Optional[float] = None      # Transit duration in days
        self.depth: Optional[float] = None         # Transit depth
        self.snr: Optional[float] = None           # Transit signal-to-noise ratio
        self.dbic: Optional[float] = None          # Delta BIC

        self.initialize_steps()

    def register_step(self, step):
        self._steps.append(step)
        return step

    def initialize_steps(self):
        """Initialize the pipeline steps.

        Returns
        -------
        None
        """
        self.ls      = self.register_step(LombScargleStep(self))
        self.cvar    = self.register_step(CeleriteStep(self))
        self.pvar    = self.register_step(PVarStep(self))
        self.bls     = self.register_step(BLSStep(self))
        self.tf_all  = self.register_step(TransitFitStep(self, 'all', ' Transit fit results  '))
        self.tf_even = self.register_step(TransitFitStep(self, 'even', ' Even tr. fit results  '))
        self.tf_odd  = self.register_step(TransitFitStep(self, 'odd', ' Odd tr. fit results  '))

    def update_data(self, time: ndarray, flux: ndarray, ferr: ndarray):
        self._data.update(time, flux, ferr)

    def read_data(self, filename: Path):
        name, time, flux, ferr = self._reader(filename)
        self._data = TSData(time, flux, ferr)
        self.name = name

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
        for step in self._steps:
            step()

    def next_planet(self):
        """Remove the current best-fit planet signal and prepare to search for another.

        Returns
        -------
        None
        """
        self.planet += 1
        flux = self.flux[self.transit_fit_masks['all']] / self.transit_fit_results['all'].transit
        self.update_data(f'Removed planet {self.planet - 1}', self.time, flux, self.ferr)

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
        fig = figure(figsize=(16, 17))
        gs = GridSpec(8, 4, figure=fig, height_ratios=(0.7, 1, 1, 1, 1, 1, 1, 0.1))
        ax_header = subplot(gs[0, :])
        ax_flux = subplot(gs[1:3, :])
        ax_snr = subplot(gs[3, :2])
        ax_ls = subplot(gs[4, :2])
        ax_transit = subplot(gs[4, 2:])
        ax_folded = subplot(gs[3, 2:])
        ax_even_odd = subplot(gs[5, 2]), subplot(gs[5, 3])
        ax_per_orbit_lnlike = subplot(gs[5, :2])
        ax_pvar = subplot(gs[6,:2])
        ax_footer = subplot(gs[-1, :])

        self.plot_header(ax_header)
        self.plot_flux_vs_time(ax_flux)
        self.bls.plot_snr(ax_snr)
        self.ls.plot_power(ax_ls)

        self.tf_all.plot_transit_fit(ax_transit, nbins=40)
        self.tf_all.plot_folded_and_binned_lc(ax_folded)
        self.plot_even_odd(ax_even_odd)
        self.plot_per_orbit_delta_lnlike(ax_per_orbit_lnlike)
        self.pvar.plot_model(ax_pvar)
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
                       f"$\Delta$BIC {self.dbic:5.0f} | Period  {self.period:5.2f} d | Zero epoch {self.zero_epoch:10.2f} | Depth {self.depth:5.4f} | Duration {24 * self.duration:>4.2f} h",
                       va='center', size=18, linespacing=1.75, family='monospace')
        ax.axhline(0.0, lw=8)
        setp(ax, frame_on=False, xticks=[], yticks=[])

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
    def plot_per_orbit_delta_lnlike(self, ax=None):
        ax.plot(self.transit_fits['all'].dll_epochs, self.transit_fits['all'].dll_values)
        for marker, model in zip('ox', ('even', 'odd')):
            tf = self.transit_fits[model]
            ax.plot(tf.dll_epochs, tf.dll_values, ls='', marker=marker, label=model)
        ax.axhline(0, c='k', ls='--', alpha=0.25, lw=1)
        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Epoch', ylabel="$\Delta$ log likelihood")

    @bplot
    def plot_even_odd(self, axs=None, nbins: int = 20, alpha=0.2):
        for i, ms in enumerate(('even', 'odd')):
            m = self.transit_fits[ms]
            sids = argsort(m.phase)
            phase = m.phase[sids]
            pmask = abs(phase) < 1.5 * self.duration
            phase = phase[pmask]
            fmod = m.ftra[sids][pmask]
            fobs = m.fobs[sids][pmask]

            pb, fb, eb = downsample_time(phase, fobs, phase.ptp() / nbins)
            axs[0].errorbar(24 * pb, fb, eb, fmt='o-', label=ms)
            axs[1].plot(phase, fmod, label=ms)

        for ax in axs:
            ax.legend(loc='upper right')
            ax.autoscale(axis='x', tight='true')

        setp(axs[0], ylabel='Normalized flux')
        setp(axs, xlabel='Phase [h]')

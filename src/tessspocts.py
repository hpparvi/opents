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
from logging import info
from pathlib import Path
from typing import Union, List, Optional, Dict

from astropy.io.fits import Card, getheader, Header, HDUList
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.table import Table
from matplotlib.artist import setp
from numpy import median, isfinite, argsort, ndarray, unique, concatenate
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import epoch
from pytransit.utils.misc import fold

from .transitsearch import TransitSearch
from .plots import bplot


class TESSSPOCTS(TransitSearch):
    def __init__(self, pmin: float = 0.25, pmax: Optional[float] = None, nper: int = 10000, bic_limit: float = 5, min_transits: int = 3,
                 nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):

        self.bjdrefi: int = 2457000
        self.sectors: List = []
        self.files: List = []
        self._h0: Optional[Header] = None
        self._h1: Optional[Header] = None

        self.time_raw: Optional[ndarray] = None
        self.time_detrended: Optional[ndarray] = None
        self.time_flattened: Optional[ndarray] = None

        self.flux_raw: Optional[ndarray] = None
        self.flux_detrended: Optional[ndarray] = None
        self.flux_flattened: Optional[ndarray] = None

        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    @staticmethod
    def can_read_input(source: Path) -> bool:
        """Tests whether the data files are readable by TESSTransitSearch.

        Parameters
        ----------
        source: Path
            Either a directory with data files or a single file

        Returns
        -------
            True if the files are readable, False if not.
        """
        try:
            dfile = sorted(source.glob('tess*.fits'))[0] if source.is_dir() else source
            h = getheader(dfile)
            return h['TELESCOP'] == 'TESS' and 'spoc' in h['PROCVER']
        except:
            return False

    @staticmethod
    def gather_data(source: Path, target: Optional[int] = None) -> Dict:
        """Gathers all the data files in a source directory into a dictionary

        Parameters
        ----------
        source: Path
            Either a directory with data files or a single file
        target: int, optional
            Glob pattern to select a subset of TICs

        Returns
        -------
            Dictionary of lists where each list contains data files for a single TIC.
        """
        if source.is_dir():
            tic_pattern = f'*{target}*' if target is not None else '*'
            files = sorted(source.glob(f'tess*-s*-{tic_pattern}-*s_lc.fits'))
        elif source.is_file():
            files = [source]
        else:
            raise NotImplementedError()

        tic_files = {}
        for f in files:
            tic = int(f.name.split('-')[2])
            if not tic in tic_files:
                tic_files[tic] = []
            tic_files[tic].append(f)
        return tic_files

    def _reader(self, files: Union[Path, str, List[Path]]):
        if isinstance(files, Path) or isinstance(files, str):
            files = [files]

        times, fluxes, ferrs, f2, t2 = [], [], [], [], []
        for filename in files:
            filename = Path(filename).resolve()

            tb = Table.read(filename)
            time = tb['TIME'].data.astype('d') + tb.meta['BJDREFI']
            flux = tb['PDCSAP_FLUX'].data.astype('d')
            ferr = tb['PDCSAP_FLUX_ERR'].data.astype('d')
            mask = isfinite(time) & isfinite(flux) & isfinite(ferr)
            time, flux, ferr = time[mask], flux[mask], ferr[mask]
            ferr /= median(flux)
            flux /= median(flux)
            if self._h0 is None:
                self._h0 = getheader(filename, 0)
                self._h1 = getheader(filename, 1)
                self.teff = self._h0['TEFF']

            times.append(time)
            fluxes.append(flux)
            ferrs.append(ferr)
            f2.append(tb['SAP_FLUX'].data.astype('d')[mask])
            f2[-1] /= median(f2[-1])

        time = concatenate(times)
        flux = concatenate(fluxes)
        ferr = concatenate(ferrs)

        self.time_detrended = time
        self.time_raw = time
        self.flux_detrended = flux
        self.flux_raw = concatenate(f2)
        self.mag = self._h0['TESSMAG']

        name = self._h0['OBJECT'].replace(' ', '_')
        self.logger = logging.getLogger(f"tessts:{name}")

        self.logger.info(f"Target {self._h0['OBJECT']}")
        self.logger.info(f"Read {len(files)} sectors, {time.size} points covering {time.ptp():.2f} days")
        return name, time, flux, ferr

    # FITS file output
    # ================
    # The `TransitSearch` class creates a fits file with the basic information, but we can override or
    # supplement the fits file creation methods to add TESS specific information that can be useful in
    # candidate vetting.
    #
    # Here we override the method to add the information about the star available in the SPOC TESS light
    # curve file, and the method to add the CDPP, variability, and crowding estimates from the SPOC TESS
    # light curve file.
    def _cf_add_summary_statistics(self, hdul):
        super()._cf_add_summary_statistics(hdul)
        h = hdul[0].header
        keys = "cdpp0_5 cdpp1_0 cdpp2_0 crowdsap flfrcsap pdcvar".upper().split()
        for k in keys:
            h.append(Card(k, self._h1[k], self._h1.comments[k]), bottom=True)

    def _cf_post_setup_hook(self, hdul: HDUList):
        self._cf_add_stellar_info(hdul)

    def _cf_add_stellar_info(self, hdul):
        h = hdul[0].header
        h.append(Card('COMMENT', '======================'))
        h.append(Card('COMMENT', ' Stellar information  '))
        h.append(Card('COMMENT', '======================'))
        keys = 'ticid tessmag ra_obj dec_obj teff logg radius'.upper().split()
        for k in keys:
            h.append(Card(k, self._h0[k], self._h0.comments[k]), bottom=True)

    # Plotting
    # ========
    # While the basic `TransitSearch` class implements most of the plotting methods, some should be
    # customised for the data at hand. For example, you can plot the detrended and raw light curves
    # separately in the `plot_flux_vs_time` method rather than just the flux used to do the search.

    @bplot
    def plot_flux_vs_time(self, ax=None):
        rstd = mad_std(self.flux_raw)
        dstd = mad_std(self.flux_detrended)
        offset = 4 * rstd + 4 * dstd

        ax.plot(self.time_raw - self.bjdrefi, self.flux_raw, label='raw')
        ax.plot(self.time_detrended - self.bjdrefi, self.flux + offset, label='detrended')

        if self.zero_epoch is not None:
            transits = self.zero_epoch + unique(epoch(self.time, self.zero_epoch, self.period)) * self.period
            [ax.axvline(t - self.bjdrefi, ls='--', alpha=0.5, lw=1) for t in transits]

            def time2epoch(x):
                return (x + self.bjdrefi - self.zero_epoch) / self.period

            def epoch2time(x):
                return self.zero_epoch - self.bjdrefi + x * self.period

            secax = ax.secondary_xaxis('top', functions=(time2epoch, epoch2time))
            secax.set_xlabel('Epoch')

        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel=f'Time - {self.bjdrefi} [BJD]', ylabel='Normalized flux',
             ylim=(1 - 5 * rstd, 1 + offset + 5 * dstd))

    @bplot
    def plot_flux_vs_phase(self, ax=None, n: float = 1, nbin: int = 100):
        fsap = self._flux_sap
        fpdc = self._flux_pdcsap
        phase = (fold(self.time, n * self.period, self.zero_epoch, 0.5 - ((n - 1) * 0.25)) - 0.5) * n * self.period
        sids = argsort(phase)
        pb, fb, eb = downsample_time(phase[sids], fpdc[sids], phase.ptp() / nbin)
        ax.errorbar(pb, fb, eb, fmt='.')
        phase = (fold(self.time, n * self.period, self.zero_epoch, 0.5 - ((n - 1) * 0.25)) - 0.5) * n * self.period
        sids = argsort(phase)
        pb, fb, eb = downsample_time(phase[sids], fsap[sids], phase.ptp() / nbin)
        ax.errorbar(pb, fb, eb, fmt='.')
        ax.autoscale(axis='x', tight=True)

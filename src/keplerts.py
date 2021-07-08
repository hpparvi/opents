#  OpenTS: Open exoplanet transit search pipeline.
#  Copyright (C) 2015-2021  Hannu Parviainen
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
from typing import Optional, List, Dict, Union

from astropy.io.fits import Header, getheader, getval, HDUList, Card
from astropy.table import Table
from matplotlib.artist import setp
from matplotlib.ticker import MaxNLocator
from numpy import percentile, unique, diff, median, isfinite, concatenate
from numpy.core._multiarray_umath import ndarray
from pytransit.orbits import epoch

from .transitsearch import TransitSearch
from .plots import bplot

class KeplerTS(TransitSearch):
    fnformat = 'kplr{}_llc.fits'

    def __init__(self, pmin: float = 0.25, pmax: Optional[float] = None, nper: int = 10000, bic_limit: float = 5,
                 min_transits: int = 3, nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True,
                 use_opencl: bool = True):
        self.bjdrefi: int = 2454833.0  # Kepler reference date
        self.quarters: List = []  # Kepler sectors loded
        self.files: List = []  # Files ingested
        self._h0: Optional[Header] = None
        self._h1: Optional[Header] = None
        self.tic: int = None

        # Unmodified time and flux arrays
        # -------------------------------
        self.time_raw: Optional[ndarray] = None
        self.time_detrended: Optional[ndarray] = None
        self.time_flattened: Optional[ndarray] = None

        self.flux_raw: Optional[ndarray] = None
        self.flux_detrended: Optional[ndarray] = None
        self.flux_flattened: Optional[ndarray] = None

        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)

    @classmethod
    def kic_from_name(cls, f: Path):
        return int(f.name.split('-')[0][4:])

    @classmethod
    def gather_data(cls, source: Path, target: Optional[int] = None) -> Dict:
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
            kic_pattern = f'*{target}*' if target is not None else '*'
            files = sorted(source.glob(cls.fnformat.format(kic_pattern)))
        elif source.is_file():
            files = [source]
        else:
            raise NotImplementedError()

        kic_files = {}
        for f in files:
            kic = cls.kic_from_name(f)
            if not kic in kic_files:
                kic_files[kic] = []
            kic_files[kic].append(f)
        return kic_files

    @classmethod
    def can_read_input(cls, source: Path) -> bool:
        """Tests whether the data files are readable by KeplerTransitSearch.

        Parameters
        ----------
        source: Path
            Either a directory with data files or a single file

        Returns
        -------
            True if the files are readable, False if not.
        """
        try:
            dfile = sorted(source.glob(cls.fnformat.format('*')))[0] if source.is_dir() else source
            h = getheader(dfile)
            return h['TELESCOP'] == 'Kepler' and h['ORIGIN'] == 'NASA/Ames'
        except:
            return False

    def _reader(self, files: Union[Path, str, List[Path]]):
        """Reads the data from a file or a list of files

        Parameters
        ----------
        files: Path or List[Path]

        Returns
        -------
        name, time, flux, ferr
        """
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
                self.teff = self.teff if self.teff > 2500 else 5000
            self.quarters.append(getval(filename, 'QUARTER', 0))

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
        self.mag = self._h0['KEPMAG']

        self.tic = self._h0['KEPLERID']
        name = self._h0['OBJECT'].replace(' ', '_')
        self.logger = logging.getLogger(f"keplerts:{name}")

        self.logger.info(f"Target {self._h0['OBJECT']}")
        self.logger.info(f"Read {len(files)} quarters, {time.size} points covering {time.ptp():.2f} days")
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
        h = hdul[1].header
        keys = "cdpp3_0 cdpp6_0 cdpp12_0 crowdsap flfrcsap pdcvar".upper().split()
        for k in keys:
            h.append(Card(k, self._h1[k], self._h1.comments[k]), bottom=True)

    def _cf_post_setup_hook(self, hdul: HDUList):
        self._cf_add_stellar_info(hdul)

    def _cf_add_stellar_info(self, hdul):
        h = hdul[0].header
        h.append(Card('COMMENT', '======================'))
        h.append(Card('COMMENT', ' Stellar information  '))
        h.append(Card('COMMENT', '======================'))
        keys = 'keplerid kepmag ra_obj dec_obj teff logg radius'.upper().split()
        for k in keys:
            h.append(Card(k, self._h0[k], self._h0.comments[k]), bottom=True)

    @bplot
    def plot_flux_vs_time(self, ax=None):
        rpc = percentile(self.flux_raw, [0.5, 99.5, 50])
        dpc = percentile(self.flux_detrended, [0.5, 99.5, 50])

        d = 1.15
        bbox_raw = rpc[-1] + d * (rpc[0] - rpc[-1]), rpc[-1] + d * (rpc[1] - rpc[-1])
        bbox_dtr = dpc[-1] + d * (dpc[0] - dpc[-1]), dpc[-1] + d * (dpc[1] - dpc[-1])
        offset = d * (rpc[1] - rpc[-1]) - d * (dpc[0] - dpc[-1])

        ax.plot(self.time_detrended - self.bjdrefi, self.flux_detrended + 1.2 * offset, label='detrended')
        ax.plot(self.time_raw - self.bjdrefi, self.flux_raw, label='raw')

        if self.zero_epoch:
            transits = self.zero_epoch + unique(epoch(self.time, self.zero_epoch, self.period)) * self.period
            [ax.axvline(t - self.bjdrefi, ls='--', alpha=0.5, lw=1) for t in transits if
             self.time[0] < t < self.time[-1]]

            def time2epoch(x):
                return (x + self.bjdrefi - self.zero_epoch) / self.period

            def epoch2time(x):
                return self.zero_epoch - self.bjdrefi + x * self.period

            secax = ax.secondary_xaxis('top', functions=(time2epoch, epoch2time))
            secax.set_xlabel('Epoch')
            secax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        yp_offset = diff(ax.transData.inverted().transform([[0, 0], [0, 40]])[:, 1])
        setp(ax, xlabel=f'Time - {self.bjdrefi} [BJD]', ylabel='Normalized flux',
             ylim=(bbox_raw[0] - yp_offset, bbox_dtr[1] + offset + yp_offset))
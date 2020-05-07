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
from typing import Optional, List, Dict

from astropy.io.fits import Header
from matplotlib.artist import setp
from matplotlib.ticker import MaxNLocator
from numpy import percentile, unique, diff
from numpy.core._multiarray_umath import ndarray
from pytransit.orbits import epoch

from .transitsearch import TransitSearch
from .plots import bplot


class TESSTS(TransitSearch):
    fnformat = ''

    def __init__(self, pmin: float = 0.25, pmax: Optional[float] = None, nper: int = 10000, bic_limit: float = 5,
                 min_transits: int = 3, nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):
        self.bjdrefi: int = 2457000         # TESS reference date
        self.sectors: List = []             # TESS sectors loded
        self.files: List = []               # Files ingested
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
    def tic_from_name(cls, f: Path):
        raise NotImplementedError

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
            tic_pattern = f'*{target}*' if target is not None else '*'
            files = sorted(source.glob(cls.fnformat.format(tic_pattern)))
        elif source.is_file():
            files = [source]
        else:
            raise NotImplementedError()

        tic_files = {}
        for f in files:
            tic = cls.tic_from_name(f)
            if not tic in tic_files:
                tic_files[tic] = []
            tic_files[tic].append(f)
        return tic_files

    # Plotting
    # ========
    # While the basic `TransitSearch` class implements most of the plotting methods, some should be
    # customised for the data at hand. For example, you can plot the detrended and raw light curves
    # separately in the `plot_flux_vs_time` method rather than just the flux used to do the search.

    @bplot
    def plot_flux_vs_time(self, ax=None):
        rpc = percentile(self.flux_raw, [0.5, 99.5, 50])
        dpc = percentile(self.flux_detrended, [0.5, 99.5, 50])

        d = 1.15
        bbox_raw = rpc[-1] + d * (rpc[0] - rpc[-1]), rpc[-1] + d * (rpc[1] - rpc[-1])
        bbox_dtr = dpc[-1] + d * (dpc[0] - dpc[-1]), dpc[-1] + d * (dpc[1] - dpc[-1])
        offset = d * (rpc[1] - rpc[-1]) - d * (dpc[0] - dpc[-1])

        ax.plot(self.time_detrended - self.bjdrefi, self.flux_detrended + 1.2*offset, label='detrended')
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
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
from typing import Union, List, Optional

from PIL.ImageChops import offset
from astropy.io.fits import Card, getheader, Header, diff
from astropy.stats import sigma_clipped_stats, mad_std
from matplotlib.artist import setp
from numpy import median, isfinite, argsort, ndarray, unique, array, ones, load, concatenate
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import epoch

from scipy.ndimage import binary_erosion

from .transitsearch import TransitSearch
from .plots import bplot

def read_data_directory(ddir: Path, tic: int = None, tic_pattern: str = '*'):
    if tic is not None:
        tic_pattern = f'*{tic}*'
    files = sorted(ddir.glob(f'lc_{tic_pattern}_data.npz'))
    tic_files = {}
    for f in files:
        tic = int(f.name.split('_')[1])
        if not tic in tic_files:
            tic_files[tic] = []
        tic_files[tic].append(f)
    return tic_files


class IACTESSTransitSearch(TransitSearch):
    def __init__(self, pmin: float = 0.25, pmax: float = 10., nper: int = 10000, bic_limit: float = 5,
                 nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):

        self.bjdrefi: int = 2457000
        self.sectors: List = []
        self.files: List = []
        self.sector = None
        self._h0: Optional[Header] = None
        self._h1: Optional[Header] = None

        self.time_raw: Optional[ndarray] = None
        self.time_detrended: Optional[ndarray] = None
        self.time_flattened: Optional[ndarray] = None

        self.flux_raw: Optional[ndarray] = None
        self.flux_detrended: Optional[ndarray] = None
        self.flux_flattened: Optional[ndarray] = None

        self._flux_sap: Optional[ndarray] = None
        self._flux_pdcsap: Optional[ndarray] = None
        super().__init__(pmin, pmax, nper, bic_limit, nsamples, exptime, use_tqdm, use_opencl)

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    def _reader(self, files: Union[Path, str, List[Path]]):
        if isinstance(files, Path) or isinstance(files, str):
            files = [files]

        times, fluxes, ferrs, f2, t2 = [], [], [], [], []
        for filename in files:
            filename = Path(filename).resolve()
            f = load(filename, allow_pickle=True)
            self.files.append(f)
            if self._h0 is None:
                self._h0 = f['fitsheader'][1]
                self.teff = self._h0['TEFF'][1]
                self.teff = self.teff if self.teff > 2500 else 5000

            sector = f['fitsheader'][1]['SECTOR'][1]
            self.sectors.append(sector)

            flux = f['flux_flat']
            time = f['time_flat'] + self.bjdrefi
            ferr = f['ferr_flat']

            fraw = f['flux_LC1'] / median(f['flux_LC1'])
            traw = f['time'] + self.bjdrefi

            m = f['qual'] == 0

            times.append(time)
            fluxes.append(flux)
            ferrs.append(ferr)
            t2.append(traw[m])
            f2.append(fraw[m])

        self.time_raw = concatenate(t2)
        self.time_detrended = concatenate(times)

        self.flux_raw = concatenate(f2)
        self.flux_detrended = concatenate(fluxes)

        name = self._h0['OBJECT'][1].replace(' ', '_')
        return name, concatenate(times), concatenate(fluxes), concatenate(ferrs)

    @bplot
    def plot_transit_fit(self, ax=None, full_phase: bool = False, mode='all', nbins: int = 10, alpha=0.2):
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

        if pmask.sum() < 100:
            alpha = 1

        flux_m = flux_m[sids]
        flux_o = model.obs[sids]
        ax.plot(24 * phase[pmask], flux_o[pmask], '.', alpha=alpha)
        ax.plot(24 * phase[pmask], flux_m[pmask], 'k')

        if duration > 1 / 24:
            pb, fb, eb = downsample_time(phase[pmask], flux_o[pmask], phase[pmask].ptp() / nbins)
            ax.errorbar(24 * pb, fb, eb, fmt='ok')
            ylim = fb.min() - 2 * eb.max(), fb.max() + 2 * eb.max()
        else:
            ylim = flux_o[pmask].min(), flux_o[pmask].max()

        ax.text(24 * 2.5 * hdur[0], flux_m.min(), f'$\Delta$F {1 - flux_m.min():6.4f}', size=10, va='center',
                bbox=dict(color='white'))
        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
        [ax.axvline(24 * hd, alpha=0.25, ls='-', lw=1) for hd in hdur]

        ax.autoscale(axis='x', tight='true')
        setp(ax, ylim=ylim, xlabel='Phase [h]', ylabel='Normalised flux')

    @bplot
    def plot_flux_vs_time(self, ax=None):
        rstd = mad_std(self.flux_raw)
        dstd = mad_std(self.flux_detrended)
        offset = 4*rstd + 4*dstd

        ax.plot(self.time_raw - self.bjdrefi, self.flux_raw, label='raw')
        ax.plot(self.time_detrended - self.bjdrefi, self.flux_detrended + offset, label='detrended')

        if hasattr(self, 't0'):
            transits = self.t0 + unique(epoch(self.time, self.t0, self.period)) * self.period
            [ax.axvline(t - self.bjdrefi, ls='--', alpha=0.5, lw=1) for t in transits]

            def time2epoch(x):
                return (x + self.bjdrefi - self.t0) / self.period

            def epoch2time(x):
                return self.t0 - self.bjdrefi + x * self.period

            secax = ax.secondary_xaxis('top', functions=(time2epoch, epoch2time))
            secax.set_xlabel('Epoch')

        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel=f'Time - {self.bjdrefi} [BJD]', ylabel='Normalized flux',
             ylim=(1 - 5 * rstd, 1 + offset + 5 * dstd))

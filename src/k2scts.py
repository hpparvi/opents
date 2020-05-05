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
from typing import Optional, List, Dict, Union

import astropy.io.fits as pf

from pathlib import Path

from astropy.io.fits import getheader, Undefined
from matplotlib.pyplot import setp
from numpy import isfinite, nan, nanmedian, concatenate, unique
from pytransit.orbits import epoch
from scipy.ndimage import binary_dilation, median_filter as mf

from .k2ts import K2TS
from .plots import bplot


class K2SCTS(K2TS):
    fnformat = 'EPIC_{}.fits'

    @classmethod
    def epic_from_name(cls, f: Path):
        return int(f.name.split('_')[1])

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
            dfile = str(sorted(source.glob('EPIC*.fits'))[0] if source.is_dir() else source)
            h = getheader(dfile, 1)
            return 'k2_syscor' in h['PROGRAM']
        except:
            return False

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

        epic_files = {}
        for f in files:
            epic = cls.epic_from_name(f)
            if not epic in epic_files:
                epic_files[epic] = []
            epic_files[epic].append(f)
        return epic_files

    def _reader(self, files: Union[Path, List[Path]]):
        if isinstance(files, Path) or isinstance(files, str):
            files: List[Path] = [files]

        kp = pf.getval(files[0], 'kepmag')
        kp = kp if not isinstance(kp, Undefined) else nan
        epic = int(files[0].name.split('_')[1])

        times, fluxes, ferrs, = [], [], []
        fraws, trtimes, trposis = [], [], []
        for file in files:
            d = pf.getdata(file, 1)
            time = d.time + self.bjdrefi
            quality = d.quality.copy()

            try:
                flux = d.flux.copy()
                error = d.error.copy()
                mflags = d.mflags.copy()
                ttime = d.trtime.copy()
                tposi = d.trposi.copy()
            except AttributeError:
                flux = d.flux_1.copy()
                error = d.error_1.copy()
                mflags = d.mflags_1.copy()
                ttime = d.trend_t_1.copy()
                tposi = d.trend_p_1.copy()

            m = isfinite(flux) & isfinite(time) & (~(mflags & 2 ** 3).astype(bool))
            m &= ~binary_dilation((quality & 2 ** 20) != 0)

            # Remove outliers
            # ---------------
            fl = flux.copy()
            fm = mf(fl[m], 3)
            sigma = (fl[m] - fm).std()
            m[m] &= abs(fl[m] - fm) < 5 * sigma

            time = time[m]
            flux_c = (flux[m]
                      - ttime[m] + nanmedian(tposi[m])
                      - tposi[m] + nanmedian(tposi[m]))
            mflux = nanmedian(flux_c)
            flux_c /= mflux
            ferr = error[m] / abs(mflux)

            fraws.append(flux[m] / mflux)
            trtimes.append(ttime[m] / mflux)
            trposis.append(tposi[m] / mflux)

            times.append(time)
            fluxes.append(flux_c)
            ferrs.append(ferr)

        name = f"EPIC_{epic}"
        self.mag = kp
        self.fraw, self.trposi, self.trtime = concatenate(fraws), concatenate(trposis), concatenate(trtimes)
        return name, concatenate(times), concatenate(fluxes), concatenate(ferrs)

    @bplot
    def plot_flux_vs_time(self, ax=None):
        transits = self.zero_epoch + unique(epoch(self.time, self.zero_epoch, self.period)) * self.period
        [ax.axvline(t - self.bjdrefi, ls='--', alpha=0.5, lw=1) for t in transits]
        offset1 = -1.05 * (self.trtime - self.fraw).min()
        offset2 = offset1 - 1.05 * (self.trposi - self.trtime).min()
        offset3 = offset2 - 1.05 * (self.flux - self.trposi).min()
        time = self.time - self.bjdrefi
        ax.plot(time, self.flux + offset3, label='Detrended flux')
        ax.plot(time, self.trposi + offset2, label='Rotation trend')
        ax.plot(time, self.trtime + offset1, label='Time trend')
        ax.plot(time, self.fraw, label='PDC Flux')
        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel=f'Time - {self.bjdrefi} [BJD]', ylabel='Normalized flux')

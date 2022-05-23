#  OpenTS: Open exoplanet transit search pipeline.
#  Copyright (C) 2015-2022  Hannu Parviainen
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
from typing import Union, List, Optional, Dict

from astropy.io.fits import Card, getheader, HDUList, getval
from astropy.table import Table
from numpy import median, ndarray, load, concatenate, nanmedian, isfinite, nan

from .k2ts import K2TS


class K2K2TS(K2TS):
    fnformat = 'ktwo{}-c*_llc.fits'

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    def __init__(self, pmin: float = 0.25, pmax: Optional[float] = None, nper: int = 10000, bic_limit: float = 5,
                 min_transits: int = 3, nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True,
                 use_opencl: bool = True):
        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)
        self.tic = None

    @classmethod
    def epic_from_name(cls, f: Path):
        return int(f.name.split('-')[0].split('o')[1])

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
            dfile = str(sorted(source.glob('ktwo*.fits'))[0] if source.is_dir() else source)
            h = getheader(dfile)
            return 'K2' in h['MISSION'] and 'Kepler' in h['TELESCOP']
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
            time = tb['TIME'].filled(nan).astype('d') + tb.meta['BJDREFI']
            flux = tb['PDCSAP_FLUX'].filled(nan).astype('d')
            ferr = tb['PDCSAP_FLUX_ERR'].filled(nan).astype('d')
            mask = isfinite(time) & isfinite(flux) & isfinite(ferr)
            time, flux, ferr = time[mask], flux[mask], ferr[mask]
            ferr /= median(flux)
            flux /= median(flux)

            if self._h0 is None:
                self._h0 = getheader(filename, 0)
                self._h1 = getheader(filename, 1)

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
        self.logger = logging.getLogger(f"k2k2ts:{name}")

        self.logger.info(f"Target {self._h0['OBJECT']}")
        self.logger.info(f"Read {len(files)} sectors, {time.size} points covering {time.ptp():.2f} days")

        return name, time, flux, ferr


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
from typing import Union, List, Optional, Dict

from astropy.io.fits import Header
from numpy import median, ndarray, load, concatenate

from .tessts import TESSTS


class TESSIACTS(TESSTS):
    fnformat = 'lc_{}_data.npz'

    def __init__(self, pmin: float = 0.25, pmax: Optional[float] = None, nper: int = 10000, bic_limit: float = 5, min_transits: int = 3,
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

        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    @classmethod
    def tic_from_name(cls, f: Path):
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
            dfile = str(sorted(source.glob('lc*_data.npz'))[0] if source.is_dir() else source)
            f = load(dfile)
            return 'flux_flat' in f and 'time_flat' in f
        except:
            return False


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
        self.mag = self._h0['TESSMAG'][1]

        name = self._h0['OBJECT'][1].replace(' ', '_')
        return name, concatenate(times), concatenate(fluxes), concatenate(ferrs)


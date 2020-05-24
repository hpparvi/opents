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

from astropy.io import fits
from numpy import median, ndarray, load, concatenate, nanmedian

from .tessts import TESSTS


class ELEANORTS(TESSTS):
    fnformat = 'hlsp_eleanor_tess_ffi_{}_lc.fits'

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    @classmethod
    def tic_from_name(cls, f: Path):
        return int(f.name.split("_")[4][3:])

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
            dfile = str(sorted(source.glob('hlsp_eleanor_tess_ffi_*_lc.fits'))[0] if source.is_dir() else source)
            h = fits.getheader(dfile, 1)
            cards_names = h.values()
            return "TIME" in cards_names and "CORR_FLUX" in cards_names
        except:
            return False


    def _reader(self, files: Union[Path, str, List[Path]]):
        if isinstance(files, Path) or isinstance(files, str):
            files = [files]

        times, fluxes, ferrs, f2, t2 = [], [], [], [], []
        for filename in files:
            filename = Path(filename).resolve()
            hdu_list = fits.open(filename)

            f = hdu_list[1].data
        
            self.files.append(f)
            if self._h0 is None:
                self._h0 = hdu_list[0].header
                self.teff = self._h0.get('TEFF', 5000)
                self.teff = self.teff if self.teff > 2500 else 5000

            sector = self._h0['SECTOR']
            self.sectors.append(sector)

            # what is the difference between CORR_FLUX/TIME and flux_flat/time_flat
            flux = f['CORR_FLUX'] / nanmedian(f['CORR_FLUX'])
            time = f['TIME'] + self.bjdrefi # TESS reference date
            ferr = f['FLUX_ERR']

            fraw = f['RAW_FLUX'] / median(f['RAW_FLUX'])
            traw = f['TIME'] + self.bjdrefi

            m = f['QUALITY'] == 0

            times.append(time[m])    # <-- Why don't filter by quality flag as with fraw?
            fluxes.append(flux[m])
            ferrs.append(ferr[m])
            t2.append(traw[m])
            f2.append(fraw[m])

            hdu_list.close()

        self.time_raw = concatenate(t2)
        self.time_detrended = concatenate(times)

        self.flux_raw = concatenate(f2)
        self.flux_detrended = concatenate(fluxes)
        self.mag = self._h0['TMAG']   # often is missing (?)

        name = "TIC {}".format(self._h0['TIC_ID'])
        return name, concatenate(times), concatenate(fluxes), concatenate(ferrs)


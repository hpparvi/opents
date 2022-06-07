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
from typing import Union, List, Optional, Dict

from astropy.io.fits import Card, getheader, HDUList, getval
from astropy.table import Table
from numpy import median, isfinite, concatenate, nan

from .tessts import TESSTS


class TESSSPOCTS2(TESSTS):
    fnformat = 'hlsp_tess-spoc_tess_phot_{}-s*lc.fits'

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    @classmethod
    def tic_from_name(cls, f: Path):
        return int(f.name.split('_')[4].split('-')[0])

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
            dfile = sorted(source.glob('hlsp_tess*.fits'))[0] if source.is_dir() else source
            h = getheader(dfile)
            return h['TELESCOP'].strip() == 'TESS' and 'spoc' in h['PROCVER']
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
            time = tb['TIME'].filled(nan).data.astype('d') + tb.meta['BJDREFI']
            flux = tb['PDCSAP_FLUX'].filled(nan).data.astype('d')
            ferr = tb['PDCSAP_FLUX_ERR'].filled(nan).data.astype('d')
            mask = isfinite(time) & isfinite(flux) & isfinite(ferr)
            time, flux, ferr = time[mask], flux[mask], ferr[mask]
            ferr /= median(flux)
            flux /= median(flux)
            if self._h0 is None:
                self._h0 = getheader(filename, 0)
                self._h1 = getheader(filename, 1)
                self.teff = self._h0['TEFF']
                self.teff = self.teff if self.teff > 2500 else 5000
            self.sectors.append(getval(filename, 'sector', 0))

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

        self.tic = self._h0['TICID']
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


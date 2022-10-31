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
from matplotlib.artist import setp
from matplotlib.ticker import MaxNLocator
from numpy import median, ndarray, load, concatenate, nanmedian, percentile, unique, diff
from pytransit.orbits import epoch

from .blsstep import BLSStep
from .eleanorstep import EleanorStep
from .lsstep import LombScargleStep
from .plots import bplot
from .pvarstep import PVarStep
from .tessts import TESSTS
from .tfstep import TransitFitStep


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

    def initialize_steps(self):
        """Initialize the pipeline steps.

        Returns
        -------
        None
        """
        self.ls      = self.register_step(LombScargleStep(self))
        self.pvar    = self.register_step(PVarStep(self))
        self.cvar    = self.register_step(EleanorStep(self))
        self.bls     = self.register_step(BLSStep(self))
        self.tf_all  = self.register_step(TransitFitStep(self, 'all', ' Transit fit results  ', use_tqdm=self.use_tqdm))
        self.tf_even = self.register_step(TransitFitStep(self, 'even', ' Even tr. fit results  ', use_tqdm=self.use_tqdm))
        self.tf_odd  = self.register_step(TransitFitStep(self, 'odd', ' Odd tr. fit results  ', use_tqdm=self.use_tqdm))

    @bplot
    def plot_flux_vs_time(self, ax=None):
        rpc = percentile(self.flux_raw, [0.5, 99.5, 50])
        dpc = percentile(self._data.flux, [0.5, 99.5, 50])

        d = 1.15
        bbox_raw = rpc[-1] + d * (rpc[0] - rpc[-1]), rpc[-1] + d * (rpc[1] - rpc[-1])
        bbox_dtr = dpc[-1] + d * (dpc[0] - dpc[-1]), dpc[-1] + d * (dpc[1] - dpc[-1])
        bbox_dtr = 0.99*self._data.flux.min(), 1.01*self._data.flux.max()
        offset = d * (rpc[1] - rpc[-1]) - d * (dpc[0] - dpc[-1])

        ax.plot(self._data.time - self.bjdrefi, self._data.flux, label='detrended',zorder=1)
        #ax.plot(self.time_detrended - self.bjdrefi, self.flux_detrended + 1.2*offset, label='detrended')
        ax.plot(self.time_raw - self.bjdrefi, self.flux_raw, label='raw', zorder=-1, alpha=0.15)

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
        setp(ax, xlabel=f'Time - {self.bjdrefi} [BJD]', ylabel='Normalized flux', ylim=bbox_dtr)

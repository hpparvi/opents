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
from logging import getLogger

from astropy import units as u
from astropy.io.fits import HDUList, Card
from astropy.timeseries import BoxLeastSquares
from matplotlib.pyplot import setp
from numpy import linspace, argmax, array, exp
from pytransit.orbits import epoch

from .otsstep import OTSStep
from .plots import bplot

def maskf(x, c, w):
    return (1 - exp(-(x-c)**2/w**2))**4

class BLSStep(OTSStep):
    name = "bls"
    def __init__(self, ts):
        super().__init__(ts)
        self._periods = None
        self._durations = array([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) / 24
        self.bls = None
        self.result = None
        self.period = None      # Best-fit period
        self.zero_epoch = None  # Best-fit zero epoch
        self.duration = None    # Best-fit duration
        self.depth = None       # Best-fit depth
        self.snr = None         # Best-fit Signal to noise ratio

    def __call__(self, *args, **kwargs):
        self.logger = getLogger(f"{self.name}:{self.ts.name.lower().replace('_','-')}")
        self.logger.info("Running BLS periodogram")
        self._periods = linspace(self.ts.pmin, self.ts.pmax, self.ts.nper)
        self.bls = BoxLeastSquares(self.ts.time * u.day, self.ts.flux, self.ts.ferr)
        self.result = self.bls.power(self._periods, self._durations, objective='snr')
        for p in self.ts.masked_periods:
            self.result.depth_snr *= maskf(self._periods, p, .1)
            self.result.log_likelihood *= maskf(self._periods, p, .1)
        i = argmax(self.result.depth_snr)
        self.period = self.result.period[i].value
        self.snr = self.result.depth_snr[i]
        self.duration = self.result.duration[i].value
        self.depth = self.result.depth[i]
        t0 = self.result.transit_time[i].value
        ep = epoch(self.ts.time.min(), t0, self.period)
        self.zero_epoch = t0 + ep * self.period
        self.ts.update_ephemeris(self.zero_epoch, self.period, self.duration, self.depth)
        self.logger.info(f"BLS SNR {self.snr:.2f} period {self.period:.2f} d, duration {24*self.duration:.2f} h")

    def add_to_fits(self, hdul: HDUList):
        if self.bls is not None:
            h = hdul[0].header
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', '     BLS results      '))
            h.append(Card('COMMENT', '======================'))
            h.append(Card('bls_snr', self.snr, 'BLS depth signal to noise ratio'), bottom=True)
            h.append(Card('period', self.period, 'Orbital period [d]'), bottom=True)
            h.append(Card('epoch', self.zero_epoch, 'Zero epoch [BJD]'), bottom=True)
            h.append(Card('duration', self.duration, 'Transit duration [d]'), bottom=True)
            h.append(Card('depth', self.depth, 'Transit depth'), bottom=True)

    @bplot
    def plot_snr(self, ax=None):
        if self.period < 1.:
            ax.semilogx(self._periods, self.result.depth_snr, drawstyle='steps-mid')
        else:
            ax.plot(self._periods, self.result.depth_snr, drawstyle='steps-mid')

        ax.axvline(self.period, alpha=0.15, c='orangered', ls='-', lw=10, zorder=-100)
        setp(ax, xlabel='Period [d]', ylabel='Depth SNR')
        ax.autoscale(axis='x', tight=True)

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

from astropy.io.fits import HDUList, Card
from astropy.timeseries import LombScargle
from matplotlib.pyplot import setp
from numpy import linspace, argmax

from .otsstep import OTSStep
from .plots import bplot

logger = getLogger("lomb-scargle")

class LombScargleStep(OTSStep):
    """Lomb-Scargle step.

    Pipeline step to calculate the Lomb-Scargle periodogram.
    """
    def __init__(self, ts):
        super().__init__(ts)

        self.period = None  # Best-fit period
        self.power = None   # Power of the best-fit period
        self.fap = None     # False alarm probability for the best-period period

        self.ls = None
        self._periods = 1 / linspace(1 / self.ts.pmax, 1 / self.ts.pmin, 1000)
        self._powers = None
        self._faps = None

    def __call__(self):
        logger.info("Running Lomb-Scargle periodogram")
        self.ls = LombScargle(self.ts.time, self.ts.flux)
        self._powers = self.ls.power(1 / self._periods)
        self._faps = self.ls.false_alarm_probability(self._powers)
        i = argmax(self._powers)
        self.period = self._periods[i]
        self.power = self._powers[i]
        self.fap = self._faps[i]
        logger.info(f"LS period {self.period:.2f}, power {self.power:.4f}, FAP {self.fap:.4f}")

    def add_to_fits(self, hdul: HDUList):
        if self.ls is not None:
            h = hdul[0].header
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', ' Lomb-Scargle results '))
            h.append(Card('COMMENT', '======================'))
            h.append(Card('lsper', self.period, 'Lomb-Scargle period [d]'), bottom=True)
            h.append(Card('lspow', self.power, 'Lomb-Scargle power'), bottom=True)
            h.append(Card('lsfap', self.fap, 'Lomb-Scargle false alarm probability'), bottom=True)

    @bplot
    def plot_power(self, ax=None):
        ax.semilogx(self._periods, self._powers, drawstyle='steps-mid')
        ax.autoscale(axis='x', tight=True)
        ax.text(0.81, 0.93, f"Period {self.period:5.2f} d\nFAP    {self.fap:5.2f}", va='top',
                family='monospace', transform=ax.transAxes, bbox=dict(facecolor='w'))
        setp(ax, xlabel='Period [d]', ylabel='LS Power')

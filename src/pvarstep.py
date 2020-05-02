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

from matplotlib.pyplot import setp
from numba import njit
from numpy import linspace, zeros_like, median, diff, percentile, fabs, log, argsort, ones, where, nan
from pytransit.utils.misc import fold
from scipy.interpolate import interp1d

from .otsstep import OTSStep
from .plots import bplot

@njit
def running_median(xs, ys, width, nbin):
    bx = linspace(0, 1, nbin)
    by = zeros_like(bx)
    hwidth = 0.5 * width
    for i, x in enumerate(bx):
        m = fabs(xs - x) < hwidth
        if x < hwidth:
            m |= xs > x + 1 - hwidth
        if x > 1 - hwidth:
            m |= xs < x - 1 + hwidth
        by[i] = median(ys[m])
    return bx, by


class PVarStep(OTSStep):
    name = 'pvar'
    def __call__(self):
        self.logger = getLogger(f"{self.name}:{self.ts.name.lower().replace('_','-')}")
        self.logger.info("Testing for a significant non-transit-like periodic variability")

        phase = fold(self.ts.time, self.ts.ls.period)
        bp, bf = running_median(phase, self.ts.flux, 0.05, 100)
        pv = interp1d(bp, bf)(phase)

        df = abs(diff(pv) / diff(self.ts.time))
        ps = percentile(df, [80, 99.5])

        pv_amplitude = bf.ptp()
        bfn = -bf
        bfn -= bfn.min() - 1e-12
        bfn /= bfn.sum()
        pv_entropy = -(bfn * log(bfn)).sum()

        max_slope = df.max()
        slope_percentile_ratio = ps[0] / ps[1]

        self.pvar_phase = phase * self.ts.ls.period
        self.pvar_model = pv
        self.pvar_spr = slope_percentile_ratio
        self.pvar_amplitude = pv_amplitude
        self.pvar_entropy = pv_entropy
        self.pvar_max_slope = max_slope

        self.pvar_is_sigificant = pv_amplitude > 0.001
        self.pvar_is_transit_like = slope_percentile_ratio < 0.1  # Conservative limit to make sure we're not removing transits

        if self.pvar_is_sigificant and not self.pvar_is_transit_like:
            self.logger.info("Found and removed a periodic signal")
            self.model_removed = True
            flux = self.ts.flux - self.pvar_model + 1
            #self.ts.update_data('pvarstep', self.ts.time, flux, self.ts.ferr)
        else:
            self.model_removed = False
            if self.pvar_is_sigificant:
                self.logger.info("Found a periodic signal but it's too transit-like to be removed")
            else:
                self.logger.info("Didn't find any significant periodic signals")

    @bplot
    def plot_over_phase(self, ax):
        sids = argsort(self.pvar_phase)
        if self.model_removed:
            ax.plot(self.pvar_phase[sids], 1e2 * (self.pvar_model[sids] - 1))
        else:
            ax.plot(self.pvar_phase[sids], 1e2 * (self.pvar_model[sids] - 1), '--', alpha=0.5)
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Phase [d]', ylabel='Periodic model [%]')

    def plot_over_time(self, ax):
        mbreak = ones(self.ts.time.size, bool)
        mbreak[1:] = diff(self.ts.time) < 1
        ax.plot(self.ts.time - self.ts.bjdrefi, where(mbreak, self.ts.flux_detrended, nan))
        ax.plot(self.ts.time - self.ts.bjdrefi, where(mbreak, self.pvar_model, nan), 'k')
        setp(ax, xlabel=f'Time - {self.ts.bjdrefi} [BJD]')
        ax.autoscale(axis='x', tight=True)
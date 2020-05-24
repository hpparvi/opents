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
from numpy import linspace, zeros_like, median, diff, percentile, fabs, log, argsort, ones, where, nan, argmax, \
    concatenate, tile, array, ndarray
from numpy.random.mtrand import uniform
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

def dip_significance(phase: ndarray, flux: ndarray, p0: float = None, tdur: float = 3, nsamples: int = 150, nbl: int = 500):
    """Calculates the significance of a dip in a phase-folded light curve.

    Calculates how significant a dip in a phase-folded light curve is considering the overall variability by comparing
    the dip "flux" integrated over a given time-span against other time spans excluding the dip.

    Parameters
    ----------
    phase
    flux
    p0
    tdur
    nsamples
    nbl

    Returns
    -------

    """
    flux = -(flux - flux.max())
    flux /= flux.sum()
    period = phase.ptp()
    if tdur > 0.3*period:
        tdur = 0.3*period
    p0 = p0 if p0 is not None else phase[argmax(flux)]
    phase = phase - p0
    pphase = concatenate([phase-period, phase, phase+period])
    pflux = tile(flux, 3)
    ip = interp1d(pphase, pflux)
    xs = linspace(-0.5*tdur, 0.5*tdur, nbl)
    samples = uniform(tdur, period-tdur, size=nsamples)
    s0 = ip(xs).sum() / tdur
    sbl = array([ip(xs+tc).sum() / tdur for tc in samples])
    return (s0-sbl.mean()) / sbl.std()


class PVarStep(OTSStep):
    name = 'pvar'

    def __init__(self, ts):
        super().__init__(ts)
        self.time_before = None
        self.flux_before = None
        self.time_after = None
        self.flux_after = None
        self.model = None
        self.phase = None

    def __call__(self):
        self.logger = getLogger(f"{self.name}:{self.ts.name.lower().replace('_','-')}")
        self.logger.info("Testing for a significant non-transit-like periodic variability")

        self.time_before = self.ts.time
        self.flux_before = self.ts.flux
        self.time_after = self.ts.time

        phase = fold(self.time_before, self.ts.ls.period)
        bp, bf = running_median(phase, self.flux_before, 0.05, 100)
        pv = interp1d(bp, bf)(phase)

        df = abs(diff(pv) / diff(self.time_before))
        ps = percentile(df, [80, 99.5])

        pv_amplitude = bf.ptp()
        bfn = -bf
        bfn -= bfn.min() - 1e-12
        bfn /= bfn.sum()
        pv_entropy = -(bfn * log(bfn)).sum()

        max_slope = df.max()
        slope_percentile_ratio = ps[0] / ps[1]

        self._bp = bp
        self._bf = bf

        self.phase = phase * self.ts.ls.period
        self.model = pv
        self.spr = slope_percentile_ratio
        self.amplitude = pv_amplitude
        self.entropy = pv_entropy
        self.max_slope = max_slope
        self.peak_siginifance = dip_significance(24 * self.ts.ls.period * bp, bf)

        self.is_sigificant = pv_amplitude > 0.001
        self.is_transit_like = (slope_percentile_ratio < 0.1 or self.peak_siginifance > 3.0)

        if self.is_sigificant and not self.is_transit_like:
            self.logger.info("Found and removed a periodic signal")
            self.model_removed = True
            flux = self.ts.flux - self.model + 1
            self.flux_after = flux
            self.ts.update_data('pvarstep', self.ts.time, flux, self.ts.ferr)
        else:
            self.flux_after = self.ts.flux
            self.model_removed = False
            if self.is_sigificant:
                self.logger.info("Found a periodic signal but it's too transit-like to be removed")
            else:
                self.logger.info("Didn't find any significant periodic signals")

    @bplot
    def plot_over_phase(self, ax):
        sids = argsort(self.phase)
        if self.model_removed:
            ax.plot(self.phase[sids], 1e2 * (self.model[sids] - 1))
        else:
            ax.plot(self.phase[sids], 1e2 * (self.model[sids] - 1), '--', alpha=0.5)
        ax.autoscale(axis='x', tight=True)
        setp(ax, xlabel='Phase [d]', ylabel='Periodic model [%]')

    def plot_over_time(self, ax):
        mbreak = ones(self.time_before.size, bool)
        mbreak[1:] = diff(self.time_before) < 1
        ax.plot(self.time_before - self.ts.bjdrefi, where(mbreak, self.flux_before, nan))
        ax.plot(self.time_before - self.ts.bjdrefi, where(mbreak, self.model, nan), 'k')
        setp(ax, xlabel=f'Time - {self.ts.bjdrefi} [BJD]', ylabel='Normalized flux')
        ax.autoscale(axis='x', tight=True)
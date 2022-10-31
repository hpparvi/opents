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

from logging import getLogger
from typing import Optional
from george import GP
from george.kernels import Matern32Kernel as M32
from astropy.stats import mad_std, sigma_clip
from numpy import ones, log, pi, sqrt, diff, inf, zeros_like, median, ones_like, ceil, where
from scipy.signal import medfilt

import matplotlib.pyplot as pl

from .otsstep import OTSStep

logger = getLogger("celerite-step")


def create_mask(t, f, m=None, dt=0.5):
    m_final = m if m is not None else ones(t.size, bool)

    # Mask blocks of data with weird slopes
    # -------------------------------------
    nd = int(ceil(t[-1] - t[0]) / dt)

    diff_stds = zeros_like(f)
    flux_stds = zeros_like(f)
    for i in range(nd):
        t1 = t[0] + dt * i
        t2 = t1 + dt
        m = (t >= t1) & (t < t2)
        flux_stds[m] = mad_std(f[m])
        diff_stds[m] = mad_std(abs(diff(f[m])))
    m_ds  = flux_stds < 2 * median(flux_stds)
    m_ds &= diff_stds < 2 * median(diff_stds)
    if any(m_ds[:20] == 0):
        m_ds[:20] = 0
    if any(m_ds[-100:] == 0):
        m_ds[-100:] = 0

    #fig, axs = pl.subplots(1, 2, sharey='all')
    #axs[0].plot(f-1)
    #axs[0].plot(flux_stds, 'k')
    #axs[0].axhline(1.5 * median(flux_stds), c='k')
    #axs[1].plot(f-1)
    #axs[1].plot(diff_stds, 'k')
    #axs[1].axhline(1.5 * median(diff_stds), c='k')
    #pl.show()

    m_final &= m_ds

    m_neg = f > 0.5
    m_final &= m_neg

    m_out = ones_like(m_final)
    fm = f[m_final] - medfilt(f[m_final], 9)
    m_out[m_final] = fm < 5 * mad_std(fm)
    m_final &= m_out
    return m_final


def detrend(t, f, m):
    k = f[m].var() * M32(4)
    wn = diff(f[m]).std() / sqrt(2)
    gp = GP(k, mean=1.0, white_noise=log(wn**2))
    gp.compute(t[m])
    fp = gp.predict(f[m], t, return_cov=False)
    return f / fp


class EleanorStep(OTSStep):
    def __init__(self, ts):
        super().__init__(ts)
        self.mask = None
        self.prediction = None

    def __call__(self, rho: Optional[float] = 2., niter: int = 5):
        logger.info("Running Eleanor detrending")

        time = self.ts.time.copy()
        flux = self.ts.flux.copy()
        mask = ones(time.size, bool)
        detrended_flux = zeros_like(flux)

        breaks = where(diff(time) > 0.5)[0]
        slices = [slice(None, breaks[0] + 1)]
        for i in range(breaks.size - 1):
            slices.append(slice(breaks[i] + 1, breaks[i + 1] + 1))
        slices.append(slice(breaks[-1] + 1, None))

        for sl in slices:
            mf = create_mask(time[sl], flux[sl])
            if mf.sum() > 10:
                f2 = detrend(time[sl], flux[sl], mf)
                detrended_flux[sl] = f2
            mf &= detrended_flux[sl] > 0.5
            mask[sl] = mf

        self.mask = mask
        self.ts._data.update('eleanor', time[mask], detrended_flux[mask], self.ts.ferr[mask])
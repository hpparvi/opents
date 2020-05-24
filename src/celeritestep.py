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
from typing import Optional

from astropy.io.fits import HDUList, Card
from astropy.stats import mad_std, sigma_clip
from celerite import GP
from celerite.terms import SHOTerm, Matern32Term
from numpy import ones, log, pi, sqrt, diff, inf
from scipy.optimize import minimize

from .otsstep import OTSStep

logger = getLogger("celerite-step")


class CeleriteM32Step(OTSStep):
    def __init__(self, ts):
        super().__init__(ts)
        self.mask = None
        self.prediction = None

    def __call__(self, rho: Optional[float] = 1.):
        logger.info("Running Celerite Matern 3/2 detrending")

        time = self.ts.time.copy()
        flux = self.ts.flux.copy()
        mask = ones(time.size, bool)

        for i in range(3):
            time_learn = time[mask]
            flux_learn = flux[mask]

            wn = mad_std(diff(flux_learn)) / sqrt(2)
            log_sigma = log(mad_std(flux_learn))
            log_rho = log(rho)

            kernel = Matern32Term(log_sigma, log_rho)
            gp = GP(kernel, mean=1.0)
            gp.compute(time_learn, yerr=wn)
            self.prediction = gp.predict(flux_learn, time, return_cov=False)
            mask &= ~sigma_clip(flux - self.prediction, sigma=5).mask

        residuals = flux - self.prediction
        self.mask = m = ~(sigma_clip(residuals, sigma_lower=inf, sigma_upper=5).mask)
        self.ts._data.update('celerite_m32', time[m], (flux - self.prediction + 1)[m], self.ts.ferr[m])


class CeleriteSHOTStep(OTSStep):
    def __init__(self, ts):
        super().__init__(ts)
        self.result = None
        self.parameters = None
        self.prediction = None
        self.period = None

    def __call__(self, period: Optional[float] = None):
        logger.info("Running Celerite")
        assert hasattr(self.ts, 'ls'), "Celerite step requires a prior Lomb-Scargle step"
        self.period = period if period is not None else self.ts.ls.period

        gp = GP(SHOTerm(log(self.ts.flux.var()), log(10), log(2 * pi / self.period)), mean=1.)
        gp.freeze_parameter('kernel:log_omega0')
        gp.compute(self.ts.time, yerr=self.ts.ferr)

        def minfun(pv):
            gp.set_parameter_vector(pv)
            gp.compute(self.ts.time, yerr=self.ts.ferr)
            return -gp.log_likelihood(self.ts.flux)

        res = minimize(minfun, gp.get_parameter_vector(), jac=False, method='powell')
        self.result = res
        self.parameters = res.x
        self.prediction = gp.predict(self.ts.flux, return_cov=False)
        #self.flux -= self.gp_prediction - 1.


    def add_to_fits(self, hdul: HDUList, *nargs, **kwargs):
        if self.result is not None:
            h = hdul[0].header
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', ' Celerite periodicity '))
            h.append(Card('COMMENT', '======================'))
            h.append(Card('GPPLS', self.parameters[0], 'GP SHOT term log S0'), bottom=True)
            h.append(Card('GPPLQ', self.parameters[1], 'GP SHOT term log Q'), bottom=True)
            h.append(Card('GPPLO', log(2 * pi / self.period), 'GP SHOT term log omega'), bottom=True)
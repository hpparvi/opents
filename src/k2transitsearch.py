import astropy.io.fits as pf

from pathlib import Path
from numpy import isfinite, nan, nanmedian
from scipy.ndimage import binary_dilation, median_filter as mf

from .transitsearch import TransitSearch


class K2TransitSearch(TransitSearch):

    def _reader(self, filename: Path):
        d = pf.getdata(filename, 1)
        time = d.time.copy() + 2454833.
        quality = d.quality.copy()

        try:
            flux = d.flux.copy()
            error = d.error.copy()
            mflags = d.mflags.copy()
            ttime = d.trtime.copy()
            tposi = d.trposi.copy()
        except AttributeError:
            flux = d.flux_1.copy()
            error = d.error_1.copy()
            mflags = d.mflags_1.copy()
            ttime = d.trend_t_1.copy()
            tposi = d.trend_p_1.copy()

        m = isfinite(flux) & isfinite(time) & (~(mflags & 2 ** 3).astype(bool))
        m &= ~binary_dilation((quality & 2 ** 20) != 0)

        # Remove outliers
        # ---------------
        fl = flux.copy()
        fm = mf(fl[m], 3)
        sigma = (fl[m] - fm).std()
        m[m] &= abs(fl[m] - fm) < 5 * sigma

        # Remove exluded regions
        # ----------------------
        for emin, emax in self.excluded_ranges:
            m[(time > emin) & (time < emax)] = 0

        kp = pf.getval(filename, 'kepmag')
        kp = kp if not isinstance(kp, pf.Undefined) else nan

        epic = int(filename.name.split('_')[1])
        time = time[m]
        flux_c = (flux[m]
                  - ttime[m] + nanmedian(tposi[m])
                  - tposi[m] + nanmedian(tposi[m]))
        mflux = nanmedian(flux_c)
        flux_c /= mflux
        ferr = error[m] / abs(mflux)

        self.flux_r = flux[m] / mflux
        self.trtime = ttime[m] / mflux
        self.trposi = tposi[m] / mflux

        return time, flux_c, ferr

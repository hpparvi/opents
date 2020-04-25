from pathlib import Path
from typing import Union, List, Optional

from astropy.io.fits import Card, getheader, Header
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from matplotlib.artist import setp
from numpy import median, isfinite, argsort, ndarray, unique
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits import epoch
from pytransit.utils.misc import fold

from .transitsearch import TransitSearch


class TESSTransitSearch(TransitSearch):
    def __init__(self, pmin: float = 0.25, pmax: float = 15., nper: int = 10000, snr_limit: float = 3,
                 nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):
        self.sectors: List = []
        self._h0: Optional[Header] = None
        self._h1: Optional[Header] = None
        self._flux_sap: Optional[ndarray] = None
        self._flux_pdcsap: Optional[ndarray] = None
        super().__init__(pmin, pmax, nper, snr_limit, nsamples, exptime, use_tqdm, use_opencl)

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    def _reader(self, filename: Union[Path, str]):
        filename = Path(filename).resolve()
        tb = Table.read(filename)
        time = tb['TIME'].data.astype('d') + tb.meta['BJDREFI']
        flux = tb['PDCSAP_FLUX'].data.astype('d')
        ferr = tb['PDCSAP_FLUX_ERR'].data.astype('d')
        mask = isfinite(time) & isfinite(flux) & isfinite(ferr)
        time, flux, ferr = time[mask], flux[mask], ferr[mask]
        ferr /= median(flux)
        flux /= median(flux)
        self._h0 = getheader(filename, 0)
        self._h1 = getheader(filename, 1)
        self._flux_pdcsap = flux
        self._flux_sap = tb['SAP_FLUX'].data.astype('d')[mask]
        self._flux_sap /= median(self._flux_sap)
        name = self._h0['OBJECT'].replace(' ', '_')
        self.teff = self._h0['TEFF']
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

    def _cf_add_stellar_info(self, hdul):
        h = hdul[0].header
        h.append(Card('COMMENT', '======================'))
        h.append(Card('COMMENT', ' Stellar information  '))
        h.append(Card('COMMENT', '======================'))
        keys = 'ticid tessmag ra_obj dec_obj teff logg radius'.upper().split()
        for k in keys:
            h.append(Card(k, self._h0[k], self._h0.comments[k]), bottom=True)

    def _cf_add_summary_statistics(self, hdul):
        super()._cf_add_summary_statistics(hdul)
        h = hdul[0].header
        keys = "cdpp0_5 cdpp1_0 cdpp2_0 crowdsap flfrcsap pdcvar".upper().split()
        for k in keys:
            h.append(Card(k, self._h1[k], self._h1.comments[k]), bottom=True)

    # Plotting
    # ========
    # While the basic `TransitSearch` class implements most of the plotting methods, some should be
    # customised for the data at hand. For example, you can plot the detrended and raw light curves
    # separately in the `plot_flux_vs_time` method rather than just the flux used to do the search.

    def plot_flux_vs_time(self, ax=None):
        tref = 2457000

        ax.text(0.013, 0.96, "Raw and detrended flux", size=15, va='top', transform=ax.transAxes,
                bbox={'facecolor': 'w', 'edgecolor': 'w'})

        transits = self.t0 + unique(epoch(self.time, self.t0, self.period)) * self.period
        [ax.axvline(t - tref, ls='--', alpha=0.5, lw=1) for t in transits]

        fsap = self._flux_sap
        fpdc = self._flux_pdcsap
        fpdc = fpdc + 1 - fpdc.min() + 3 * fsap.std()

        ax.plot(self.time - tref, fpdc, label='PDC')
        tb, fb, eb = downsample_time(self.time - tref, fpdc, 1 / 24)
        ax.plot(tb, fb, 'k', lw=1)

        ax.plot(self.time - tref, fsap, label='SAP')
        tb, fb, eb = downsample_time(self.time - tref, fsap, 1 / 24)
        ax.plot(tb, fb, 'k', lw=1)

        def time2epoch(x):
            return (x + tref - self.t0) / self.period

        def epoch2time(x):
            return self.t0 - tref + x * self.period

        secax = ax.secondary_xaxis('top', functions=(time2epoch, epoch2time))
        secax.set_xlabel('Epoch')

        ax.legend(loc='upper right')
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim((ax.get_ylim()[0], median(fpdc) + 5 * sigma_clipped_stats(fpdc)[2]))
        setp(ax, xlabel=f'Time - {tref} [BJD]', ylabel='Normalized flux')

    def plot_flux_vs_phase(self, ax=None, n: float = 1, nbin: int = 100):
        fsap = self._flux_sap
        fpdc = self._flux_pdcsap
        phase = (fold(self.time, n * self.period, self.t0, 0.5 - ((n - 1) * 0.25)) - 0.5) * n * self.period
        sids = argsort(phase)
        pb, fb, eb = downsample_time(phase[sids], fpdc[sids], phase.ptp() / nbin)
        ax.errorbar(pb, fb, eb, fmt='.')
        phase = (fold(self.time, n * self.period, self.t0, 0.5 - ((n - 1) * 0.25)) - 0.5) * n * self.period
        sids = argsort(phase)
        pb, fb, eb = downsample_time(phase[sids], fsap[sids], phase.ptp() / nbin)
        ax.errorbar(pb, fb, eb, fmt='.')
        ax.autoscale(axis='x', tight=True)

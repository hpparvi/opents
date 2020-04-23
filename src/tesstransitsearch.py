from pathlib import Path
from typing import Union

from astropy.io.fits import Card, getheader, PrimaryHDU
from astropy.stats import mad_std
from astropy.table import Table
from numpy import median, isfinite, percentile, argsort
from pytransit.lpf.tesslpf import downsample_time
from pytransit.utils.misc import fold

from .transitsearch import TransitSearch


class TESSTransitSearch(TransitSearch):
    def __init__(self, pmin: float = 0.25, pmax: float = 15., nper: int = 10000, snr_limit: float = 3,
                 nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):

        self.bls_result = None
        self.ls_result = None

        self._h0 = None
        self._h1 = None
        self._flux_sap = None
        self._flux_pdcsap = None

        super().__init__(pmin, pmax, nper, snr_limit, nsamples, exptime, use_tqdm, use_opencl)

    @property
    def basename(self):
        return self.name

    def save_fits(self, savedir: Path):
        raise NotImplementedError

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
        self.name = self._h0['OBJECT'].replace(' ', '_')
        self._flux_pdcsap = flux
        self._flux_sap = tb['SAP_FLUX'].data.astype('d')[mask]
        self._flux_sap /= median(self._flux_sap)
        return time, flux, ferr

    def _create_fits(self):
        phdu = PrimaryHDU()
        h = phdu.header

        h.append(Card('name', self.name))
        h.append(Card('ticid', self._h0['TICID'], self._h0.comments['TICID']))
        h.append(Card('pmin', self.pmin, 'Minimum search period [d]'))
        h.append(Card('pmax', self.pmax, 'Maximum search period [d]'))
        h.append(Card('nper', self.nper, 'Period grid size'))
        h.append(Card('dper', (self.pmax - self.pmin) / self.nper, 'Period grid step size [d]'))

        h.append(Card('COMMENT', '======================'))
        h.append(Card('COMMENT', ' Stellar information '))
        h.append(Card('COMMENT', '======================'))

        keys = 'tessmag ra_obj dec_obj teff logg radius'.upper().split()
        for k in keys:
            h.append(Card(k, self._h0[k], self._h0.comments[k]), bottom=True)

        h.append(Card('COMMENT', '======================'))
        h.append(Card('COMMENT', ' Summary statistics  '))
        h.append(Card('COMMENT', '======================'))

        keys = "cdpp0_5 cdpp1_0 cdpp2_0 crowdsap flfrcsap pdcvar".upper().split()
        for k in keys:
            h.append(Card(k, self._h1[k], self._h1.comments[k]), bottom=True)

        h.append(Card('fstd', self.flux.std(), "Flux standard deviation"))
        h.append(Card('fmadstd', mad_std(self.flux), "Flux MAD standard deviation"))

        ps = [0.1, 1, 5, 95, 99, 99.9]
        pvs = percentile(self.flux, ps)
        pks = [f"fpc{int(10 * p):03d}" for p in ps]

        for p, pk, pv in zip(ps, pks, pvs):
            h.append(Card(pk, pv, f"{p:4.1f} normalized flux percentile"), bottom=True)

        if self.bls_result is not None:
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', '     BLS results     '))
            h.append(Card('COMMENT', '======================'))

            h.append(Card('bls_snr', self.snr, 'BLS depth signal to noise ratio'), bottom=True)
            h.append(Card('period', self.period, 'Orbital period [d]'), bottom=True)
            h.append(Card('epoch', self.zero_epoch, 'Zero epoch [BJD]'), bottom=True)
            h.append(Card('duration', self.duration, 'Transit duration [d]'), bottom=True)
            h.append(Card('depth', self.bls_result.depth, 'Transit depth'), bottom=True)

        if self.ls_result is not None:
            h.append(Card('COMMENT', '======================'))
            h.append(Card('COMMENT', ' Lomb-Scargle results '))
            h.append(Card('COMMENT', '======================'))
            h.append(Card('lsper', self.ls_result.period, 'Lomg-Scargle period [d]'), bottom=True)
            h.append(Card('lspow', self.ls_result.power, 'Lomg-Scargle power'), bottom=True)
            h.append(Card('lsfap', self.ls.false_alarm_probability(self.ls_result.power),
                          'Lomg-Scargle false alarm probability'), bottom=True)

        return phdu

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

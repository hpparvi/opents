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

from astropy.io.fits import Header
from astropy.visualization import simple_norm
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.pyplot import subplot, setp, figure
from matplotlib.transforms import offset_copy
from numpy import median, ndarray, load, concatenate, array

from .tessts import TESSTS


class TESSIACTS(TESSTS):
    fnformat = 'lc_{}_data.npz'

    # Data input
    # ==========
    # The `TransitSearch`class doesn't implement the method for reading in the data. This is the absolute
    # minimum any subclass needs to implement for the class to function.
    #
    @classmethod
    def tic_from_name(cls, f: Path):
        return int(f.name.split('_')[1])

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
            dfile = str(sorted(source.glob('lc*_data.npz'))[0] if source.is_dir() else source)
            f = load(dfile)
            return 'flux_flat' in f and 'time_flat' in f
        except:
            return False

    def set_plot_parameters(self):
        self._tf_plot_nbins: int = 20

    def _reader(self, files: Union[Path, str, List[Path]]):
        if isinstance(files, Path) or isinstance(files, str):
            files = [files]

        times, fluxes, ferrs, f2, t2 = [], [], [], [], []
        for filename in files:
            filename = Path(filename).resolve()
            f = load(filename, allow_pickle=True)
            self.files.append(f)
            if self._h0 is None:
                self._h0 = f['fitsheader'][1]
                self.teff = self._h0['TEFF'][1]
                self.teff = self.teff if self.teff > 2500 else 5000

            sector = f['fitsheader'][1]['SECTOR'][1]
            self.sectors.append(sector)

            m = (f['flux_flat'] > 0.0) & (f['flux_flat'] < 2.0)

            flux = f['flux_flat'][m]
            time = f['time_flat'][m] + self.bjdrefi
            ferr = f['ferr_flat'][m]

            fraw = f['flux_LC1'] / median(f['flux_LC1'])
            traw = f['time'] + self.bjdrefi

            m = f['qual'] == 0

            times.append(time)
            fluxes.append(flux)
            ferrs.append(ferr)
            t2.append(traw[m])
            f2.append(fraw[m])

        self.time_raw = concatenate(t2)
        self.time_detrended = concatenate(times)

        self.flux_raw = concatenate(f2)
        self.flux_detrended = concatenate(fluxes)
        self.mag = self._h0['TESSMAG'][1]
        self.tic = int(files[0].name.split('_')[1])

        name = f"TIC_{self.tic:011d}"
        return name, concatenate(times), concatenate(fluxes), concatenate(ferrs)

    def plot_header(self, ax):
        ax.text(0.01, 0.85, f"{self.name.replace('_', ' ')} - Candidate {self.planet}", size=33, va='top', weight='bold')
        mags = f"{self.mag:3.1f}" if self.mag else '   '
        teffs = f"{self.teff:5.0f}" if self.teff else '     '
        s = (
            f"$\Delta$BIC {self.dbic:6.1f} | SNR {self.bls.snr:6.1f}   | Period    {self.period:5.2f} d | T$_0$ {self.zero_epoch:10.2f} | Depth {self.depth:5.4f} | Duration {24 * self.duration:>4.2f} h\n"
            f"Mag     {mags} | TEff {teffs} K | LS Period {self.ls.period:5.2f} d | LS FAP {self.ls.fap:5.4f} | Period limits  {self.pmin:5.2f} – {self.pmax:5.2f} d")
        ax.text(0.01, 0.55, s, va='top', size=15.6, linespacing=2, family='monospace')

    def plot_report(self):
        def sb(*nargs, **kwargs) -> Axes:
            return subplot(*nargs, **kwargs)

        lmargin = 0.06
        rmargin = 0.01

        headheight = 0.11
        headpad = 0.04
        footheight = 0.03
        footpad = 0.04

        fig = figure(figsize=(16, 19))
        gs = GridSpec(8, 4, figure=fig, wspace=0.4, hspace=0.5,
                      left=lmargin, right=1 - rmargin, bottom=footheight + footpad, top=1 - headheight - headpad)

        header = fig.add_axes((0, 1 - headheight, 0.8, headheight), frame_on=False, xticks=[], yticks=[])
        footer = fig.add_axes((0, 0, 1, footheight), frame_on=False, xticks=[], yticks=[])

        # Cutout
        # ------
        gs_cutout = GridSpec(1, 2, figure=fig, wspace=0.03, hspace=0.03,
                             left=0.81, right=1 - rmargin, bottom=1 - headheight, top=1)
        ax_cutout = subplot(gs_cutout[0])
        ax_aperture = subplot(gs_cutout[1])
        d = self.files[0]['pixel_images'][0]
        n = simple_norm(d, stretch='log')
        ax_cutout.matshow(d, norm=n, cmap=cm.gray)
        ax_aperture.matshow(self.files[0]['aperture'], cmap=cm.gray)
        setp((ax_cutout, ax_aperture), xticks=[], yticks=[])

        # Flux over time
        # --------------
        aflux = sb(gs[0:2, :])
        self.plot_flux_vs_time(aflux)

        # Periodograms
        # ------------
        gs_periodograms = gs[2:4, :2].subgridspec(2, 1, hspace=0.05)
        abls = sb(gs_periodograms[0])
        als = sb(gs_periodograms[1], sharex=abls)

        self.bls.plot_snr(abls)
        abls.text(0.02, 1, 'Periodograms', va='center', ha='left', transform=abls.transAxes, size=11,
                  bbox=dict(facecolor='w'))
        tra = offset_copy(abls.transAxes, fig=fig, x=-10, y=10, units='points')
        abls.text(1, 0, 'BLS', va='bottom', ha='right', transform=tra, size=11, bbox=dict(facecolor='w'))

        self.ls.plot_power(als)
        tra = offset_copy(als.transAxes, fig=fig, x=-10, y=10, units='points')
        als.text(1, 0, 'Lomb-Scargle', va='bottom', ha='right', transform=tra, size=11, bbox=dict(facecolor='w'))

        # Per-orbit log likelihood difference
        # -----------------------------------
        gs_dll = gs[4:6, :2].subgridspec(2, 1, hspace=0.05)
        adll = sb(gs_dll[0])
        adllc = sb(gs_dll[1], sharex=adll)

        self.plot_dll(adll)
        self.plot_cumulative_dll(adllc)

        adll.text(0.02, 1, 'Per-orbit $\Delta$ log likelihood', va='center', transform=adll.transAxes, size=11,
                  bbox=dict(facecolor='w'))
        setp(adll.get_xticklabels(), visible=False)
        setp(adll, xlabel='')

        # Periodic variability
        # --------------------
        apvp = sb(gs[6, :2])
        apvt = sb(gs[6, 2:])

        self.pvar.plot_over_phase(apvp)
        apvp.text(0.02, 1, 'Periodic variability', va='center', transform=apvp.transAxes, size=11,
                  bbox=dict(facecolor='w'))

        self.pvar.plot_over_time(apvt)
        apvt.text(0.02, 1, 'Periodic variability', va='center', transform=apvt.transAxes, size=11,
                  bbox=dict(facecolor='w'))

        # Transit and orbit
        # -----------------
        gs_to = gs[2:5, 2:].subgridspec(2, 1)
        atr = sb(gs_to[0])
        aor = sb(gs_to[1], sharey=atr)

        self.tf_all.plot_transit_fit(atr, nbins=self._tf_plot_nbins)
        self.plot_folded_orbit(aor)

        atr.text(0.02, 1, 'Phase-folded transit', va='center', transform=atr.transAxes, size=11,
                 bbox=dict(facecolor='w'))
        aor.text(0.02, 1, 'Phase-folded orbit', va='center', transform=aor.transAxes, size=11, bbox=dict(facecolor='w'))

        # Even ad odd transit fits
        # ------------------------
        gs3 = gs[5, 2:].subgridspec(1, 2, wspace=0.05)
        ato = sb(gs3[0])
        atm = sb(gs3[1], sharey=ato)

        self.plot_even_odd((ato, atm))
        ato.text(0.02, 1, 'Even and odd transits', va='center', transform=ato.transAxes, size=11,
                 bbox=dict(facecolor='w'))
        setp(atm.get_yticklabels(), visible=False)

        # Parameter posteriors
        # --------------------
        gs_posteriors = gs[7, :].subgridspec(3, 5, hspace=0, wspace=0.01)
        aposteriors = array([[subplot(gs_posteriors[j, i]) for i in range(5)] for j in range(3)])
        self.plot_posteriors(aposteriors)
        aposteriors[0, 0].text(0.02, 1, 'Parameter posteriors', va='center', transform=aposteriors[0, 0].transAxes, size=11,
                               bbox=dict(facecolor='w'))

        fig.lines.extend([
            Line2D([0, 1], [1, 1], lw=10, transform=fig.transFigure, figure=fig),
            Line2D([0, 1], [1 - headheight, 1 - headheight], lw=10, transform=fig.transFigure, figure=fig),
            Line2D([0, 1], [footheight, footheight], lw=5, transform=fig.transFigure, figure=fig),
            Line2D([0, 1], [0, 0], lw=10, transform=fig.transFigure, figure=fig)
        ])

        footer.text(0.5, 0.5, 'Open Exoplanet Transit Search Pipeline', ha='center', va='center', size='large')
        self.plot_header(header)

        fig.align_ylabels([aflux, abls, als, adll, adllc, apvp])
        fig.align_ylabels([atr, aor, ato, apvt])
        return fig
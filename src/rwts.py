#  OpenTS: Open exoplanet transit search pipeline.
#  Copyright (C) 2015-2025  Hannu Parviainen
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
from typing import Optional, Dict

from scipy.ndimage import label
from scipy.signal import medfilt

from .transitsearch import TransitSearch
from numpy import load, median, ones, diff, ones_like

class RWTS(TransitSearch):
    def _reader(self, file: Path | str):
        file = Path(file)
        time, flux, ferr = load(file)
        nflux = flux / median(flux)

        m = abs(diff(medfilt(nflux, 5))) < 0.004
        labels = ones(flux.size, dtype=int)
        labels[1:], nl = label(m)
        cflux = ones_like(flux)
        for i in range(1, nl+1):
            m = labels == i
            cflux[m] = flux[m]/median(flux[m])

        self.bjdrefi = 0
        self.mag = 0
        self.tic = 0
        self.teff = 5000
        self.sectors = [0]
        self.time_raw = time
        self.time_detrended = time.copy()

        self.flux_raw = flux
        self.flux_detrended = cflux

        name = f"ts_{file.stem}"
        return name, time.copy(), cflux, ferr.copy()

    @staticmethod
    def can_read_input(source: Path | str) -> bool:
        source = Path(source)
        return 'in' in source.name and source.suffix == '.npy'

    @classmethod
    def gather_data(cls, source: Path, target: Optional[int] = None) -> Dict:
        if source.is_dir():
            raise ValueError('RWTS should be run only on a single file, not a directory.')
        elif source.is_file():
            files = [source]
        else:
            raise NotImplementedError()

        return {'a': files[0]}

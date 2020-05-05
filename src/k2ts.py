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
from typing import Optional, List, Dict

from astropy.io.fits import Header
from numpy.core._multiarray_umath import ndarray

from .transitsearch import TransitSearch


class K2TS(TransitSearch):
    fnformat = ''

    def __init__(self, pmin: float = 0.25, pmax: Optional[float] = None, nper: int = 10000, bic_limit: float = 5,
                 min_transits: int = 3, nsamples: int = 1, exptime: float = 0.0, use_tqdm: bool = True, use_opencl: bool = True):
        self.bjdrefi: int = 2454833         # Kepler reference date
        self.campaigns: List = []           # K2 Campaigns loaded
        self.files: List = []               # Files ingested
        self._h0: Optional[Header] = None
        self._h1: Optional[Header] = None

        # Unmodified time and flux arrays
        # -------------------------------
        self.time_raw: Optional[ndarray] = None
        self.time_detrended: Optional[ndarray] = None
        self.time_flattened: Optional[ndarray] = None

        self.flux_raw: Optional[ndarray] = None
        self.flux_detrended: Optional[ndarray] = None
        self.flux_flattened: Optional[ndarray] = None

        super().__init__(pmin, pmax, nper, bic_limit, min_transits, nsamples, exptime, use_tqdm, use_opencl)

    @classmethod
    def epic_from_name(cls, f: Path):
        raise NotImplementedError

    @classmethod
    def gather_data(cls, source: Path, target: Optional[int] = None) -> Dict:
        """Gathers all the data files in a source directory into a dictionary

        Parameters
        ----------
        source: Path
            Either a directory with data files or a single file
        target: int, optional
            Glob pattern to select a subset of EPICs

        Returns
        -------
            Dictionary of lists where each list contains data files for a single EPIC.
        """
        if source.is_dir():
            epic_pattern = f'*{target}*' if target is not None else '*'
            files = sorted(source.glob(cls.fnformat.format(epic_pattern)))
        elif source.is_file():
            files = [source]
        else:
            raise NotImplementedError()

        tic_files = {}
        for f in files:
            tic = cls.epic_from_name(f)
            if not tic in tic_files:
                tic_files[tic] = []
            tic_files[tic].append(f)
        return tic_files
#  OpenTS: Open exoplanet transit search pipeline.
#  Copyright (C) 2015-2021  Hannu Parviainen
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
import logging
from math import sqrt

import pandas as pd

from pathlib import Path

from astropy.stats import mad_std
from numpy import ones_like, diff, full_like
from opents import TransitSearch
from typing import Union, List, Optional, Dict

from .celeritestep import CeleriteM32Step, CeleriteSHOTStep
from .plots import bplot
from .blsstep import BLSStep
from .lsstep import LombScargleStep
from .pvarstep import PVarStep
from .tfstep import TransitFitStep


class PlatoTestTransitSearch(TransitSearch):
    fnformat = 'plato_{}.csv'
    bjdrefi = 0.0

    def initialize_steps(self):
        """Initialize the pipeline steps.

        Returns
        -------
        None
        """
        self.ls = self.register_step(LombScargleStep(self))
        self.pvar = self.register_step(PVarStep(self))
        # self.cvar    = self.register_step(CeleriteM32Step(self))
        self.bls = self.register_step(BLSStep(self))
        self.tf_all = self.register_step(TransitFitStep(self, 'all', ' Transit fit results  ', use_tqdm=self.use_tqdm))
        self.tf_even = self.register_step(TransitFitStep(self, 'even', ' Even tr. fit results  ', use_tqdm=self.use_tqdm))
        self.tf_odd = self.register_step(TransitFitStep(self, 'odd', ' Odd tr. fit results  ', use_tqdm=self.use_tqdm))

    @classmethod
    def name_from_filename(cls, f: Path):
        return f.stem

    @classmethod
    def gather_data(cls, source: Path, target: Optional[str] = None) -> Dict:
        """Gathers all the data files in a source directory into a dictionary

        Parameters
        ----------
        source: Path
            Either a directory with data files or a single file
        target: int, optional
            Glob pattern to select a subset of TICs

        Returns
        -------
            Dictionary of lists where each list contains data files for a single TIC.
        """
        if source.is_dir():
            target_pattern = f'*{target}*' if target is not None else '*'
            files = sorted(source.glob(cls.fnformat.format(target_pattern)))
        elif source.is_file():
            files = [source]
        else:
            raise NotImplementedError()

        target_files = {}
        for f in files:
            name = cls.name_from_filename(f)
            if not name in target_files:
                target_files[name] = []
            target_files[name].append(f)
        return target_files

    @staticmethod
    def can_read_input(source: Path) -> bool:
        """Tests whether the data files are readable by PlatoTestTransitSearch.

        Parameters
        ----------
        source: Path
            A single data file

        Returns
        -------
            True if the files are readable, False if not.
        """

        try:
            if source.is_dir():
                source = sorted(source.glob('plato*.csv'))[0]
            df = pd.read_csv(source)
            df.time
            df.fdet
            return True
        except:
            return False

    def _reader(self, file: Union[Path, str, List]):
        """Reads the data from a file or a list of files

        Parameters
        ----------
        file: Path

        Returns
        -------
        name, time, flux, ferr
        """

        if isinstance(file, list):
            file = file[0]

        df = pd.read_csv(file)
        time = df.time.values.copy()
        flux = 1.0 + df.fdet.values.copy()
        ferr = full_like(flux, mad_std(diff(flux)) / sqrt(2))
        name = file.stem

        self.logger = logging.getLogger(f"platotestts:{name}")
        self.logger.info(f"Target {name}")
        return name, time, flux, ferr

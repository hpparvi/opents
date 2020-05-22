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

from .transitsearch import TransitSearch
from .k2scts import K2SCTS
from .tessspocts import TESSSPOCTS
from .tessiacts import TESSIACTS
from .eleanorts import ELEANORTS

ts_classes = (TESSIACTS, TESSSPOCTS, ELEANORTS, K2SCTS)

def select_ts_class(data_source: Path):
    """Selects the correct transit search class given an input data file or directory

    Parameters
    ----------
    data_source: Path
        Data file or directory with data files

    Returns
    -------
        TransitSearch subclass specialized for the input data
    """
    for tsc in ts_classes:
        if tsc.can_read_input(data_source):
            return tsc
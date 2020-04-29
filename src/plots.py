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

from functools import wraps
from typing import Callable

from matplotlib.pyplot import subplots


def bplot(plotf: Callable):
    @wraps(plotf)
    def wrapper(self, ax=None, *args, **kwargs):
        if ax is None:
            fig, ax = subplots(1, 1)
        else:
            fig, ax = None, ax
        try:
            plotf(self, ax, **kwargs)
        except ValueError:
            pass
        return fig

    return wrapper
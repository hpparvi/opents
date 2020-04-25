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

from setuptools import setup
from version import version
long_description = 'Open exoplanet transit search pipeline.'

setup(name='OpenTS',
      version=version,
      description='Open transit search pipeline.',
      long_description=long_description,
      author='Hannu Parviainen',
      author_email='hannu@iac.es',
      url='',
      package_dir={'opents':'src'},
      packages=['opents'],
      scripts=['bin/tessts'],
      install_requires=["numpy", "pytransit"],
      package_data={'': ['data/*']},
      include_package_data=True,
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python"
      ]
     )

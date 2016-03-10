from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

long_description = ''


setup(name='K2PS',
      version='0.5',
      description='K2 planet search toolkit.',
      long_description=long_description,
      author='Hannu Parviainen',
      author_email='hannu.parviainen@physics.ox.ac.uk',
      url='',
      package_dir={'k2ps':'src'},
      scripts=['bin/k2inject','bin/k2train','bin/k2search','bin/k2aux'],
      extra_options = ['-fopenmp'],
      packages=['k2ps'],
      ext_modules=[Extension('k2ps.blsf', ['src/bls.f90'], libraries=['gomp','m'])],
      install_requires=["numpy"],
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

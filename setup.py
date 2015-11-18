from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

long_description = ''

setup(name='K2PS',
      version='0.5',
      description='K2 planet search toolkit.',
      long_description=long_description,
      author='Suzanne Aigrain',
      author_email='',
      url='',
      package_dir={'k2ps':'src'},
      scripts=['bin/k2inj'],
      packages=['k2ps'],
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

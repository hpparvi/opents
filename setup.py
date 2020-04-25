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

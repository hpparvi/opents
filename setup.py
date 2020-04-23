from setuptools import setup

long_description = 'K2 Planet Search: code to search transit-like signals from the k2sc-detrended K2 light curves.'

setup(name='K2PS',
      version='0.5',
      description='K2 planet search toolkit.',
      long_description=long_description,
      author='Hannu Parviainen',
      author_email='hannu@iac.es',
      url='',
      package_dir={'k2ps':'src'},
      packages=['k2ps'],
      scripts=['bin/k2search'],
      install_requires=["numpy", "PyTransit"],
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

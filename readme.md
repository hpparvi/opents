# Open exoplanet transit search pipeline

Open exoplanet transit search pipeline (OpenTS) is a Python package for searching exoplanet
transit signals from photometric light curves. OpenTS currently supports K2SC and SPOC TESS 
light curves, but more input sources will be supported soon.

The pipeline is a successor to the *"Oxford K2 planet search pipeline"* that was used in 
Pope et al. (2016).

## Installation

    python setup.py install

## Usage

At its simplest, the transit search can be run as

    opents data_directory

or as 

    mpirun -n X opents data_directory
    
where ``X`` is the number of nodes to use.

By default, the search is carried our for all the targets in ``data_directory``,
combining all the TESS sectors or K2 Campaigns for each target. 

## Authors

- Hannu Parviainen
- Benjamin Pope
- Suzanne Aigrain

# HODfit2sim: HOD model fitted to model galaxies

This python code corresponds to the work presented in [GRP2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231213199R/abstract) and VGP2024 (in prep.). It provides a halo occupation distribution (HOD) model with parameters chosen to reproduce the clustering of a sample of model tracers. These can be either galaxies, QSOs or clouds of gas.

To use this code, acknowledge the source work by adding references to [GRP2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231213199R/abstract).


## Construct a sample of shuffled tracers

To generate a file with both the positions and velocities of the tracers, together with those for the DM haloes, run **main.py** with get_sample set to True.
```
python3 main.py
```

At the moment, the code does not include a modelling for assembly bias, and thus, the input sample needs to be shuffled. This means that model tracers populating a halo of a given mass are randomly placed on the position of a different halo with a mass within a set range.

In GRP2024, the shuffling was done in bins of 0.057 dex of halo mass, using M200c within the UNIT simulation.


## Characterise the sample of tracers

To characterise the sample of tracers, run **main.py** with get_params set to True.

There are several quantities that need to be measured to characterise the sample of tracers.

* Shape of the mean HOD for central and satellite tracers. This requires counting the total number of DM haloes in each mass bin.
* The global conformity factors from the mean HOD for satellites with and without a central tracer of the same type. If the global factors are not a good description of the tracers, try using the factors in mass bins, with flags [....].
* Radial profile of satellite tracers. If the modified NFW profile is not a good description, try using directly the measured average radial profile.


## Populate haloes with tracers with the HOD model

To populate DM haloes with the HOD model, run **main.py** with run_HOD set to True. This:

1. Reads the information calculated for the sample of tracers and DM haloes.
2. Runs the HOD model with the given parameters.


# Installation

Requirements: python 3 and its main packages all installable through pip: astropy, numpy, scipy, matplotlib, math ...

```
git clone git@github.com:computationalAstroUAM/HODfit2sim.git
```

## Liscence

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
# HOD model fitting the clustering from model galaxies

This python code corresponds to the work presented in [GRP2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231213199R/abstract) and VGP2024 (in prep.). It provides a halo occupation distribution (HOD) model with parameters chosen to reproduce the clustering of a sample of model tracers. These can be either galaxies, QSOs or clouds of gas.

To use this code, acknowledge the source work by adding references to [GRP2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231213199R/abstract).


## Construct a sample of shuffled tracers
At the moment, the code does not include a modelling for assembly bias, and thus, the input sample needs to be shuffled. This means that model tracers populating a halo of a given mass are randomly placed on the position of a different halo with a mass within a set range.

In GRP2024, the shuffling was done in bins of 0.057 dex of halo mass ($10.5<log_{q0}(M_{\rm 200c}/)<14.5$), within the UNIT simulation.

## Characterise the dark matter (DM) haloes in the simulation

* Measure the number of haloes per mass bins, to be used for the shape of the mean HODs.

## Characterise the sample of tracers

There are several quantities that need to be measured to characterise the sample of tracers. In several instances, the number of dark matter 

* Shape of the mean HOD for central and satellite tracers.
* The global conformity factors from the mean HOD for satellites with and without a central tracer of the same type. If the global factors are not a good description of the tracers, try using the factors in mass bins, with flags [....].
* Radial profile of satellite tracers. If the modified NFW profile is not a good description, try using directly the measured average radial profile.

## Construct an HOD model

1. Read the information calculated for the sample of tracers and DM haloes.
2. Run the HOD model with the given parameters.


# Installaling and running the code

Requirements: python 3 and its main packages all installable through pip: astropy, numpy, scipy, matplotlib, math ...

```
git clone https://github.com/...
```

## How to run the code

All the executables are within the folder **HODfit2sim**.

Adapt **main.py** to your simulation and run it:

```
python3 main.py
```


## Liscence

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
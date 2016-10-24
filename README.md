# *PyHillFit* - python code to perform Bayesian inference of Hill curve parameters from dose-response data
Code to load dose-response data and fit dose Hill response curves in a Bayesian inference framework.

![schematic of PyHillFit inputs and outputs](https://github.com/mirams/PyHillFit/blob/master/schematic.png)

## Pre-requisites

The following python packages are pre-requisites for running `PyHillFit`:
 * [numpy](http://www.numpy.org/)
 * [cma](https://www.lri.fr/~hansen/cmaes_inmatlab.html#python)
 * [matplotlib](http://matplotlib.org/)
 * [scipy](https://www.scipy.org/)
 * [pandas](http://pandas.pydata.org/)
 
On most linux distributions you can install these via `pip`, which itself can be installed, if it isn't already present, following the instructions [on the pip homepage](https://pip.pypa.io/en/latest/installing/).

Then all the above dependencies can be installed in one go with:
```
sudo pip install numpy cma matplotlib scipy pandas
```

## Crumb dataset

We have made a .csv file of the [Crumb et al.](http://dx.doi.org/10.1016/j.vascn.2016.03.009) dataset, which is available in the `data` folder, together with some example python scripts for reading it. You can fit your own data by putting them into a similar format to this `.csv` file. Note that doses/concentrations should be given in microMolar.

## Running PyHillFit

To run the python-based dose-response fitting code, see the [README](python/README.md) in the `python` folder.

## Uncertainty Propagation

To run the Uncertainty Propagation example based on `PyHillFit` output, see the [README](chaste/README.md) in the `chaste` folder.

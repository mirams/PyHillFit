# Chaste code

For performing the uncertainty propagation simulations.

## Pre-requisites

 * `Chaste` - from www.github.com/Chaste/Chaste (following the installation instructions at https://chaste.cs.ox.ac.uk/trac/wiki/InstallGuides/InstallGuide)
 * `ApPredict` - from www.github.com/Chaste/ApPredict

The files in this folder should be copied into `ApPredict/test`.

## Contents

 * `samples/*` - contains samples from hierarchical inference results (based on running the scripts in other folders).
 * `drug_list.txt` - a simple text file telling the Chaste code which concentrations to use for which compounds.
 * `CrumbDrugList.hpp` - a helper class to read in `drug_list.txt`
 * `CrumbDataReader.hpp` - a helper class to read in the samples from `samples/*`
 * `TestCrumbPredictions.hpp` - the main test to run.

## Running the simulations

Should be as simple as getting a copy of Chaste:
```
git clone https://github.com/Chaste/Chaste.git
cd Chaste
git checkout 4dc2ad3264b92b6d1871cf4277e591a1fa70ebf7
cd ..
```
Then a copy of ApPredict:
```
git clone --recursive https://github.com/Chaste/ApPredict.git
cd ApPredict
git checkout a38aafa02ee21c2825bc14e5efc07c7ef8626b01
cd ..
```
Here we have hardcoded two particular revisions of Chaste and ApPredict (c. 18th October 2016) that are known to work with these files.

Then link ApPredict so it can be built as a Chaste project:
```
cd Chaste/projects
ln -s ../../ApPredict
```
Then copy all the contents of the folder this README is in, to:
```
ApPredict/test/
```
(you could put it somewhere else, but there are hard-coded paths that assume this is where the data files will be)
and run the simulations from the Chaste folder, with:
```
scons cl=1 b=GccOptNative projects/ApPredict/test/TestCrumbPredictions.hpp
```
## Visualizing the results

The results can be visualized into the histogram using the matlab script `process_results.m`.










# Running PyHillFit

You need to be inside the python directory (this directory) for this to work.

## Single-level and hierarchical MCMC

The MCMC script is
```
PyHillFit.py
```
which is called from the command line. If it runs without any arguments it will display a list of required and optional command line arguments.

You HAVE to specify an input data csv file, in the same format as the provided crumb_data.csv.
The default is to run the single-level MCMC algorithm.
To do this with the Crumb data, for example, run
```
python PyHillFit.py --data-file ../data/crumb_data.csv
```

Alternatively, include the --hierarchical argument:
```
python PyHillFit.py --data-file ../data/crumb_data.csv --hierarchical
```

There is an optional argument --fix-hill to fix Hill=1, and only vary pIC50 and sigma when doing the MCMC.
The default, not including this argument, allows Hill to vary.
```
python PyHillFit.py --data-file ../data/crumb_data.csv --fix-hill
```

After running the hierarchical MCMC, you should run:
```
python construct_hierarchical_cdfs.py
```
and select from the drug/channel menu, which will compute the posterior predictive distributions for pIC50_i and Hill_i.
After you have run ALL of these, you can plot a comparison of predicted dose-response curves:
```
python plot_all_predicted_curves.py -c 0.05 2
```
where "-c 0.05 2" means "plot predictions for % block at concentrations of 0.05 and 5 uM", you can do more or fewer than two, but the plots will get messy.  Again, you'll have to choose drug and channel from the list provided.

After only hierarchical (as in, you don't need non-hierarchical for this), you can plot superimposed marginal distributions and predicted dose-response curves using the mid-level parameters:
```
python plot_hierarchical_superimposed.py
```
and again select from the menu.

Everything will be saved in `python/output/...` plus a very verbose and hopefully-easy-to-follow chain of subdirectories.

## Input data files --data-file

Input data should be a csv file in the same format as data/crumb_data.csv, with one header line: "Compound,Channel,Experiment,Dose(uM),Response(%inhibition)".
Each data file can have as many compounds and channels as required. Note that doses should be given in microMolar, as the conversion to pIC50 is hard-coded to these units.

# Bayes Factor calculations with PyHillTemp

PyHillTemp runs the same MCMC code as (non-hierarchical) PyHillFit, except at different temperatures, i.e. the likelihood is raised to the power t (the temperature) where 0 <= t <= 1.

t = 1 is the full likelihood and therefore "normal" MCMC.

These different temperatures are for approximating Bayes factors using thermodynamic integration.

"Temp" = "Temperature", not "Temporary"!

A bash script like this runs for a range of compounds and channels:
```
for channel in 0 1 2 3 <...>
do
    for drug in 0 1 2 3 <...>
    do
          for model in 1 2
          do
              python python/PyHillTemp.py --data-file ~/data.csv -m ${model} -a -nc 16 -d ${drug} -c ${channel}
          done
          python python/compute_bayes_factors.py --data-file ~/data.csv -d ${drug} -c ${channel}
    done
done
```

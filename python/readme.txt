You need to be inside the python directory for this.

To run non-hierarchical MCMC, from command line:

python dose_response_mcmc.py -h

will actually give you the list of optional arguments.  You can just run:

python dose_response_mcmc.py

and you will be asked to choose drugs and channels from the input data file. You'll be presented with a list, and you can choose multiple options.

Similarly, for hierarchical MCMC:

python hierarchical_dose_response_mcmc_logistics.py

After running the hierarchical MCMC, you should run:

python construct_hierarchical_cdfs.py

and select from the drug/channel menu.

which will compute the posterior predictive distributions for pIC50_i and Hill_i

After you have run ALL of these, you can plot a comparison of predicted dose-response curves:

python plot_all_predicted_curves.py -c 0.05 2

where "-c 0.05 2" means "plot predictions for % block at concentrations of 0.05 and 5 uM", you can do more or fewer than two, but the plots will get messy.  Again, you'll have to choose drug and channel from the list provided.

After only hierarchical (as in, you don't need non-hierarchical for this), you can plot superimposed marginal distributions and predicted dose-response curves using the mid-level parameters:

python plot_hierarchical_superimposed.py

and again select from the menu.

Everything will be saved in python/output/ + a very verbose and hopefully easy-to-follow chain of subdirectories.

After running the hierarchical MCMC, (alpha,mu) samples will be automatically saved into chaste/samples/ for later use in AP predictions.

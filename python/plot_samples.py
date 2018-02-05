import doseresponse as dr
import numpy as np
import numpy.random as npr
import argparse
import itertools as it
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv", required=True)
parser.add_argument("-n", "--num-samples", type=int, help="number of MCMC samples with which to plot dose-response curves", required=True)
parser.add_argument("--hierarchical", action='store_true', help="take samples from hierarchical MCMC (single-level if this option is omitted)", default=False)
parser.add_argument("-m", "--model", type=int, help="For non-hierarchical: 1. fix Hill=1; 2. vary Hill", required=True)
parser.add_argument("--all", action='store_true', help='run hierarchical MCMC on all drugs and channels', default=False)
parser.add_argument("--plot-data", action='store_true', help='plot data points on top of predicted curves', default=False)
parser.add_argument("--save-pdf", action='store_true', help='save dose-response curves figure as pdf (in addition to png), but the file will probably be massive', default=False)
args = parser.parse_args()

dr.define_model(args.model)
temperature = 1
num_params = dr.num_params

dr.setup(args.data_file)
drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

num_x_pts = 50
alpha = 0.002  # this is the lowest value I've found that actually shows anything

for drug, channel in it.product(drugs_to_run, channels_to_run):

    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    drug, channel, chain_file, images_dir = dr.nonhierarchical_chain_file_and_figs_dir(args.model, drug, channel, temperature)
    
    concs = np.array([])
    responses = np.array([])
    for i in xrange(num_expts):
        concs = np.concatenate((concs,experiments[i][:,0]))
        responses = np.concatenate((responses,experiments[i][:,1]))
        
    min_x = int(np.log10(np.min(concs)))-1
    max_x = int(np.log10(np.max(concs)))+2
    
    try:
        chain = np.loadtxt(chain_file)
    except:
        print "\nCan't find/load chain file for {} + {}, model {} --- skipping\n".format(drug, channel, args.model)
        continue
    saved_its, d = chain.shape
    
    rand_idx = npr.randint(0, saved_its, args.num_samples)  # PyHillFit discards burn-in before saving chain to .txt file
    
    chain = chain[rand_idx, :2]
    if args.model==1:
        chain[:, 1] = 1.
    
    samples_file, samples_png, samples_pdf = dr.samples_file(drug, channel, args.model, args.hierarchical, args.num_samples, temperature)
    
    # could technically save 50% of the space for Model 1 by not bothering to save Hill=1 in every sample...
    with open(samples_file, "w") as outfile:
        outfile.write("# {} (pIC50,Hill) samples from single-level MCMC (model {}) for {} + {}\n".format(args.num_samples, args.model, drug, channel))
        np.savetxt(outfile, chain)
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    ax.grid()
    ax.set_xlabel("{} concentration ($\mu$M)".format(drug))
    ax.set_ylabel("% {} block".format(channel))
    ax.set_xscale("log")
    x = np.logspace(min_x, max_x, num_x_pts)
    
    for t in xrange(args.num_samples):
        pic50, hill = chain[t, :]
        predicted_response_curve = dr.dose_response_model(x, hill, dr.pic50_to_ic50(pic50))
        ax.plot(x, predicted_response_curve, color='black', alpha=alpha)
    if args.plot_data:
        ax.plot(concs, responses, 'o', color='orange', ms=8, zorder=10)
    fig.tight_layout()
    fig.savefig(samples_png)
    print "\nSaved {}\n".format(samples_png)
    if args.save_pdf:
        fig.savefig(samples_pdf)
        print "\nSaved {}\n".format(samples_pdf)
        
    plt.close()


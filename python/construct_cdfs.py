import doseresponse as dr
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy.random as npr
import argparse
import itertools as it

# get rid of for real version
import pandas as pd
import os

seed = 1
npr.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--samples", type=int, help="number of Hill and pIC50 samples for use in AP model",default=500)
parser.add_argument("-a", "--all", action='store_true', help='construct posterior predictive CDFs for Hill and pIC50 for all drugs and channels', default=False)

args = parser.parse_args()

drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

def construct_posterior_predictive_cdfs(alphas,betas,mus,ss):
    num_x_pts = 501
    hill_min = 0.
    hill_max = 12.
    pic50_min = -2.
    pic50_max = 12.
    hill_x_range = np.linspace(hill_min,hill_max,num_x_pts)
    pic50_x_range = np.linspace(pic50_min,pic50_max,num_x_pts)
    num_iterations = len(alphas) # assuming burn already discarded
    hill_cdf_sum = np.zeros(num_x_pts)
    pic50_cdf_sum = np.zeros(num_x_pts)
    fisk = st.fisk.cdf
    logistic = st.logistic.cdf
    for i in xrange(num_iterations):
        hill_cdf_sum += fisk(hill_x_range,c=betas[i],scale=alphas[i],loc=0)
        pic50_cdf_sum += logistic(pic50_x_range,mus[i],ss[i])
    hill_cdf_sum /= num_iterations
    pic50_cdf_sum /= num_iterations
    return hill_x_range, hill_cdf_sum, pic50_x_range, pic50_cdf_sum

def run(drug,channel):
    
    drug, channel, output_dir, chain_dir, figs_dir, chain_file = dr.hierarchical_output_dirs_and_chain_file(drug,channel)
    mcmc = np.loadtxt(chain_file,usecols=range(4))
    total_iterations = mcmc.shape[0]
    burn = total_iterations/4
    mcmc = mcmc[burn:,:]

    hill_x_range, hill_cdf_sum, pic50_x_range, pic50_cdf_sum = construct_posterior_predictive_cdfs(mcmc[:,0],mcmc[:,1],mcmc[:,2],mcmc[:,3])
    labels = ["Hill","pIC50"]
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(hill_x_range,hill_cdf_sum)
    ax1.set_xlim(hill_x_range[0],hill_x_range[-1])
    ax1.set_ylim(0,1)
    ax1.set_xlabel("Hill")
    ax1.set_ylabel("Cumulative distribution")
    ax2 = fig.add_subplot(122,sharey=ax1)
    ax2.plot(pic50_x_range,pic50_cdf_sum)
    ax2.set_xlim(pic50_x_range[0],pic50_x_range[-1])
    ax2.set_xlabel("pIC50")
    plt.setp(ax2.get_yticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig(figs_dir+"{}_{}_posterior_predictive_cdfs.png".format(drug,channel))
    plt.close()

    hill_cdf_file, pic50_cdf_file = dr.hierarchical_posterior_predictive_cdf_files(drug,channel)

    np.savetxt(hill_cdf_file,np.vstack((hill_x_range, hill_cdf_sum)).T)
    np.savetxt(pic50_cdf_file,np.vstack((pic50_x_range, pic50_cdf_sum)).T)



    hill_uniform_samples = npr.rand(args.samples)
    pic50_uniform_samples = npr.rand(args.samples)

    hill_interpolated_inverse_cdf_samples = np.interp(hill_uniform_samples,hill_cdf_sum,hill_x_range)
    pic50_interpolated_inverse_cdf_samples = np.interp(pic50_uniform_samples,pic50_cdf_sum,pic50_x_range)

    samples_file = dr.hierarchical_hill_and_pic50_samples_for_AP_file(drug,channel)
    with open(samples_file,'a') as outfile:
        np.savetxt(outfile,np.vstack((hill_interpolated_inverse_cdf_samples,pic50_interpolated_inverse_cdf_samples)).T)


    print "\nDone!\n"
    
for drug,channel in it.product(drugs_to_run,channels_to_run):
    run(drug,channel)





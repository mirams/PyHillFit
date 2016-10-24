import doseresponse as dr
import numpy as np
import scipy.stats as st

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
parser.add_argument("--num-cores", type=int, help="number of cores to parallelise drug/channel combinations",default=1)
parser.add_argument("-np", "--no-plots", action='store_true', help="don't make any plots, just save posterior predictive samples", default=False)
parser.add_argument("-tu", "--top-up", action='store_true', help="to use with --all, run on all drugs who don't already have MCMC files", default=False)
parser.add_argument("-sy", "--synthetic", action='store_true', help="use synthetic data (only one drug/channel combination exists currently", default=False)
parser.add_argument("-Ne", "--num_expts", type=int, help="how many experiments to fit to", default=0)
parser.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv")

args = parser.parse_args()

dr.setup(args.data_file)

drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

def construct_posterior_predictive_cdfs(alphas,betas,mus,ss):
    num_x_pts = 501
    hill_min = 0.
    hill_max = 4.
    pic50_min = -2.
    pic50_max = 12.
    hill_x_range = np.linspace(hill_min,hill_max,num_x_pts)
    pic50_x_range = np.linspace(pic50_min,pic50_max,num_x_pts)
    num_iterations = len(alphas) # assuming burn already discarded
    hill_pdf_sum = np.zeros(num_x_pts)
    hill_cdf_sum = np.zeros(num_x_pts)
    pic50_pdf_sum = np.zeros(num_x_pts)
    pic50_cdf_sum = np.zeros(num_x_pts)
    fisk = st.fisk.cdf
    fisk_pdf = st.fisk.pdf
    logistic = st.logistic.cdf
    logistic_pdf = st.logistic.pdf
    for i in xrange(num_iterations):
        hill_cdf_sum += fisk(hill_x_range,c=betas[i],scale=alphas[i],loc=0)
        hill_pdf_sum += fisk_pdf(hill_x_range,c=betas[i],scale=alphas[i],loc=0)
        pic50_cdf_sum += logistic(pic50_x_range,mus[i],ss[i])
        pic50_pdf_sum += logistic_pdf(pic50_x_range,mus[i],ss[i])
    hill_cdf_sum /= num_iterations
    pic50_cdf_sum /= num_iterations
    hill_pdf_sum /= num_iterations
    pic50_pdf_sum /= num_iterations
    return hill_x_range, hill_cdf_sum, pic50_x_range, pic50_cdf_sum, hill_pdf_sum, pic50_pdf_sum

def run(drug_channel):

    drug, channel = drug_channel
    
    print "\n\n{} + {}\n\n".format(drug,channel)
    
    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    if (0 < args.num_expts < num_expts):
        num_expts = args.num_expts
        save_samples_for_APs = False
    else:
        print "Doing all experiments\n"
        save_samples_for_APs = True
    
    
    drug, channel, output_dir, chain_dir, figs_dir, chain_file = dr.hierarchical_output_dirs_and_chain_file(drug,channel,args.synthetic,num_expts)
    

    try:
        mcmc = np.loadtxt(chain_file,usecols=range(4))
    except IOError:
        print "tried loading", chain_file
        print "No MCMC file found for {} + {}\n".format(drug,channel)
        return None
    total_iterations = mcmc.shape[0]
    burn = total_iterations/4
    mcmc = mcmc[burn:,:]
    
    

    hill_x_range, hill_cdf_sum, pic50_x_range, pic50_cdf_sum, hill_pdf_sum, pic50_pdf_sum = construct_posterior_predictive_cdfs(mcmc[:,0],mcmc[:,1],mcmc[:,2],mcmc[:,3])
    
    if (not args.no_plots):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt    
        labels = ["Hill","pIC50"]
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(121)
        ax1.plot(hill_x_range,hill_cdf_sum)
        ax1.set_xlim(hill_x_range[0],hill_x_range[-1])
        ax1.set_ylim(0,1)
        ax1.set_xlabel("Hill")
        ax1.set_ylabel("Cumulative distribution")
        ax1.grid()
        ax2 = fig.add_subplot(122,sharey=ax1)
        ax2.plot(pic50_x_range,pic50_cdf_sum)
        ax2.set_xlim(pic50_x_range[0],pic50_x_range[-1])
        ax2.set_xlabel("pIC50")
        ax2.grid()
        plt.setp(ax2.get_yticklabels(), visible=False)
        fig.tight_layout()
        fig.savefig(figs_dir+"{}_{}_posterior_predictive_cdfs.png".format(drug,channel))
        plt.close()
        xs = [hill_x_range,pic50_x_range]
        ys = [hill_pdf_sum,pic50_pdf_sum]
        labels = ['$Hill$','$pIC50$']
        file_labels = ['hill','pic50']
        for i in xrange(2):
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            ax.plot(xs[i],ys[i],color='blue')
            ax.grid()
            ax.set_xlabel(labels[i])
            ax.set_ylabel('Probability density')
            ax.set_title('{} posterior predictive'.format(labels[i][1:-1]))
            fig.tight_layout()
            fig.savefig(figs_dir+"{}_{}_{}_posterior_predictive.png".format(drug,channel,file_labels[i]))
            plt.close()

    hill_cdf_file, pic50_cdf_file = dr.hierarchical_posterior_predictive_cdf_files(drug,channel,args.synthetic,num_expts)

    np.savetxt(hill_cdf_file,np.vstack((hill_x_range, hill_cdf_sum)).T)
    np.savetxt(pic50_cdf_file,np.vstack((pic50_x_range, pic50_cdf_sum)).T)


    hill_uniform_samples = npr.rand(args.samples)
    pic50_uniform_samples = npr.rand(args.samples)

    hill_interpolated_inverse_cdf_samples = np.interp(hill_uniform_samples,hill_cdf_sum,hill_x_range)
    pic50_interpolated_inverse_cdf_samples = np.interp(pic50_uniform_samples,pic50_cdf_sum,pic50_x_range)

    # save a number of MCMC samples for use in AP models
    # we currently have it set to 500
    # in theory, the more samples, the better the AP histograms will look!
    if save_samples_for_APs:
        samples_file = dr.hierarchical_hill_and_pic50_samples_for_AP_file(drug,channel,args.synthetic)
        with open(samples_file,'w') as outfile:
            outfile.write('# {} samples of (alpha,mu) from MCMC output, to be used as (Hill,pIC50) in AP predictions\n'.format(args.samples))
            np.savetxt(outfile,np.vstack((hill_interpolated_inverse_cdf_samples,pic50_interpolated_inverse_cdf_samples)).T)


    print "\n{} + {} done!\n".format(drug,channel)
    return None
    
drugs_channels = it.product(drugs_to_run,channels_to_run)
if (args.num_cores<=1) or (len(drugs_to_run)==1):
    for drug_channel in drugs_channels:
        #run(drug_channel)
        
        # try/except is good when running multiple MCMCs and leaving them overnight,say
        # if one or more crash then the others will survive!
        # however, if you need more "control", comment out the try/except, and uncomment the other run(drug_channel) line
        try:
            run(drug_channel)
        except Exception,e:
            print e
            print "Failed to run {} + {}!".format(drug_channel[0],drug_channel[1])
# run multiple MCMCs in parallel
elif (args.num_cores>1):
    import multiprocessing as mp
    num_cores = min(args.num_cores, mp.cpu_count()-1)
    pool = mp.Pool(processes=num_cores)
    pool.map_async(run,drugs_channels).get(9999999)
    pool.close()
    pool.join()




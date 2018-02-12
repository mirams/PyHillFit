# this requires both the hierarchical and nonhierarchical MCMCs to have been run
# also requires the posterior predictive CDFs to have been computed in the hierarchical case

import doseresponse as dr
import matplotlib
#matplotlib.rc('font', family='ubuntu')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import itertools as it
import numpy as np
import sys
import numpy.random as npr
import scipy.stats as st

seed = 1
npr.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--concs", nargs="+", type=float, help="drug concentrations at which to find predicted responses",default=[])
parser.add_argument("-a", "--all", action='store_true', help='plot histograms from hierarchical MCMC on all drugs and channels', default=False)
parser.add_argument("-nc", "--num-cores", type=int, help="number of cores to parallelise drug/channel combinations", default=1)
parser.add_argument("-Ne", "--num-expts", type=int, help="how many experiments to fit to, otherwise will fit to all in the data file", default=0)
parser.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv")
parser.add_argument("--fix-hill", action='store_true', help="fix Hill=1 through fitting and MCMC",default=False)
parser.add_argument("-T", "--num-samples", type=int, help="number of samples to plot", required=True)
parser.add_argument("-H", "--num-hist-samples", type=int, help="number of samples for histogram --- should be much higher, really", required=True)
args = parser.parse_args()

dr.setup(args.data_file)

drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

    
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99']

def run(drug_channel):
    drug,channel = drug_channel
    print "\n\n{} + {}\n\n".format(drug,channel)
    
    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    if (0 < args.num_expts < num_expts):
        num_expts = args.num_expts
        
    drug, channel, output_dir, chain_dir, figs_dir, chain_file = dr.hierarchical_output_dirs_and_chain_file(drug,channel,num_expts)
    
    chain = np.loadtxt(chain_file)
    end = chain.shape[0]
    burn = end/4
    
    pic50_samples = np.zeros(args.num_hist_samples)
    hill_samples = np.zeros(args.num_hist_samples)
    rand_idx = npr.randint(burn, end, args.num_hist_samples)    
    for t in xrange(args.num_hist_samples):
        alpha, beta, mu, s = chain[rand_idx[t], :4]
        hill_samples[t] = st.fisk.rvs(c=beta, scale=alpha, loc=0)
        pic50_samples[t] = st.logistic.rvs(mu, s)
    
    
    num_pts = 40
    fig = plt.figure(figsize=(11,7))
    
    
    ax1 = fig.add_subplot(231)
    ax1.grid()
    xmin = -4
    xmax = 3
    concs = np.logspace(xmin,xmax,num_pts)
    ax1.set_xscale('log')
    ax1.set_ylim(0,100)
    ax1.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    ax1.set_ylabel(r'% {} block'.format(channel))
    ax1.set_title('A. Hierarchical predicted\nfuture experiments')
    ax1.set_xlim(10**xmin,10**xmax)
    
    for expt in experiment_numbers:
        ax1.scatter(experiments[expt][:,0],experiments[expt][:,1],label='Expt {}'.format(expt+1),color=colors[expt],s=100,zorder=10)
    
    for i, conc in enumerate(args.concs):
        ax1.axvline(conc,color=colors[3+i],lw=2,label=r"{} $\mu$M".format(conc),alpha=0.8)
        
    subset_idx = npr.randint(0, args.num_hist_samples, args.num_samples)
    for i in xrange(args.num_samples):  # only plot the first T of the H samples (should be fine because they're all randomly selected)
        ax1.plot(concs,dr.dose_response_model(concs, hill_samples[i], dr.pic50_to_ic50(pic50_samples[i])),color='black', alpha=0.01)
    
    lfs = 8
    
    ax1.legend(loc=2,fontsize=lfs)
    
    
    ax2 = fig.add_subplot(234)
    ax2.set_xlim(0,100)
    ax2.set_xlabel(r'% {} block'.format(channel))
    ax2.set_ylabel(r'Probability density')
    ax2.grid()
    for i, conc in enumerate(args.concs):
        ax2.hist(dr.dose_response_model(conc,hill_samples,dr.pic50_to_ic50(pic50_samples)),bins=50,normed=True,color=colors[3+i],alpha=0.8,edgecolor='none',label=r"{} $\mu$M {}".format(conc,drug))
    
    ax2.set_title('D. Hierarchical predicted\nfuture experiments')
    ax2.legend(loc="best",fontsize=lfs)
    
    ax3 = fig.add_subplot(232, sharey=ax1, sharex=ax1)
    ax3.grid()
    ax3.set_xscale('log')
    ax3.set_ylim(0,100)
    ax3.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    ax3.set_title('B. Hierarchical inferred\nunderlying effects')
    ax3.set_xlim(10**xmin,10**xmax)
    
    for expt in experiment_numbers:
        ax3.scatter(experiments[expt][:,0],experiments[expt][:,1],label='Expt {}'.format(expt+1),color=colors[expt],s=100,zorder=10)
    

    
    alpha_indices = npr.randint(burn, end, args.num_samples)
    alpha_samples = chain[alpha_indices, 0]
    mu_samples = chain[alpha_indices, 2]
    for i, conc in enumerate(args.concs):
        ax3.axvline(conc, color=colors[3+i], lw=2, label=r"{} $\mu$M".format(conc), alpha=0.8)
    for i in xrange(args.num_samples):
        ax3.plot(concs,dr.dose_response_model(concs,alpha_samples[i],dr.pic50_to_ic50(mu_samples[i])),color='black',alpha=0.01)
    ax3.legend(loc=2,fontsize=lfs)
    ax4 = fig.add_subplot(235, sharey=ax2, sharex=ax2)
    ax4.set_xlim(0,100)
    ax4.set_xlabel(r'% {} block'.format(channel))
    ax4.grid()
    
    hist_indices = npr.randint(burn,end,args.num_hist_samples)
    alphas = chain[hist_indices,0]
    mus = chain[hist_indices,2]
    
    for i, conc in enumerate(args.concs):
        ax4.hist(dr.dose_response_model(conc,alphas,dr.pic50_to_ic50(mus)),bins=50,normed=True,color=colors[3+i],alpha=0.8,edgecolor='none',label=r"{} $\mu$M {}".format(conc,drug))
    ax4.set_title('E. Hierarchical inferred\nunderlying effects')
    
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    
    
    # now plot non-hierarchical
    
    num_params = 3
    temperature = 1
    if args.fix_hill:
        model = 1
    else:
        model = 2
    drug,channel,chain_file,figs_dir = dr.nonhierarchical_chain_file_and_figs_dir(model, drug, channel, temperature)
    chain = np.loadtxt(chain_file,usecols=range(num_params-1)) # not interested in log-target values right now
    end = chain.shape[0]
    burn = end/4

    sample_indices = npr.randint(burn, end, args.num_samples)
    samples = chain[sample_indices, :]
    
    
    ax5 = fig.add_subplot(233, sharey=ax1, sharex=ax1)
    ax5.grid()
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.set_xscale('log')
    ax5.set_ylim(0,100)
    ax5.set_xlim(10**xmin,10**xmax)
    ax5.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    ax5.set_title('C. Single-level inferred\neffects')
    ax4.legend(loc="best", fontsize=lfs)
    
    for expt in experiment_numbers:
        if expt==1:
            ax5.scatter(experiments[expt][:,0],experiments[expt][:,1],color='orange',s=100,label='All expts',zorder=10)
        else:
            ax5.scatter(experiments[expt][:,0],experiments[expt][:,1],color='orange',s=100,zorder=10)
    
    for i, conc in enumerate(args.concs):
        ax5.axvline(conc,color=colors[3+i],alpha=0.8,lw=2,label=r"{} $\mu$M".format(conc))
    for i in xrange(args.num_samples):
        pic50, hill = samples[i, :]
        ax5.plot(concs,dr.dose_response_model(concs, hill, dr.pic50_to_ic50(pic50)),color='black',alpha=0.01)
    ax5.legend(loc=2,fontsize=lfs)
    
    sample_indices = npr.randint(burn,end, args.num_hist_samples)
    samples = chain[sample_indices,:]
    ax6 = fig.add_subplot(236, sharey=ax2, sharex=ax2)
    ax6.set_xlim(0,100)
    ax6.set_xlabel(r'% {} block'.format(channel))
    plt.setp(ax6.get_yticklabels(), visible=False)
    ax6.grid()
    for i, conc in enumerate(args.concs):
        ax6.hist(dr.dose_response_model(conc, samples[:,1], dr.pic50_to_ic50(samples[:,0])),bins=50,normed=True,alpha=0.8,color=colors[3+i],edgecolor='none',label=r"{} $\mu$M {}".format(conc,drug))
    ax6.set_title('F. Single-level inferred\neffects')

    ax6.legend(loc="best",fontsize=lfs)
    
    
    
    plot_dir = dr.all_predictions_dir(drug,channel)
    
    fig.tight_layout()
    png_file = plot_dir+'{}_{}_all_predictions_corrected.png'.format(drug,channel)
    print png_file
    fig.savefig(png_file)
    pdf_file = plot_dir+'{}_{}_all_predictions_corrected.pdf'.format(drug,channel)
    print pdf_file
    fig.savefig(pdf_file) # uncomment to save as pdf, or change extension to whatever you want
    

    plt.close()
    
    print "Figures saved in", plot_dir
    

drugs_channels = it.product(drugs_to_run,channels_to_run)
if (args.num_cores<=1) or (len(drugs_to_run)==1):
    for drug_channel in drugs_channels:
        #run(drug_channel)
        
        # try/except is good when running multiple plots and leaving them overnight,say
        # if one or more crash then the others will survive!
        # however, if you need more "control", comment out the try/except, and uncomment the other run(drug_channel) line
        try:
            run(drug_channel)
        except Exception,e:
            print e
            print "Failed to run {} + {}!".format(drug_channel[0],drug_channel[1])
# run multiple plots in parallel
elif (args.num_cores>1):
    import multiprocessing as mp
    num_cores = min(args.num_cores, mp.cpu_count()-1)
    pool = mp.Pool(processes=num_cores)
    pool.map_async(run,drugs_channels).get(9999999)
    pool.close()
    pool.join()

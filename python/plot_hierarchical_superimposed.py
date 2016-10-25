import doseresponse as dr
import matplotlib
#matplotlib.rc('font', family='ubuntu')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import itertools as it
import numpy as np
import numpy.random as npr
import scipy.stats as st
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", action='store_true', help='plot histograms from hierarchical MCMC on all drugs and channels', default=False)
parser.add_argument("-Ne", "--num_expts", type=int, help="how many experiments to fit to, otherwise will fit to all experiments in data file",default=0)
parser.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv")
args = parser.parse_args()

dr.setup(args.data_file)

drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

def run(drug,channel):
    print "\n\n{} + {}\n\n".format(drug,channel)
    
    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    if (0 < (args.num_expts) < num_expts):
        num_expts = args.num_expts
        experiment_numbers = [x for x in experiment_numbers[:num_expts]]
        experiments = [x for x in experiments[:num_expts]]
    drug, channel, output_dir, chain_dir, figs_dir, chain_file = dr.hierarchical_output_dirs_and_chain_file(drug,channel,num_expts)
    chain = np.loadtxt(chain_file)
    end, num_params = chain.shape
    burn = end/4
          
    
    top_params = ['alpha','beta','mu','s','sigma']
    top_param_indices = [0,1,2,3,num_params-2]
    mid_param_indices = [i for i in range(num_params-1) if i not in top_param_indices]
    num_expts = len(mid_param_indices)/2
    if num_expts <= 4: # qualitative and colourblind safe, apparently
        colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
        colors = ['#d7191c','#fdae61','#2c7bb6']
    else:
        colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    top_param_labels = [r'$\alpha$',r'$\beta$',r'$\mu$',r'$s$',r'$\sigma$']
    
        
    
    num_curves = 50
    indices = npr.randint(burn,end,num_curves)
    samples = chain[indices,:]
    
    
    all_fig = plt.figure(figsize=(4,8))
    curves = all_fig.add_subplot(311)
    curves.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    curves.set_ylabel(r'% {} block'.format(channel))
    curves.grid()
    curves.set_xscale('log')
    x_range = np.logspace(-4,2,201)
    for j in xrange(num_curves):
        for i in xrange(3):
            response = dr.dose_response_model(x_range,samples[j,4+2*i],dr.pic50_to_ic50(samples[j,4+2*i+1]))
            curves.plot(x_range,response,color=colors[i],alpha=0.2)
    
    
    for i,expt in enumerate(experiments):
        curves.plot(expt[:,0],expt[:,1],'o',color=colors[i],zorder=10,ms=10)
    curves.set_xlim(10**-4,10**2)
    
    pic50s = all_fig.add_subplot(312)
    pic50s.grid()
    pic50s.set_ylabel('Probability density')
    pic50s.set_xlabel(r'$pIC50_i$')
    hills = all_fig.add_subplot(313)
    hills.grid()
    hills.set_ylabel('Probability density')
    hills.set_xlabel(r'$Hill_i$')
        
    
    alpha = 1./(num_expts-1)
    for i, col in enumerate(mid_param_indices):
        color = matplotlib.colors.ColorConverter().to_rgba(colors[i/2],alpha=alpha)
        if i%2==0:
            label = r'$Hill_{}$'.format(i/2+1)
            file_label = 'Hill_{}'.format(i/2+1)
            hills.hist(chain[burn:,col],normed=True,bins=40,color=color,edgecolor='none',label=r'$i = {}$'.format(i/2+1))
        else:
            label = r'$pIC50_{}$'.format(i/2+1)
            file_label = 'pIC50_{}'.format(i/2+1)
            pic50s.hist(chain[burn:,col],normed=True,bins=40,color=color,edgecolor='none',label=r'$i = {}$'.format(i/2+1))
    hills.legend(loc=1,fontsize=10)
    all_fig.tight_layout()
    all_fig.savefig(figs_dir+'{}_{}_hierarchical_curves_and_hists.png'.format(drug,channel))
    #all_fig.savefig(figs_dir+'{}_{}_hierarchical_curves_and_hists.pdf'.format(drug,channel))
    plt.show(block=True)
    
    print "Figures saved in", figs_dir
    
    print "\n\n{} + {} done\n\n".format(drug,channel)
        
for drug,channel in it.product(drugs_to_run,channels_to_run):
    run(drug,channel)

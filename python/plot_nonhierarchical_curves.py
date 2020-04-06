import doseresponse as dr
import numpy as np
import numpy.random as npr
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
import itertools as it

#test

seed = 5
npr.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--all", action='store_true', help='plot graphs for all drugs and channels', default=False)
parser.add_argument("-c", "--num-cores", type=int, help="number of cores to parallelise drug/channel combinations",default=1)

requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv", required=True)

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

temperature = 1

# load data from specified data file
dr.setup(args.data_file)

# list drug and channel options, select from command line
# can select more than one of either
drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

num_curves = 800
num_pts = 201


def do_plots(drug_channel):
    top_drug, top_channel = drug_channel

    num_expts, experiment_numbers, experiments = dr.load_crumb_data(top_drug, top_channel)
    
    concs = np.array([])
    responses = np.array([])
    for i in xrange(num_expts):
        concs = np.concatenate((concs,experiments[i][:,0]))
        responses = np.concatenate((responses,experiments[i][:,1]))
    
    xmin = 1000
    xmax = -1000
    for expt in experiments:
        a = np.min(expt[:,0])
        b = np.max(expt[:,0])
        if a > 0 and a < xmin:
            xmin = a
        if b > xmax:
            xmax = b
     
    xmin = int(np.log10(xmin))-2
    xmax = int(np.log10(xmax))+3

    x = np.logspace(xmin,xmax,num_pts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4), sharey=True, sharex=True)
    ax1.set_xscale('log')
    ax1.grid()
    ax2.grid()
    ax1.set_xlim(10**xmin,10**xmax)
    ax1.set_ylim(0,100)
    ax1.set_xlabel(r'{} concentration ($\mu$M)'.format(top_drug))
    ax2.set_xlabel(r'{} concentration ($\mu$M)'.format(top_drug))
    ax1.set_ylabel(r'% {} block'.format(top_channel))
    
    model = 1
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(model, top_drug, top_channel, temperature)
    
    chain = np.loadtxt(chain_file)
    best_idx = np.argmax(chain[:,-1])
    best_pic50, best_sigma = chain[best_idx, [0,1]]
    
    saved_its, h = chain.shape
    rand_idx = npr.randint(saved_its, size=num_curves)

    pic50s = chain[rand_idx, 0]
    ax1.set_title(r'Model 1: $pIC50 = {}$, fixed $Hill = 1$'.format(np.round(best_pic50,2)))
    for i in xrange(num_curves):
        ax1.plot(x, dr.dose_response_model(x, 1., dr.pic50_to_ic50(pic50s[i])), color='black', alpha=0.02)
    max_pd_curve = dr.dose_response_model(x, 1., dr.pic50_to_ic50(best_pic50))
    ax1.plot(x, max_pd_curve, label='Max PD', lw=1.5, color='red')
    ax1.plot(concs,responses,"o",color='orange',ms=10,label='Data',zorder=10)
    
    anyArtist = plt.Line2D((0,1),(0,0), color='k')
    
    handles, labels = ax1.get_legend_handles_labels()
    
    if drug=="Quinine" and channel=="Nav1.5-late":
        loc = 4
    else:
        loc = 2
    ax1.legend(handles+[anyArtist], labels+["Samples"], loc=loc)
    
    model = 2
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(model, top_drug, top_channel, temperature)
    
    chain = np.loadtxt(chain_file)
    best_idx = np.argmax(chain[:,-1])
    best_pic50, best_hill, best_sigma = chain[best_idx, [0,1,2]]
    
    ax2.set_title(r"Model 2: $pIC50={}$, Hill = {}".format(np.round(best_pic50,2),np.round(best_hill,2)))
    
    saved_its, h = chain.shape
    rand_idx = npr.randint(saved_its, size=num_curves)

    pic50s = chain[rand_idx, 0]
    hills = chain[rand_idx, 1]
    for i in xrange(num_curves):
        ax2.plot(x, dr.dose_response_model(x, hills[i], dr.pic50_to_ic50(pic50s[i])), color='black', alpha=0.02)
    max_pd_curve = dr.dose_response_model(x, best_hill, dr.pic50_to_ic50(best_pic50))
    ax2.plot(x, max_pd_curve, label='Max PD', lw=1.5, color='red')
    ax2.plot(concs,responses,"o",color='orange',ms=10,label='Data',zorder=10)
    
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles+[anyArtist], labels+["Samples"], loc=loc)


    fig.tight_layout()
    #plt.show(block=True)
    #sys.exit()
    
    fig.savefig("{}_{}_nonh_both_models_mcmc_prediction_curves.png".format(drug, channel))
    plt.close()
    
    return None

for drug_channel in it.product(drugs_to_run, channels_to_run):
    print drug_channel
    do_plots(drug_channel)



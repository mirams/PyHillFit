import doseresponse as dr
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv", required=True)
args = parser.parse_args()

temperature = 1.0

# load data from specified data file
dr.setup(args.data_file)

# list drug and channel options, select from command line
# can select more than one of either
run_all = False
drugs_to_run, channels_to_run = dr.list_drug_channel_options(run_all)


def do_plots(drug_channel):
    top_drug, top_channel = drug_channel

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3), sharey=True, sharex=True)
    ax1.grid()
    ax2.grid()
    fs = 16
    ax1.set_xlabel(r'$\sigma$', fontsize=fs)
    ax2.set_xlabel(r'$\sigma$', fontsize=fs)
    ax1.set_title("Model 1")
    ax1.set_title("Model 2")
    ax1.set_ylabel("Normalised frequency")
    
    model = 1
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(model, top_drug, top_channel, temperature)
    
    sigmas = np.loadtxt(chain_file, usecols=[1])
    ax1.hist(sigmas, bins=40, normed=True, color='blue', edgecolor='blue')
    
    model = 2
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(model, top_drug, top_channel, temperature)
    
    sigmas = np.loadtxt(chain_file, usecols=[2])
    ax2.hist(sigmas, bins=40, normed=True, color='blue', edgecolor='blue')

    fig.tight_layout()
    fig.savefig("{}_{}_nonh_both_models_sigma_hists.png".format(drug, channel))
    plt.show(block=True)
    plt.close()
    
    return None

do_plots(drugs_to_run+channels_to_run)



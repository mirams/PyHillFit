import os
import pandas as pd
import numpy as np

def setup(given_file):
    global file_name, dir_name, df, drugs, channels
    file_name = given_file
    dir_name = given_file.split('/')[-1][:-4]
    df = pd.read_csv(file_name, names=['Drug','Channel','Experiment','Concentration','Inhibition'])
    drugs = df.Drug.unique()
    channels = df.Channel.unique()

def file_len(fname,num_its):
    with open(fname,'r') as f:
        for i, l in enumerate(f):
            if i < num_its:
                pass
            else:
                return i
    return i # because first line is header

def list_drug_channel_options(args_all):
    if not args_all:
        print "\nDrugs:\n"
        for i in range(len(drugs)):
            print "{}. {}".format(i+1,drugs[i])
        drug_indices = [x-1 for x in map(int,raw_input("\nSelect drug numbers: ").split())]
        assert(0 <= len(drug_indices) <= len(drugs))
        drugs_to_run = [drugs[drug_index] for drug_index in drug_indices]
        print "\nChannels:\n"
        for i in range(len(channels)):
            print "{}. {}".format(i+1,channels[i])
        channel_indices = [x-1 for x in map(int,raw_input("\nSelect channel numbers: ").split())]
        assert(0 <= len(channel_indices) <= len(channels))
        channels_to_run = [channels[channel_index] for channel_index in channel_indices]
    else:
        drugs_to_run = drugs
        channels_to_run = channels
    return drugs_to_run, channels_to_run
    
def load_crumb_data(drug,channel):
    experiment_numbers = np.array(df[(df['Drug'] == drug) & (df['Channel'] == channel)].Experiment.unique())
    num_expts = max(experiment_numbers)
    experiments = []
    for expt in experiment_numbers:
        experiments.append(np.array(df[(df['Drug'] == drug) & (df['Channel'] == channel) & (df['Experiment'] == expt)][['Concentration','Inhibition']]))
    experiment_numbers -= 1
    return num_expts, experiment_numbers, experiments
    
    
def hierarchical_output_dirs_and_chain_file(drug,channel,synthetic=False,Ne=0):
    if ('/' in drug):
        drug = drug.replace('/','_')
    if ('/' in channel):
        channel = channel.replace('/','_')
    output_dir = 'output/{}/hierarchical/{}/{}/{}_expts/'.format(dir_name,drug,channel,Ne)
    chain_dir = output_dir+'chain/'
    figs_dir = output_dir+'figures/'
    for directory in [output_dir,chain_dir,figs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    chain_file = chain_dir+'{}_{}_{}_hierarchical_chain.txt'.format(dir_name,drug,channel)
    return drug, channel, output_dir, chain_dir, figs_dir, chain_file
    
def dose_response_model(dose,hill,IC50):
    return 100. * ( 1. - 1./(1.+(1.*dose/IC50)**hill) )
    
def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)
    
def ic50_to_pic50(ic50): # IC50 in uM
    return 6-np.log10(ic50)
    
def hierarchical_posterior_predictive_cdf_files(drug,channel,synthetic,Ne):
    cdf_dir = 'output/{}/hierarchical/{}/{}/{}_expts/cdfs/'.format(dir_name,drug,channel,Ne)
    if not os.path.exists(cdf_dir):
        os.makedirs(cdf_dir)
    hill_cdf_file = cdf_dir+'{}_{}_{}_posterior_predictive_hill_cdf.txt'.format(dir_name,drug,channel)
    pic50_cdf_file = cdf_dir+'{}_{}_{}_posterior_predictive_pic50_cdf.txt'.format(dir_name,drug,channel)
    return hill_cdf_file, pic50_cdf_file
    
def hierarchical_hill_and_pic50_samples_for_AP_file(drug,channel,synthetic):
    output_dir = 'output/{}/hierarchical/synthetic/posterior_predictive_hill_pic50_samples/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '{}_{}_{}_hill_pic50_samples.txt'.format(dir_name,drug,channel)
    return output_file
    
def hierarchical_downsampling_folder_and_file(drug,channel):
    output_dir = 'output/{}/hierarchical/downsampling/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '{}_{}_downsampled_alpha_beta_mu_s.txt'.format(drug,channel)
    return output_file

def nonhierarchical_chain_file_and_figs_dir(drug,channel,synthetic):
    if ('/' in drug):
        drug = drug.replace('/','_')
    if ('/' in channel):
        channel = channel.replace('/','_')
    output_dir = 'output/{}/nonhierarchical/{}/{}/'.format(dir_name,drug,channel)
    chain_dir = output_dir+'chain/'
    images_dir = output_dir+'figures/'
    dirs = [output_dir,chain_dir,images_dir]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    chain_file = chain_dir+'{}_{}_{}_nonhierarchical_chain.txt'.format(dir_name,drug,channel)
    return drug,channel,chain_file,images_dir
    
def alpha_mu_downsampling(drug,channel,synthetic):
    if (not synthetic):
        output_dir = '../chaste/samples/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '{}_{}_hill_pic50_samples.txt'.format(drug,channel)
    return output_file
    
def all_predictions_dir(drug,channel):
    main_dir = 'output/{}/all_prediction_curves/{}/{}/'.format(dir_name,drug,channel)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    return main_dir


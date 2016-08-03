import os
import pandas as pd
import numpy as np

file_name = 'python_input_data.csv'
df = pd.read_csv(file_name, names=['Drug','Channel','Experiment','Concentration','Inhibition'])
drugs = df.Drug.unique()
channels = df.Channel.unique()

def list_drug_channel_options(args_all):
    if not args_all:
        print "\nDrugs:\n"
        for i in range(len(drugs)):
            print "{}. {}".format(i+1,drugs[i])
        drug_index = int(raw_input("\nSelect drug number: "))-1
        assert(0 <= drug_index < len(drugs))
        drug = drugs[drug_index]
        print "\nChannels:\n"
        for i in range(len(channels)):
            print "{}. {}".format(i+1,channels[i])
        channel_index = int(raw_input("\nSelect channel number: "))-1
        assert(0 <= channel_index < len(channels))
        channel = channels[channel_index]
        drugs_to_run = [drug]
        channels_to_run = [channel]
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
    
    
def hierarchical_output_dirs_and_chain_file(drug,channel):
    if ('/' in drug):
        drug = drug.replace('/','_')
    if ('/' in channel):
        channel = channel.replace('/','_')
    output_dir = 'output/hierarchical/drugs/{}/{}/'.format(drug,channel)
    chain_dir = output_dir+'chain/'
    figs_dir = output_dir+'figures/'
    for directory in [output_dir,chain_dir,figs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    chain_file = chain_dir+'{}_{}_hierarchical_chain.txt'.format(drug,channel)
    return drug, channel, output_dir, chain_dir, figs_dir, chain_file
    
def dose_response_model(dose,hill,IC50):
    return 100. * ( 1. - 1./(1.+(1.*dose/IC50)**hill) )
    
def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)
    
def hierarchical_posterior_predictive_cdf_files(drug,channel):
    cdf_dir = 'output/hierarchical/drugs/{}/{}/cdfs/'.format(drug,channel)
    if not os.path.exists(cdf_dir):
        os.makedirs(cdf_dir)
    hill_cdf_file = cdf_dir+'{}_{}_posterior_predictive_hill_cdf.txt'.format(drug,channel)
    pic50_cdf_file = cdf_dir+'{}_{}_posterior_predictive_pic50_cdf.txt'.format(drug,channel)
    return hill_cdf_file, pic50_cdf_file
    
def hierarchical_hill_and_pic50_samples_for_AP_file(drug,channel):
    output_dir = 'output/hierarchical/hill_pic50_samples/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '{}_{}_hill_pic50_samples.txt'.format(drug,channel)
    with open(output_file,'w') as outfile:
        outfile.write('# Hill, pIC50\n')
    return output_file

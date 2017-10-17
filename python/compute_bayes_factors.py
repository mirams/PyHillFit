import doseresponse as dr
import numpy as np
from glob import glob
import itertools as it
import os

num_models = 2
model_pairs = it.combinations(range(1, num_models+1), r=2)
expectations = {}

data_file = "../data/modified_crumb_data.csv"
#run_all = True



dr.setup(data_file)
#drugs_to_run, channels_to_run = dr.list_drug_channel_options(run_all)

drug = 'Amitriptyline'
channel = 'Cav1.2'

#drug = 'Amiodarone'
#channel = 'hERG'

num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug, channel)

concs = np.array([])
responses = np.array([])
for i in xrange(num_expts):
    concs = np.concatenate((concs, experiments[i][:, 0]))
    responses = np.concatenate((responses, experiments[i][:, 1]))

where_r_0 = responses==0
where_r_100 = responses==100
where_r_other = (responses>0) & (responses<100)

pi_bit = dr.compute_pi_bit_of_log_likelihood(where_r_other)
num_pts = where_r_other.sum()
for m in xrange(1, num_models+1):
    dr.define_model(m)

    n = 40
    c = 3
    temps = (np.arange(n+1.)/n)**c


    num_temps = len(temps)
    log_p_ys = np.zeros(num_temps)
    for i in xrange(num_temps):
        print i+1, "/", num_temps
        drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(m, drug, channel, temps[i])
        chain = np.loadtxt(chain_file, usecols=range(dr.num_params))
        num_its = chain.shape[0]
        total = 0.
        start = 0
        for it in xrange(start,num_its):
            temperature = 1  # approximating full likelihood
            total += dr.log_data_likelihood(responses, where_r_0, where_r_100, where_r_other, concs, chain[it, :], temperature, pi_bit)
        log_p_ys[i] = total / (num_its-start)
    expectations[m] = dr.trapezium_rule(temps, log_p_ys)

for pair in model_pairs:
    i, j = pair
    Bij = np.exp(expectations[i]-expectations[j])
    with open("{}_{}_BF.txt".format(drug,channel), "w") as outfile:
        outfile.write("{} + {}\n".format(drug,channel))
        outfile.write("B_{}{} = {}\n".format(i, j, Bij))
        outfile.write("B_{}{} = {}\n".format(j, i, 1./Bij))

import doseresponse as dr
import numpy as np
from glob import glob
import itertools as it
import os
import argparse
import sys
import multiprocessing as mp


def compute_log_py_approxn(temp):
    print temp
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(m, top_drug, top_channel, temp)
    chain = np.loadtxt(chain_file, usecols=range(dr.num_params))
    num_its = chain.shape[0]
    total = 0.
    start = 0
    for it in xrange(start,num_its):
        temperature = 1  # approximating full likelihood
        temp_bit = dr.log_data_likelihood(responses, where_r_0, where_r_100, where_r_other, concs, chain[it, :], temperature, pi_bit)
        total += temp_bit
        if temp_bit == -np.inf:
            print chain[it, :]
    answer = total / (num_its-start)
    if answer == -np.inf:
        print "ANSWER IS -INF"
    return answer


parser = argparse.ArgumentParser()
parser.add_argument("-nc", "--num-cores", type=int, help="number of cores to parallelise drug/channel combinations", default=1)

requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("-d", "--drug", type=int, help="drug index", required=True)
requiredNamed.add_argument("-c", "--channel", type=int, help="channel index", required=True)
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv", required=True)

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

num_models = 2
model_pairs = it.combinations(range(1, num_models+1), r=2)
expectations = {}

dr.setup(args.data_file)

top_drug = dr.drugs[args.drug]
top_channel = dr.channels[args.channel]

num_expts, experiment_numbers, experiments = dr.load_crumb_data(top_drug, top_channel)

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

    temps = (np.arange(dr.n+1.)/dr.n)**dr.c
    num_temps = len(temps)
    
    if args.num_cores == 1:
        log_p_ys = np.zeros(num_temps)
        for i in xrange(num_temps):
            log_p_ys[i] = compute_log_py_approxn(temps[i])
    elif args.num_cores > 1:
        pool = mp.Pool(args.num_cores)
        log_p_ys = np.array(pool.map_async(compute_log_py_approxn, temps).get(9999))
        pool.close()
        pool.join()
    print log_p_ys
    expectations[m] = dr.trapezium_rule(temps, log_p_ys)
    print expectations
    
drug, channel, chain_file, images_dir = dr.nonhierarchical_chain_file_and_figs_dir(1, top_drug, top_channel, 1)
bf_dir = "BFs/"
if not os.path.exists(bf_dir):
    os.makedirs(bf_dir)
bf_file = bf_dir + "{}_{}_B12.txt".format(drug,channel)
for pair in model_pairs:
    i, j = pair
    #print expectations[i], expectations[j]
    Bij = np.exp(expectations[i]-expectations[j])
    #print Bij
    #with open("{}_{}_BF.txt".format(drug,channel), "w") as outfile:
    #    outfile.write("{} + {}\n".format(drug,channel))
    #    outfile.write("B_{}{} = {}\n".format(i, j, Bij))
    #    outfile.write("B_{}{} = {}\n".format(j, i, 1./Bij))
    np.savetxt(bf_file, [Bij])




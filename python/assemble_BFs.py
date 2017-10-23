import doseresponse as dr
import numpy as np
import itertools as it
import os
import argparse
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv", required=True)

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

dr.setup(args.data_file)

drugs_channels_idx = it.product(range(30), range(7))
BFs = np.zeros((30, 7))
best_params_m1 = {}
best_params_m2 = {}
best_params = [best_params_m1, best_params_m2]

best_m2_hills = []
all_BFs = []

for i, j in drugs_channels_idx:

    top_drug = dr.drugs[i]
    top_channel = dr.channels[j]
        
    drug, channel, chain_file, images_dir = dr.nonhierarchical_chain_file_and_figs_dir(1, top_drug, top_channel, 1)
    bf_dir = "BFs/"
    bf_file = bf_dir + "{}_{}_B12.txt".format(drug,channel)

    BFs[i, j] = np.loadtxt(bf_file)
    
    for m in range(1,3):
        drug, channel, chain_file, images_dir = dr.nonhierarchical_chain_file_and_figs_dir(m, top_drug, top_channel, 1)
        best_params_file = images_dir+"{}_{}_best_fit_params.txt".format(drug, channel)
        best_params[m-1][(i,j)] = np.loadtxt(best_params_file)
    
    if BFs[i, j] > 1:
        print "{} + {}: B12 = {}".format(drug, channel, BFs[i, j])
        print "M1 best fit: {}".format(best_params[0][(i,j)])
        print "M2 best fit: {}".format(best_params[1][(i,j)])
        
    all_BFs.append(BFs[i, j])
    best_m2_hills.append(best_params[1][(i,j)][1])
    
max_idx = np.unravel_index(np.argmax(BFs), (30,7))
min_idx = np.unravel_index(np.argmin(BFs), (30,7))

print "max B12: {}, {} + {}".format(BFs[max_idx], dr.drugs[max_idx[0]], dr.channels[max_idx[1]])
print "max B21: {}, {} + {}".format(1./BFs[min_idx], dr.drugs[min_idx[0]], dr.channels[min_idx[1]])

print "B21 > B12 in {} cases".format(np.sum(BFs<1))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.axhline(1, color='red')
ax.set_ylabel('$B_{21}$'
ax.set_xlabel('Best $M_2 Hill$'
ax.grid()
ax.scatter(best_m2_hills, all_BFs)
plt.show(block=True)


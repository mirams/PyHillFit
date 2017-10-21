import doseresponse as dr
import numpy as np
import itertools as it
import os
import argparse
import sys

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

for i, j in drugs_channels_idx:

    top_drug = dr.drugs[i]
    top_channel = dr.channels[j]
        
    drug, channel, chain_file, images_dir = dr.nonhierarchical_chain_file_and_figs_dir(1, top_drug, top_channel, 1)
    bf_dir = "BFs/"
    bf_file = bf_dir + "{}_{}_B12.txt".format(drug,channel)

    BFs[i, j] = np.loadtxt(bf_file)
    
max_idx = np.unravel_index(np.argmax(BFs), (30,7))
min_idx = np.unravel_index(np.argmin(BFs), (30,7))

print "max:", BFs[max_idx]
print "min:", BFs[min_idx]

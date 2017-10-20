import doseresponse as dr
import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv", required=True)
requiredNamed.add_argument("-m", "--model", type=int, help="For non-hierarchical (put anything for hierarchical):1. fix Hill=1; 2. vary Hill", required=True)
requiredNamed.add_argument("-d", "--drug", type=int, help="drug index", required=True)
requiredNamed.add_argument("-c", "--channel", type=int, help="channel index", required=True)

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

dr.define_model(args.model)
num_params = dr.num_params

dr.setup(args.data_file)

top_drug = dr.drugs[args.drug]
top_channel = dr.channels[args.channel]

temperatures = (np.arange(dr.n+1.)/dr.n)**dr.c

for t in temperatures:
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(args.model, top_drug, top_channel, t)
    if not os.path.isfile(chain_file):
        print "{}, {}, model {}, temp {} - NOT FOUND".format(top_drug, top_channel, args.model, t)
        break



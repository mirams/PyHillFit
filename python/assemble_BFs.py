import doseresponse as dr
import numpy as np
import itertools as it
import os
import argparse
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

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

m2_hill_sds = []

#best_posterior_m2_hills = []

abiguous_BFs = []

m2_chain_dir = "/home/rossj/arcus-b-py-output"
def m2_chain_file(drug, channel):
    return m2_chain_dir + "{}_{}_model_2_temp_1.0_chain_nonhierarchical.txt".format(drug, channel)

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
        
    model = 2
    temp = 1.0
    drug, channel, chain_file, images_dir = dr.nonhierarchical_chain_file_and_figs_dir(model, top_drug, top_channel, temp)
    
    chain = np.loadtxt(chain_file, usecols=[1,3])  # hill and log-target
    m2_hill_sds.append(np.std(chain[:,0]))
    best_post_idx = np.argmax(chain[:,1])
    best_m2_hills.append(chain[best_post_idx, 0])
    
    """if 0.9 < BFs[i, j] < 1.1:
        abiguous_BFs.append((top_drug, top_channel))
        #print "NEARLY 1"
        #print "{} + {}: B12 = {}".format(drug, channel, BFs[i, j])
        #print "M1 best fit: {}".format(best_params[0][(i,j)])
        #print "M2 best fit: {}".format(best_params[1][(i,j)])
    elif BFs[i, j] < 1e-2:
        print "LESS THAN 1e-2"
        print "{} + {}: B12 = {}".format(drug, channel, BFs[i, j])
        print "log-targets:"
        for k in range(-3,3):
            if k==0:
                print "Next should be best"
            print chain[best_post_idx+k, -1]
        print 0"""
        #print "M1 best fit: {}".format(best_params[0][(i,j)])
        #print "M2 best fit: {}".format(best_params[1][(i,j)])
        
    all_BFs.append(BFs[i, j])
    #best_m2_hills.append(best_params[1][(i,j)][1])
    
max_idx = np.unravel_index(np.argmax(BFs), (30,7))
min_idx = np.unravel_index(np.argmin(BFs), (30,7))

print "max B12: {}, {} + {}".format(BFs[max_idx], dr.drugs[max_idx[0]], dr.channels[max_idx[1]])
print "max B21: {}, {} + {}".format(1./BFs[min_idx], dr.drugs[min_idx[0]], dr.channels[min_idx[1]])

print "B21 > B12 in {} cases".format(np.sum(BFs<1))

print "\nmin best_m2_hill: {}".format(min(best_m2_hills))
print "\nmax best_m2_hill: {}".format(max(best_m2_hills))
where_max_hill = np.unravel_index(np.argmax(best_m2_hills), (30,7))
print "\nmax best_m2_hill from {} + {}".format(dr.drugs[where_max_hill[0]], dr.channels[where_max_hill[1]])

print "\nbest_m2_hills:"
print "["+",".join([str(i) for i in best_m2_hills])+"]"
print "\nall_BFs:"
print "["+",".join([str(i) for i in all_BFs])+"]"
print "\nm2_hill_sds:"
print "["+",".join([str(i) for i in m2_hill_sds])+"]"

sys.exit()

fig = plt.figure(figsize=(4,3))
#ax = fig.add_subplot(111, projection='3d')  # for newer matplotlib
ax = Axes3D(fig)  # for older matplotlib
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(10**-1, 10**2)
#ax.axhline(1, color='red', lw=2)
#ax.axvline(1, color='green', lw=2)
ax.set_ylabel('$B_{12}$')
ax.set_xlabel('Best $M_2 Hill$')
ax.set_zlabel('Hill s.d.')
ax.grid()
ax.scatter(best_m2_hills, all_BFs, m2_hill_sds, zorder=10)
#fig.tight_layout()
fig.savefig("B12_vs_best_cmaes_M2_Hill_vs_posterior_sd.png")
plt.show(block=True)

"""fig2 = plt.figure()#figsize=(4,3))
ax2 = fig2.add_subplot(111)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlim(10**-1, 10**2)
ax2.axhline(1, color='red', lw=2)
ax2.axvline(1, color='green', lw=2)
ax2.set_ylabel('$B_{12}$')
ax2.set_xlabel('Best $M_2 Hill$')
ax2.grid()
ax2.scatter(best_posterior_m2_hills, all_BFs, zorder=10)
fig2.tight_layout()
#fig2.savefig("B12_vs_best_post_density_M2_Hill.png")"""

"""for i, j in it.product(range(30), range(7)):
    idx = 7*i + j
    #print idx, all_BFs[idx]
    if all_BFs[idx] < 1e-2:
        txt = "{}\n{}".format(dr.drugs[i], dr.channels[j])
        print txt
        print "hill:", best_posterior_m2_hills[idx]
        print "B12:", all_BFs[idx]
        ax2.annotate(txt, (best_posterior_m2_hills[idx], all_BFs[idx]))"""

"""plt.close()
print "\nAmbiguous B12s:"
for q in abiguous_BFs:
    print q"""

"""model = 2
temp = 1.0
for d_c in abiguous_BFs:
    d, c = d_c
    drug, channel, chain_file, images_dir = dr.nonhierarchical_chain_file_and_figs_dir(model, d, c, temp)
    img = mpimg.imread(images_dir + "{}_{}_Hill_marginal.png".format(drug, channel))
    plt.imshow(img)
    plt.show(block=True)"""


import numpy as np
import numpy.random as npr
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import operator

seed = 1
npr.seed(seed)

results_files = glob("*.dat")

labels = []
all_samples = np.zeros((30, 500))

for i, rf in enumerate(results_files):
    drug = rf.split('_')[0]
    #if (drug=="Quinidine"):
        #continue
    labels.append(drug)
    samples = pd.read_table(rf)["APD90(ms)"].tolist()
    samples.sort()
    all_samples[i, :] = samples
    
num_perms = 10000

rows = np.arange(30)

all_ranks = {drug_name: [] for drug_name in labels}

for _ in xrange(num_perms):
    perm_idx = npr.randint(0, 500, 30)
    random_apds = all_samples[rows, perm_idx]

    d = {drug_name: apd for (drug_name, apd) in zip(labels, random_apds)}

    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
    for i, q in enumerate(sorted_d):
        all_ranks[q[0]].append(i+1)

for drg in all_ranks:
    rank_count = [all_ranks[drg].count(i) for i in range(1, 31)]
    #print rank_count
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.grid()
    #ax.hist(all_ranks[drg], np.arange(0.5, 31.5, 1))
    ax.bar(np.arange(30)+1, rank_count, align='center')
    ax.set_title(drg)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Rank")
    old_xticks = ax.get_xticks()
    ax.set_xticks(np.concatenate(([1],old_xticks[1:])))
    ax.set_xlim(0,31)
    fig.tight_layout()
plt.show()

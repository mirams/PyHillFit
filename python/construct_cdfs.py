import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy.random as npr

# get rid of for real version
import pandas as pd
import os

seed = 1
npr.seed(seed)

def construct_posterior_predictive_cdfs(alphas,betas,mus,ss):
    num_x_pts = 501
    hill_min = 0.
    hill_max = 2.*(np.ceil(np.max(hills))+1.)
    pic50_min = -2.
    pic50_max = 2.*(np.ceil(np.max(pic50s))+1.)
    hill_x_range = np.linspace(hill_min,hill_max,num_x_pts)
    pic50_x_range = np.linspace(pic50_min,pic50_max,num_x_pts)
    num_iterations = len(hills) # assuming burn already discarded
    hill_cdf_sum = np.zeros(num_x_pts)
    pic50_cdf_sum = np.zeros(num_x_pts)
    fisk = st.fisk.cdf
    logistic = st.logistic.cdf
    for i in xrange(num_iterations):
        hill_cdf_sum += fisk(hill_x_range,c=betas[i],scale=alphas[i],loc=0)
        pic50_cdf_sum += logistic(pic50_x_range,mus[i],ss[i])
    hill_cdf_sum /= num_iterations
    pic50_cdf_sum /= num_iterations
    return hill_x_range, hill_cdf_sum, pic50_x_range, pic50_cdf_sum
    
def output_paths(drug,channel):
    output_dir = 'output/hierarchical/drugs/{}/{}/'.format(drug,channel)
    chain_dir = output_dir+'chain/'
    figs_dir = output_dir+'figures/'
    dirs = [output_dir,chain_dir,figs_dir]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    chain_file = chain_dir+'{}_{}_hierarchical_chain.txt'.format(drug,channel)
    return chain_file,figs_dir
    
def output_cdf_dir(drug,channel):
    temp_dir = 'output/hierarchical/drugs/{}/{}/cdfs/'.format(drug,channel)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir
    
file_name = 'python_input_data.csv'
df = pd.read_csv(file_name, names=['Drug','Channel','Experiment','Concentration','Inhibition'])
drugs = df.Drug.unique()
channels = df.Channel.unique()

drug = drugs[0]
channel = channels[0]
    
chain_file,figs_dir = output_paths(drug,channel)
mcmc = np.loadtxt(chain_file,usecols=range(4))
total_iterations = mcmc.shape[0]
burn = total_iterations/4
mcmc = mcmc[burn:,:]


cdf_dir = output_cdf_dir(drug,channel)
hill_x_range, hill_cdf_sum, pic50_x_range, pic50_cdf_sum = construct_posterior_predictive_cdfs(mcmc[:,0],mcmc[:,1],mcmc[:,2],mcmc[:,3])
labels = ["Hill","pIC50"]
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.plot(hill_x_range,hill_cdf_sum)
ax1.set_xlim(hill_x_range[0],hill_x_range[-1])
ax1.set_ylim(0,1)
ax1.set_xlabel("Hill")
ax1.set_ylabel("Cumulative distribution")
ax2 = fig.add_subplot(122,sharey=ax1)
ax2.plot(pic50_x_range,pic50_cdf_sum)
ax2.set_xlim(pic50_x_range[0],pic50_x_range[-1])
ax2.set_xlabel("pIC50")
plt.setp(ax2.get_yticklabels(), visible=False)
fig.tight_layout()
fig.savefig(cdf_dir+"{}_{}_posterior_predictive_cdfs.png".format(drug,channel))
plt.close()

np.savetxt(cdf_dir+'{}_{}_posterior_predictive_Hill_cdf.txt'.format(drug,channel),np.vstack((hill_x_range, hill_cdf_sum)).T)
np.savetxt(cdf_dir+'{}_{}_posterior_predictive_pIC50_cdf.txt'.format(drug,channel),np.vstack((pic50_x_range, pic50_cdf_sum)).T)

























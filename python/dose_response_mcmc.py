import doseresponse as dr
import numpy as np
import numpy.random as npr

"""I have found that these two lines are needed on *some* computers to prevent matplotlib figure windows from opening.
In general, I save the figures but do not actually open the matplotlib figure windows.
Try uncommenting these two lines if annoying unwanted figure windows open."""
import matplotlib # actually need this first one for colour normalising later on
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time
import sys
import os
import itertools as it
import multiprocessing as mp

try:
    import cma
except:
    sys.exit("Can't find CMA-ES module (cma)")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, help="number of MCMC iterations",default=500000)
parser.add_argument("-t", "--thinning", type=int, help="how often to thin the MCMC, i.e. save every t-th iteration",default=5)
parser.add_argument("-b", "--burn-in-fraction", type=int, help="given N saved MCMC iterations, discard the first N/b as burn-in",default=4)
parser.add_argument("--best-fit-only", action="store_true", help="only find best-fit parameters using CMA-ES")
parser.add_argument('-a', '--all', action='store_true', help='run all drugs and all channels',default=False)
parser.add_argument('-s', '--num-samples', type=int, help='number of MCMC samples to save for AP simulations', default=500)
parser.add_argument("--num-cores", type=int, help="number of cores to parallelise drug/channel combinations",default=1)
parser.add_argument("-sy", "--synthetic", action='store_true', help="use synthetic data (only one drug/channel combination exists currently", default=False)

args = parser.parse_args()

dr.setup(args.synthetic)
drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

def log_likelihood(measurements,doses,theta):
    hill = theta[0]
    pIC50 = theta[1]
    sigma = theta[2]
    IC50 = dr.pic50_to_ic50(pIC50)
    return -len(measurements) * np.log(sigma) - np.sum((measurements-dr.dose_response_model(doses,hill,IC50))**2)/(2.*sigma**2)
    # as usual in our MCMC, omitted the -n/2*log(2pi) term from the log-likelihood, as this is always cancelled out
    
def sum_of_square_diffs(unscaled_params,doses,responses):
    hill = unscaled_params[0]**2
    pIC50 = unscaled_params[1]**2-1
    IC50 = dr.pic50_to_ic50(pIC50)
    test_responses = dr.dose_response_model(doses,hill,IC50)
    return np.sum((test_responses-responses)**2)

def initial_sigma(n,sum_of_squares):
    return np.sqrt(sum_of_squares/n)

def all_chains_file(drug,channel,num_params):
    all_chains_folder = 'output/nonhierarchical/hill_and_pic50_samples/'
    if not os.path.exists(all_chains_folder):
        os.makedirs(all_chains_folder)
    return all_chains_folder, all_chains_folder+'{}_{}_hill_pic50.txt'.format(drug,channel)

def run(drug_channel):

    drug, channel = drug_channel

    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(drug,channel,args.synthetic)
    
    num_params = 3 # hill, pic50, mu
    
    concs = np.array([])
    responses = np.array([])
    for i in xrange(num_expts):
        concs = np.concatenate((concs,experiments[i][:,0]))
        responses = np.concatenate((responses,experiments[i][:,1]))
        
    print experiments
    print concs
    print responses
    

    hill_prior = [0, 10]
    pic50_prior = [-1, 20]
    sigma_prior = [0, 50]

    prior_lowers = np.array([hill_prior[0], pic50_prior[0],sigma_prior[0]])
    prior_uppers = np.array([hill_prior[1], pic50_prior[1],sigma_prior[1]])
    
    
    # for reproducible results, otherwise select a new random seed
    seed = 1
    npr.seed(seed)



    start = time.time()
    x0 = np.array([1.,2.5]) # not fitting sigma by CMA-ES, can maximise log-likelihood wrt sigma analytically
    sigma0 = 0.1
    opts = cma.CMAOptions()
    opts['seed'] = seed
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X = es.ask()
        es.tell(X, [sum_of_square_diffs(x,concs,responses) for x in X])
        es.disp()
    res = es.result()

    hill_cur = res[0][0]**2
    pic50_cur = res[0][1]**2-1
    sigma_cur = initial_sigma(len(responses),res[1])
    proposal_scale = 0.01

    theta_cur = np.array([hill_cur,pic50_cur,sigma_cur])
    mean_estimate = np.copy(theta_cur)
    cov_estimate = proposal_scale*np.diag(np.copy(np.abs(theta_cur)))
        
    cmaes_ll = log_likelihood(responses,concs,theta_cur)
        
    best_fit_fig = plt.figure()
    best_fit_ax = best_fit_fig.add_subplot(111)
    best_fit_ax.set_xscale('log')
    plot_lower_lim = int(np.log10(np.min(concs)))-1
    plot_upper_lim = int(np.log10(np.max(concs)))+2
    best_fit_ax.set_xlim(10**plot_lower_lim,10**plot_upper_lim)
    best_fit_ax.set_ylim(0,100)
    num_pts = 1001
    x_range = np.logspace(plot_lower_lim,plot_upper_lim,num_pts)
    best_fit_curve = dr.dose_response_model(x_range,hill_cur,dr.pic50_to_ic50(pic50_cur))
    best_fit_ax.plot(x_range,best_fit_curve,label='Best fit')
    best_fit_ax.set_ylabel('% {} block'.format(channel))
    best_fit_ax.set_xlabel('{} concentration (uM)'.format(drug))
    best_fit_ax.set_title('Hill = {}, pIC50 = {}, log-likelihood = {}'.format(np.round(hill_cur,2),np.round(pic50_cur,2),np.round(cmaes_ll,2)))
    best_fit_ax.scatter(concs,responses,marker="o",color='orange',s=100,label='Data',zorder=10)
    best_fit_ax.legend(loc=2)
    best_fit_fig.tight_layout()
    best_fit_fig.savefig(images_dir+'{}_{}_CMA-ES_best_fit.png'.format(drug,channel))
    plt.close()

    if args.best_fit_only:
        sys.exit()


    

    # let MCMC look around for a bit before adaptive covariance matrix
    # same rule (100*dimension) as in hierarchical case
    when_to_adapt = 100*num_params

    log_target_cur = log_likelihood(responses,concs,theta_cur)
    print "initial log_target_cur =", log_target_cur



    # effectively step size, scales covariance matrix
    loga = 0.
    # what fraction of proposed samples are being accepted into the chain
    acceptance = 0.
    # what fraction of samples we WANT accepted into the chain
    # loga updates itself to try to make this dream come true
    target_acceptance = 0.25

    total_iterations = args.iterations
    thinning = args.thinning
    assert(total_iterations%thinning==0)
    
    # how often to print a little status message
    status_when = total_iterations / 20

    saved_iterations = total_iterations/thinning+1
    # also want to store log-target value at each iteration
    chain = np.zeros((saved_iterations,num_params+1))

    chain[0,:] = np.concatenate((np.copy(theta_cur),[log_target_cur]))
    print chain[0]

    

    print "concs:", concs
    print "responses:", responses

    # MCMC!
    t = 1
    start = time.time()
    while t <= total_iterations:
        theta_star = npr.multivariate_normal(theta_cur,np.exp(loga)*cov_estimate)
        accepted = 0
        if np.all(prior_lowers < theta_star) and np.all(theta_star < prior_uppers):
            log_target_star = log_likelihood(responses,concs,theta_star)
            accept_prob = npr.rand()
            if (np.log(accept_prob) < log_target_star - log_target_cur):
                theta_cur = theta_star
                log_target_cur = log_target_star
                accepted = 1
        acceptance = ((t-1.)*acceptance + accepted)/t
        if (t>when_to_adapt):
            s = t - when_to_adapt
            gamma_s = 1/(s+1)**0.6
            temp_covariance_bit = np.array([theta_cur-mean_estimate])
            cov_estimate = (1-gamma_s) * cov_estimate + gamma_s * np.dot(np.transpose(temp_covariance_bit),temp_covariance_bit)
            mean_estimate = (1-gamma_s) * mean_estimate + gamma_s * theta_cur
            loga += gamma_s*(accepted-target_acceptance)
        if (t%thinning==0):
            chain[t/thinning,:] = np.concatenate((np.copy(theta_cur),[log_target_cur]))
        if (t%status_when==0):
            print "{} / {}".format(t/status_when,total_iterations/status_when)
            time_taken_so_far = time.time()-start
            estimated_time_left = time_taken_so_far/t*(total_iterations-t)
            print "Time taken: {} s = {} min".format(np.round(time_taken_so_far,1),np.round(time_taken_so_far/60,2))
            print "acceptance = {}".format(np.round(acceptance,5))
            print "Estimated time remaining: {} s = {} min".format(np.round(estimated_time_left,1),np.round(estimated_time_left/60,2))
        t += 1

    print "\nTime taken to do {} MCMC iterations: {} s\n".format(total_iterations, time.time()-start)
    print "Final iteration:", chain[-1,:], "\n"
    
    with open(chain_file,'w') as outfile:
        outfile.write('# Nonhierarchical MCMC output for {} + {}: (Hill,pIC50,sigma,log-target)\n'.format(drug,channel))
        np.savetxt(outfile,chain)
    
    try:
        assert(len(chain[:,0])==saved_iterations)
    except AssertionError:
        print "len(chain[:,0])!=saved_iterations"
        sys.exit()
        
    burn_fraction = args.burn_in_fraction
    burn = saved_iterations/burn_fraction

    best_ll_index = np.argmax(chain[:,num_params])
    best_ll_row = chain[best_ll_index,:]
    print "Best log-likelihood:", "\n", best_ll_row

    figs = []
    axs = []
    # plot all marginal posterior distributions
    for i in range(num_params):
        labels = ['Hill','pIC50',r'$\sigma$']
        file_labels =  ['Hill','pIC50','sigma']
        figs.append(plt.figure())
        axs.append([])
        axs[i].append(figs[i].add_subplot(211))
        axs[i][0].hist(chain[burn:,i],bins=40,normed=True)
        axs[i][0].legend()
        axs[i][0].set_title("MCMC marginal distributions")
        axs[i].append(figs[i].add_subplot(212,sharex=axs[i][0]))
        axs[i][1].plot(chain[burn:,i],range(burn,saved_iterations))
        axs[i][1].invert_yaxis()
        axs[i][1].set_xlabel(labels[i])
        axs[i][1].set_ylabel('Saved MCMC iteration')
        figs[i].tight_layout()
        figs[i].savefig(images_dir+'{}_{}_{}_marginal.png'.format(drug,channel,file_labels[i]))
        plt.close()

    # plot log-target path
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.plot(range(saved_iterations), chain[:,-1])
    ax3.set_xlabel('MCMC iteration')
    ax3.set_ylabel('log-target')
    fig2.tight_layout()
    fig2.savefig(images_dir+'log_target.png')
    plt.close()

    # plot scatterplot matrix of posterior(s)
    labels = ['Hill','pIC50',r'$\sigma$']
    colormin, colormax = 1e9,0
    norm = matplotlib.colors.Normalize(vmin=5,vmax=10)
    hidden_labels = []
    count = 0
    # there's probably a better way to do this
    # I plot all the histograms to normalize the colours, in an attempt to give a better comparison between the pairwise plots
    while count < 2:
        axes = {}
        matrix_fig = plt.figure(figsize=(3*num_params,3*num_params))
        for i in range(num_params):
            for j in range(i+1):
                ij = str(i)+str(j)
                subplot_position = num_params*i+j+1
                if i==j:
                    axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position)
                    axes[ij].hist(chain[burn:,i],bins=50,normed=True,color='blue')
                elif j==0: # this column shares x-axis with top-left
                    axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes["00"])
                    counts, xedges, yedges, Image = axes[ij].hist2d(chain[burn:,j],chain[burn:,i],cmap='hot_r',bins=50,norm=norm)
                    maxcounts = np.amax(counts)
                    if maxcounts > colormax:
                        colormax = maxcounts
                    mincounts = np.amin(counts)
                    if mincounts < colormin:
                        colormin = mincounts
                else:
                    axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes[str(j)+str(j)],sharey=axes[str(i)+"0"])
                    counts, xedges, yedges, Image = axes[ij].hist2d(chain[burn:,j],chain[burn:,i],cmap='hot_r',bins=50,norm=norm)
                    maxcounts = np.amax(counts)
                    if maxcounts > colormax:
                        colormax = maxcounts
                    mincounts = np.amin(counts)
                    if mincounts < colormin:
                        colormin = mincounts
                if i!=num_params-1:
                    hidden_labels.append(axes[ij].get_xticklabels())
                if j!=0:
                    hidden_labels.append(axes[ij].get_yticklabels())
                if i==num_params-1:
                    axes[str(i)+str(j)].set_xlabel(labels[j])
                if j==0:
                    axes[str(i)+str(j)].set_ylabel(labels[i])
                plt.xticks(rotation=30)
        norm = matplotlib.colors.Normalize(vmin=colormin,vmax=colormax)
        count += 1

        
    plt.setp(hidden_labels, visible=False)
    
    matrix_fig.tight_layout()
    matrix_fig.savefig(images_dir+"{}_{}_scatterplot_matrix.png".format(drug,channel))
    #matrix_fig.savefig(images_dir+"{}_{}_scatterplot_matrix.pdf".format(drug,channel))
    plt.close()


    print "\n\n{} + {} complete!\n\n".format(drug,channel)

drugs_channels = it.product(drugs_to_run,channels_to_run)
if (args.num_cores<=1) or (len(drugs_to_run)==1):
    for drug_channel in drugs_channels:
        #run(drug_channel)
        
        # try/except is good when running multiple MCMCs and leaving them overnight,say
        # if one or more crash then the others will survive!
        # however, if you need more "control", comment out the try/except, and uncomment the other run(drug_channel) line
        try:
            run(drug_channel)
        except Exception,e:
            print e
            print "Failed to run {} + {}!".format(drug_channel[0],drug_channel[1])
# run multiple MCMCs in parallel
elif (args.num_cores>1):
    import multiprocessing as mp
    num_cores = min(args.num_cores, mp.cpu_count()-1)
    pool = mp.Pool(processes=num_cores)
    pool.map_async(run,drugs_channels).get(9999999)
    pool.close()
    pool.join()




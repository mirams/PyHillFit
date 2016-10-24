import doseresponse as dr
import numpy as np
import numpy.random as npr
import itertools as it
import time
import os
import argparse
import scipy.stats as st

"""I have found that these two lines are needed on *some* computers to prevent matplotlib figure windows from opening.
In general, I save the figures but do not actually open the matplotlib figure windows.
Try uncommenting these two lines if annoying unwanted figure windows open."""
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, help="number of MCMC iterations",default=500000)
parser.add_argument("-a", "--all", action='store_true', help='run hierarchical MCMC on all drugs and channels', default=False)
parser.add_argument('-ppp', '--plot-parameter-paths', action='store_true', help='plot the path taken by each parameter through the (thinned) MCMC',default=False)
parser.add_argument("-p", "--predictions", type=int, help="number of prediction curves to plot",default=1000)
parser.add_argument("-c", "--num-cores", type=int, help="number of cores to parallelise drug/channel combinations",default=1)
parser.add_argument("-sy", "--synthetic", action='store_true', help="use synthetic data (only one drug/channel combination exists currently", default=False)
parser.add_argument("-Ne", "--num_expts", type=int, help="how many experiments to fit to", default=0)
parser.add_argument("--num-APs", type=int, help="how many (alpha,mu) samples to take for AP simulations", default=500)
parser.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv")
args = parser.parse_args()

if not args.data_file:
    sys.exit('\nPlease provide path to input data file using --data-file. Exiting now.\n')

# load either real or synthetic data depending on command line argument, default is real data
dr.setup(args.data_file)

# list drug and channel options, select from command line
# can select more than one of either
drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

def log_likelihood(measurements,doses,theta):
    sigma = theta[2]
    IC50 = dr.pic50_to_ic50(pIC50)
    return -len(measurements) * np.log(sigma) - np.sum((measurements-dr.dose_response_model(doses,hill,IC50))**2)/(2.*sigma**2)
    # as usual in our MCMC, omitted the -n/2*log(2pi) term from the log-likelihood, as this is always cancelled out
    
# for finding starting point for MCMC, so if we later decide pIC50 can go down to -2, it doesn't matter, it will just take a few
# iterations to decide if it wants to go in that direction
def sum_of_square_diffs(unscaled_params,doses,responses):
    hill = unscaled_params[0]**2 # restricting Hill>0
    pIC50 = unscaled_params[1]**2-1 # restricting pIC50>-1
    IC50 = dr.pic50_to_ic50(pIC50)
    test_responses = dr.dose_response_model(doses,hill,IC50)
    return np.sum((test_responses-responses)**2)

# analytic solution for sigma to maximise likelihood from Normal distribution
def initial_sigma(n,sum_of_squares):
    return np.sqrt(sum_of_squares/n)

# for all parts of the log target distribution:
# -inf is ok, NaN is not!
# np.log(negative) = nan, so we need to catch negatives first and set the target to -inf
# this is ok because it's equivalent to the parameter being in a region of 0 likelihood
# therefore I've put loads of warning messages, and it will abort if it doesn't catch any problems
# if CMA-ES finds a good (legal) starting point for the MCMC, it shouldn't really get any NaNs

def log_data_likelihood(hill_is,pic50_is,sigma,experiments):
    Ne = len(experiments)
    answer = 0.
    for i in range(Ne):
        ic50 = dr.pic50_to_ic50(pic50_is[i])
        concs = experiments[i][:,0]
        num_expt_pts = len(concs)
        data = experiments[i][:,1]
        model_responses = dr.dose_response_model(concs,hill_is[i],ic50)
        exp_bit = np.sum((data-model_responses)**2)/(2*sigma**2)
        # assuming noise Normal is truncated at 0 and 100
        truncated_scale = np.sum(np.log(st.norm.cdf(100,model_responses,sigma)-st.norm.cdf(0,model_responses,sigma)))
        answer -= (num_expt_pts*np.log(sigma) + exp_bit + truncated_scale)
    if np.isnan(answer):
        print "NaN from log_data_likelihood!"
        print "hill_is =", hill_is
        print "pic50_is =", pic50_is
        print "sigma =", sigma
        sys.exit()
    return answer

def log_hill_i_log_logistic_likelihood(x,alpha,beta):
    answer = np.log(beta) - beta*np.log(alpha) + (beta-1.)*np.log(x) - 2*np.log(1+(x/alpha)**beta)
    if np.any(np.isnan(answer)):
        print "NaN from log_hill_i_log_logistic_likelihood!"
        print "x =", x
        print "alpha =", alpha
        print "beta =", beta
        sys.exit()
    return answer

def log_pic50_i_logistic_likelihood(x,mu,s):
    temp_bit = (x-mu)/s
    answer = -temp_bit - np.log(s) - 2*np.log(1+np.exp(-temp_bit))
    if np.any(np.isnan(answer)):
        print "NaN from log_pic50_i_logistic_likelihood!"
        print "x =", x
        print "mu =", mu
        print "s =", s
        sys.exit()
    else:
        return answer

def log_beta_prior(x,alpha,beta,a,b):
    if (x<a) or (x>b):
        return -np.inf
    else:
        answer = (alpha-1)*np.log(x-a) + (beta-1)*np.log(b-x)
        if np.isnan(answer):
            print "NaN from log_beta_prior!"
            print "x =", x
            print "alpha =", alpha
            print "beta =", beta
            print "a =", a
            print "b =", b
            sys.exit()
        else:
            return answer

def log_gamma_prior(x,shape_param,scale_param,loc_params):
    if np.any(x<loc_params):
        return -np.inf
    answer = (shape_param-1)*np.log(x-loc_params) - (x-loc_params)/scale_param
    if np.any(np.isnan(answer)):
        print "NaN from log_gamma_prior!"
        print "x =", x
        print "shape_param =", shape_param
        print "scale_param =", scale_param
        print "loc_param =", loc_params
        sys.exit()
    else:
        return answer

def log_target_distribution(experiments,theta,shapes,scales,locs):
    dim = len(theta)
    Ne = len(experiments)
    if np.any(theta[:4] <= locs[:4]):
        return -np.inf
    alpha,beta,mu,s = theta[:4]
    hill_is = theta[4:-1:2]
    pic50_is = theta[5:-1:2]
    sigma = theta[-1]
    if np.any(hill_is<0) or np.any(pic50_is<-2) or (sigma<=locs[-1]): # these are just checking if in support of prior, roughly
        return -np.inf
    total = log_data_likelihood(hill_is,pic50_is,sigma,experiments)
    total += np.sum(log_hill_i_log_logistic_likelihood(hill_is,alpha,beta))
    total += np.sum(log_pic50_i_logistic_likelihood(pic50_is,mu,s))
    total += np.sum(log_gamma_prior(theta[[0,1,2,3,-1]],shapes,scales,locs))
    if np.isnan(total):
        print "NaN from log_target_distribution!"
        print "theta =", theta
        sys.exit()
    else:
        return total

def log_logistic_mode(alpha,beta): # from Wikipedia
    return alpha * ((beta-1.)/(beta+1.))**(1./beta)

def log_logistic_variance(alpha,beta): # from Wikipedia
    return alpha**2 * (2*np.pi/(beta*np.sin(2*np.pi/beta)) - (np.pi/(beta*np.sin(np.pi/beta)))**2)

def logistic_mode(mu,s): # from Wikipedia
    return mu

def logistic_variance(mu,s): # from Wikipedia
    return np.pi**2 * s**2 / 3.

def run(drug_channel):

    drug, channel = drug_channel
    
    print "\n\n{} + {}\n\n".format(drug,channel)

    # for reproducible results, otherwise choose a different seed
    seed = 1

    try:
        import cma
    except:
        print "couldn't find module cma"
        sys.exit()
    
    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    if (0 < (args.num_expts) < num_expts):
        num_expts = args.num_expts
        experiment_numbers = [x for x in experiment_numbers[:num_expts]]
        experiments = [x for x in experiments[:num_expts]]
    elif (args.num_expts==0):
        print "Fitting to all datasets\n"
    else:
        print "You've asked to fit to an impossible number of experiments for {} + {}\n".format(drug,channel)
        print "Therefore proceeding with all experiments in the input data file\n"
        
    # set up where to save chains and figures to
    # also renames anything with a '/' in its name and changes it to a '_'
    drug, channel, output_dir, chain_dir, figs_dir, chain_file = dr.hierarchical_output_dirs_and_chain_file(drug,channel,args.synthetic,num_expts)

    best_fits = []
    for expt in experiment_numbers:
        start = time.time()
        x0 = np.array([1.,2.5]) # not fitting sigma by CMA-ES
        sigma0 = 0.1
        opts = cma.CMAOptions()
        opts['seed'] = expt
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            X = es.ask()
            es.tell(X, [sum_of_square_diffs(x,experiments[expt][:,0],experiments[expt][:,1]) for x in X])
        res = es.result()
        best_fits.append((res[0][0]**2,res[0][1]**2-1,initial_sigma(len(experiments[expt][:,0]),res[1])))

    best_fits = np.array(best_fits)

    fig = plt.figure(figsize=(5.5,4.5))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    xmin = 1000
    xmax = -1000
    for expt in experiments:
        a = np.min(expt[:,0])
        b = np.max(expt[:,0])
        if a < xmin:
            xmin = a
        if b > xmax:
            xmax = b
    xmin = int(np.log10(xmin))-1
    xmax = int(np.log10(xmax))+3
    num_x_pts = 101
    x = np.logspace(xmin,xmax,num_x_pts)
    # from http://colorbrewer2.org
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
    skip_best_fits_plot = False
    if (num_expts>len(colors)):
        skip_best_fits_plot = True
        print "Not enough colours to print all experiments' best fits, so skipping that"
        
    if (not skip_best_fits_plot):
        for expt in experiment_numbers:
            ax.plot(x,dr.dose_response_model(x,best_fits[expt,0],dr.pic50_to_ic50(best_fits[expt,1])),color=colors[expt],lw=2)
            ax.scatter(experiments[expt][:,0],experiments[expt][:,1],label='Expt {}'.format(expt+1),color=colors[expt],s=100)
        ax.set_ylim(0,100)
        ax.set_xlim(min(x),max(x))
        ax.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
        ax.set_ylabel('% {} block'.format(channel))
        ax.legend(loc=2)
        ax.grid()
        ax.set_title('Hills = {}\nIC50s = {}'.format([round(best_fits[expt,0],2) for expt in experiment_numbers],[round(dr.pic50_to_ic50(best_fits[expt,1]),2) for expt in experiment_numbers]))
        fig.tight_layout()
        fig.savefig(figs_dir+'{}_{}_cma-es_best_fits.png'.format(drug,channel))
        fig.savefig(figs_dir+'{}_{}_cma-es_best_fits.pdf'.format(drug,channel))
    plt.close()
    
    locs = np.array([0.,2.,-4,0.01,1e-3]) # lower bounds for alpha,beta,mu,s,sigma

    sigma_cur = np.mean(best_fits[:,-1])
    if (sigma_cur <= locs[3]):
        sigma_cur = locs[3]+0.1
    print "sigma_cur =", sigma_cur
    
    # find initial alpha and beta values by fitting log-logistic distribution to best fits
    # there is an inbuilt fit function, but I found it to be unreliable for some reason
    x0 = np.array([0.5,0.5])
    sigma0 = 0.1
    opts = cma.CMAOptions()
    opts['seed'] = 1
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X = es.ask()
        es.tell(X, [-np.product(st.fisk.pdf(best_fits[:,0],c=x[1],scale=x[0],loc=0)) for x in X])
    res = es.result()
    
    alpha_cur, beta_cur = res[0]
    if alpha_cur <= locs[0]:
        alpha_cur = locs[0]+0.1
    if beta_cur <= locs[1]:
        beta_cur = locs[1]+0.1
    
    # here I have used the fit function, for some reason this one worked more consitently
    # but again, the starting point for MCMC is not too important
    # a bad starting position can increase the time you have to run MCMC for to get a "converged" output
    # at worst, it can get stuck in a local optimum, but we haven't found this to be a problem yet
    mu_cur, s_cur = st.logistic.fit(best_fits[:,1])
    if mu_cur <= locs[2]:
        mu_cur = locs[2]+0.1
    if s_cur <= locs[3]:
        s_cur = locs[3]+0.1

    first_iteration = np.concatenate(([alpha_cur,beta_cur,mu_cur,s_cur],best_fits[:,:-1].flatten(),[sigma_cur]))
    print "first mcmc iteration:\n", first_iteration
    
    # these are the numbers taken straight from Elkins (see paper for reference)
    elkins_hill_alphas = np.array([1.188, 1.744, 1.530, 0.930, 0.605, 1.325, 1.179, 0.979, 1.790, 1.708, 1.586, 1.469, 1.429, 1.127, 1.011, 1.318, 1.063])
    elkins_hill_betas = 1./np.array([0.0835, 0.1983, 0.2089, 0.1529, 0.1206, 0.2386, 0.2213, 0.2263, 0.1784, 0.1544, 0.2486, 0.2031, 0.2025, 0.1510, 0.1837, 0.1677, 0.0862])
    elkins_pic50_mus = np.array([5.235,5.765,6.060,5.315,5.571,7.378,7.248,5.249,6.408,5.625,7.321,6.852,6.169,6.217,5.927,7.414,4.860])
    elkins_pic50_sigmas = np.array([0.0760,0.1388,0.1459,0.2044,0.1597,0.2216,0.1856,0.1560,0.1034,0.1033,0.1914,0.1498,0.1464,0.1053,0.1342,0.1808,0.0860])

    elkins = [elkins_hill_alphas,elkins_hill_betas,elkins_pic50_mus,elkins_pic50_sigmas]

    # building Gamma prior distributions for alpha,beta,mu,s(,sigma, but sigma not from elkins)
    # wide enough to cover Elkins values and allow room for extra variation
    alpha_mode = np.mean(elkins_hill_alphas)
    beta_mode = np.mean(elkins_hill_betas)
    mu_mode = np.mean(elkins_pic50_mus)
    s_mode = np.mean(elkins_pic50_sigmas)
    sigma_mode = 6.

    modes = np.array([alpha_mode, beta_mode-2., mu_mode, s_mode, sigma_mode])

    print "modes:", modes
    
    # designed for priors to have modes at means of elkins data, but width is more important
    shapes = np.array([5.,2.5,7.5,2.5,5.]) # must all be greater than 1
    scales = (modes-locs)/(shapes-1.)

    labels = [r'$\alpha$',r'$\beta$',r'$\mu$',r'$s$',r'$\sigma$']
    file_labels = ['alpha','beta','mu','s','sigma']

    # ranges to plot priors
    mins = [0,0,-5,0,0]
    maxs = [8,22,20,2,25]
    
    total_axes = (6,4)
    fig = plt.figure(figsize=(6,7))
    for i in range(len(labels)-1):
        if i==0:
            axloc = (0,0)
        elif i==1:
            axloc = (0,2)
        elif i==2:
            axloc = (2,0)
        elif i==3:
            axloc = (2,2)
        ax = plt.subplot2grid(total_axes, axloc,colspan=2,rowspan=2)
        x_prior = np.linspace(mins[i],maxs[i],501)
        prior = st.gamma.pdf(x_prior,a=shapes[i],scale=scales[i],loc=locs[i])
        ax.plot(x_prior,prior,label='Gamma prior',lw=2)
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Probability density')
        ax.set_xlim(mins[i],maxs[i])
        ax.grid()
        priormax = np.max(prior)
        hist, bin_edges = np.histogram(elkins[i], bins=10)
        histmax = np.max(hist)
        w = bin_edges[1]-bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        # scaled histogram just to fit plot better, but this scaling doesn't matter
        ax.bar(bin_edges[:-1], priormax/histmax*hist, width=w,color='gray',edgecolor='grey')
    i = len(labels)-1
    ax = plt.subplot2grid(total_axes, (4,1),colspan=2,rowspan=2)
    x_prior = np.linspace(mins[i],maxs[i],501)
    ax.plot(x_prior,st.gamma.pdf(x_prior,a=shapes[i],scale=scales[i],loc=locs[i]),label='Gamma prior',lw=2)
    ax.set_xlabel(labels[i])
    ax.set_ylabel('Probability density')
    ax.set_xlim(mins[i],maxs[i])
    ax.grid()
    fig.tight_layout()
    fig.savefig(figs_dir+'all_prior_distributions.png')
    fig.savefig(figs_dir+'all_prior_distributions.pdf')
    plt.close()
    
    #sys.exit # uncomment this if you just want to plot the priors and then quit

    # create/wipe MCMC output file
    with open(chain_file,'w') as outfile:
        outfile.write("# Hill ~ log-logistic(alpha,beta), pIC50 ~ logistic(mu,s)\n")
        outfile.write("# alpha, beta, mu, s, hill_1, pic50_1, hill_2, pic50_2, ..., hill_Ne, pic50_Ne, sigma\n") # this is the order of parameters stored in the chain


    # have to choose initial covariance matrix for proposal distribution
    # we set it to a diagonal with entries scaled to the initial parameter values
    first_cov = np.diag(0.01*np.abs(first_iteration))

    mean_estimate = np.copy(first_iteration)

    dim = len(first_iteration)
    
    # we do not start adaptation straight away
    # just to give the algorithm a chance to look around
    # many of these pre-adaptation proposals will probably be rejected, if the initial step size is too lareg
    when_to_adapt = 100*dim

    theta_cur = np.copy(first_iteration)
    cov_cur = np.copy(first_cov)

    print "theta_cur =", theta_cur

    log_target_cur = log_target_distribution(experiments,theta_cur,shapes,scales,locs)

    print "initial log_target_cur =", log_target_cur

    # effectively step size, scales covariance matrix
    loga = 0.
    # what fraction of proposed samples are being accepted into the chain
    acceptance = 0.
    # what fraction of samples we WANT accepted into the chain
    # loga updates itself to try to make this dream come true
    target_acceptance = 0.25

    # perform thinning to reduce autocorrelation (make saved iterations more closely represent independent samples from target distribution)
    # also saves file space, win win
    thinning = 5

    try:
        total_iterations = args.iterations
    except:
        total_iterations = 200000
    # after what fraction of total_iterations to print a little status message
    status_when = 10000
    saved_iterations = total_iterations/thinning+1
    pre_thin_burn = total_iterations/4
    # we discard the first quarter of iterations, as this gen
    burn = saved_iterations/4

    # pre-allocate the space for MCMC iterations
    # not a problem when we don't need to do LOADS of iterations
    # but might become more of a hassle if we wanted to run it for ages along with loads of parameters
    chain = np.zeros((saved_iterations,dim+1))
    chain[0,:] = np.copy(np.concatenate((first_iteration,[log_target_cur])))


    # MCMC!
    start = time.time()
    t = 1
    while t <= total_iterations:
        theta_star = npr.multivariate_normal(theta_cur,np.exp(loga)*cov_cur)
        log_target_star = log_target_distribution(experiments,theta_star,shapes,scales,locs)
        accept_prob = npr.rand()
        if (np.log(accept_prob) < log_target_star - log_target_cur):
            theta_cur = theta_star
            log_target_cur = log_target_star
            accepted = 1
        else:
            accepted = 0
        acceptance = ((t-1.)*acceptance + accepted)/t
        if (t>when_to_adapt):
            s = t - when_to_adapt
            gamma_s = 1/(s+1)**0.6
            temp_covariance_bit = np.array([theta_cur-mean_estimate])
            cov_cur = (1-gamma_s) * cov_cur + gamma_s * np.dot(np.transpose(temp_covariance_bit),temp_covariance_bit)
            mean_estimate = (1-gamma_s) * mean_estimate + gamma_s * theta_cur
            loga += gamma_s*(accepted-target_acceptance)
        if t%thinning==0:
            chain[t/thinning,:] = np.concatenate((np.copy(theta_cur),[log_target_cur]))
        if (t%status_when==0):
            print "{} / {}".format(t/status_when,total_iterations/status_when)
            time_taken_so_far = time.time()-start
            estimated_time_left = time_taken_so_far/t*(total_iterations-t)
            print "Time taken: {} s = {} min".format(np.round(time_taken_so_far,1),np.round(time_taken_so_far/60,2))
            print "acceptance = {}".format(np.round(acceptance,5))
            print "Estimated time remaining: {} s = {} min".format(np.round(estimated_time_left,1),np.round(estimated_time_left/60,2))
        t += 1
    print "**********"
    print "final_iteration =", chain[-1,:]
    with open(chain_file,'a') as outfile:
        np.savetxt(outfile,chain)
        
    # we currently only run the AP predictions on the real data
    if (not args.synthetic):
        indices = npr.randint(burn,saved_iterations,args.num_APs)
        samples_file = dr.alpha_mu_downsampling(drug,channel,args.synthetic)
        AP_samples = chain[indices,:]
        print "saving (alpha,mu) samples to", samples_file
        with open(samples_file,'w') as outfile:
            outfile.write('# {} (alpha,mu) samples from hierarchical MCMC for {} + {}\n'.format(args.num_APs,drug,channel))
            np.savetxt(outfile,AP_samples[:,[0,2]])
        
    # this can be a quick visual check to see if the chain is mixing well
    # it will plot one big tall figure with all parameter paths plotted
    if args.plot_parameter_paths:
        fig = plt.figure(figsize=(10,4*dim))
        ax0 = fig.add_subplot(dim,1,1)
        ax0.plot(chain[:,0])
        ax0.set_ylabel(r'$\alpha$')
        plt.setp(ax0.get_xticklabels(), visible=False)
        for i in range(1,dim):
            ax = fig.add_subplot(dim,1,i+1,sharex=ax0)
            ax.plot(chain[:t,i])
            if i < dim-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            elif i==1:
                y_label = r'$\beta$'
            elif i==2:
                y_label = r'$\mu$'
            elif i==3:
                y_label = r'$s$'
            elif (i%2==0)and(i<dim-1):
                y_label = r'$Hill_{'+str(i/2-1)+'}$'
            elif (i<dim-1):
                y_label = r'$pIC50_{'+str(i/2-1)+'}$'
            else:
                y_label = r'$\sigma$'
                ax.set_xlabel('Iteration (thinned)')
            ax.set_ylabel(y_label)
        fig.tight_layout()
        fig.savefig(figs_dir+'{}_{}_parameter_paths.png'.format(drug,channel))
        plt.close()
        
    # plot all marginal posteriors separately, after discarding burn-in
    # also a good visual check to see if it looks like they have converged
    marginals_dir = figs_dir+'marginals/png/'
    if not os.path.exists(marginals_dir):
        os.makedirs(marginals_dir)
    for i in range(dim):
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        ax.hist(chain[burn:,i],bins=50,normed=True,color='blue',edgecolor='blue')
        ax.set_ylabel('Marginal probability density')
        if i==0:
            x_label = r'$\alpha$'
            filename = 'alpha'
        elif i==1:
            x_label = r'$\beta$'
            filename = 'beta'
        elif i==2:
            x_label = r'$\mu$'
            filename = 'mu'
        elif i==3:
            x_label = r'$s$'
            filename = 's'
        elif (i%2==0)and(i<dim-1):
            x_label = r'$Hill_{'+str(i/2-1)+'}$'
            filename = 'hill_{}'.format(i/2-1)
        elif (i<dim-1):
            x_label = r'$pIC50_{'+str(i/2-1)+'}$'
            filename = 'pic50_{}'.format(i/2-1)
        else:
            x_label = r'$\sigma$'
            filename = 'sigma'
        ax.set_xlabel(x_label)
        fig.tight_layout()
        fig.savefig(marginals_dir+'{}_{}_{}_marginal.png'.format(drug,channel,filename))
        #fig.savefig(marginals_dir+'{}_{}_{}_marginal.pdf'.format(drug,channel,filename))
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


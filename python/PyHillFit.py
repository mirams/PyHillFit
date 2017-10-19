import doseresponse as dr
import numpy as np
import numpy.random as npr
import itertools as it
import time
import sys
import os
import argparse
import scipy.stats as st

try:
    import cma
except:
    sys.exit("couldn't find module cma")

import matplotlib
"""I have found that these two lines are needed on *some* computers to prevent matplotlib figure windows from opening.
In general, I save the figures but do not actually open the matplotlib figure windows.
Try uncommenting this line if annoying unwanted figure windows open."""
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--iterations", type=int, help="number of MCMC iterations",default=500000)
parser.add_argument("-t", "--thinning", type=int, help="how often to thin the MCMC, i.e. save every t-th iteration",default=5)
parser.add_argument("-b", "--burn-in-fraction", type=int, help="given N saved MCMC iterations, discard the first N/b as burn-in",default=4)
parser.add_argument("-a", "--all", action='store_true', help='run hierarchical MCMC on all drugs and channels', default=False)
parser.add_argument('-ppp', '--plot-parameter-paths', action='store_true', help='plot the path taken by each parameter through the (thinned) MCMC',default=False)
parser.add_argument("-c", "--num-cores", type=int, help="number of cores to parallelise drug/channel combinations",default=1)
parser.add_argument("-Ne", "--num_expts", type=int, help="how many experiments to fit to", default=0)
parser.add_argument("--num-APs", type=int, help="how many (alpha,mu) samples to take for AP simulations", default=500)
parser.add_argument("--single", action='store_true', help="run single-level MCMC algorithm",default=True)
parser.add_argument("--hierarchical", action='store_true', help="run hierarchical MCMC algorithm",default=False)
parser.add_argument("--fix-hill", action='store_true', help="fix Hill=1 through fitting and MCMC",default=False)
parser.add_argument("-bfo", "--best-fit-only", action='store_true', help="only do CMA-ES best fit, then quit",default=False)

requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--data-file", type=str, help="csv file from which to read in data, in same format as provided crumb_data.csv", required=True)
requiredNamed.add_argument("-m", "--model", type=int, help="For non-hierarchical (put anything for hierarchical):1. fix Hill=1; 2. vary Hill", required=True)

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

dr.define_model(args.model)
temperature = 1
num_params = dr.num_params

# load data from specified data file
dr.setup(args.data_file)

# list drug and channel options, select from command line
# can select more than one of either
drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)


# log-likelihood (same as log-target for uniform priors) for single-level MCMC
def log_likelihood_single_vary_hill(measurements,doses,theta):
    hill, pIC50, sigma = theta
    IC50 = dr.pic50_to_ic50(pIC50)
    return -len(measurements) * np.log(sigma) - np.sum((measurements-dr.dose_response_model(doses,hill,IC50))**2)/(2.*sigma**2)
    # as usual in our MCMC, omitted the -n/2*log(2pi) term from the log-likelihood, as this is always cancelled out


# log-likelihood (same as log-target for uniform priors) for single-level MCMC
def log_likelihood_single_fix_hill(measurements,doses,theta):
    # using hill = 1, but not bothering to assign it
    pIC50, sigma = theta
    IC50 = dr.pic50_to_ic50(pIC50)
    return -len(measurements) * np.log(sigma) - np.sum((measurements-dr.dose_response_model(doses,1,IC50))**2)/(2.*sigma**2)
    # as usual in our MCMC, omitted the -n/2*log(2pi) term from the log-likelihood, as this is always cancelled out
    
    
# for finding starting point for MCMC, so if we later decide pIC50 can go down to -2, it doesn't matter, it will just take a few
# iterations to decide if it wants to go in that direction
def sum_of_square_diffs(_params,doses,responses):
    pIC50, hill = _params
    IC50 = dr.pic50_to_ic50(pIC50)
    test_responses = dr.dose_response_model(doses,hill,IC50)
    return np.sum((test_responses-responses)**2)


# analytic solution for sigma to maximise likelihood from Normal distribution
def initial_sigma(n,sum_of_squares):
    return np.sqrt((1.*sum_of_squares)/n)
    
    
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
            

def log_target_distribution(experiments,theta,shapes,scales,locs):
    dim = len(theta)
    Ne = len(experiments)
    if np.any(theta[:4] <= locs[:4]):
        return -np.inf
    alpha,beta,mu,s = theta[:4]
    hill_is = theta[4:-1:2]
    pic50_is = theta[5:-1:2]
    sigma = theta[-1]
    if np.any(hill_is<0) or np.any(pic50_is<pic50_prior[0]) or (sigma<=locs[-1]): # these are just checking if in support of prior, roughly
        return -np.inf
    total = log_data_likelihood(hill_is,pic50_is,sigma,experiments)
    total += np.sum(log_hill_i_log_logistic_likelihood(hill_is,alpha,beta))
    total += np.sum(log_pic50_i_logistic_likelihood(pic50_is,mu,s))
    total += np.sum(dr.log_gamma_prior(theta[[0,1,2,3,-1]],shapes,scales,locs))
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


# hierarchical MCMC
def run_hierarchical(drug_channel):
    global pic50_prior
    pic50_prior = [-2]  # bad way to deal with sum_of_square_diffs in hierarchical case

    drug, channel = drug_channel
    
    print "\n\n{} + {}\n\n".format(drug,channel)

    # for reproducible results, otherwise choose a different seed
    seed = 1
    
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
    drug, channel, output_dir, chain_dir, figs_dir, chain_file = dr.hierarchical_output_dirs_and_chain_file(drug,channel,num_expts)
    

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
    
    
    locs = np.array([0.,2.,-4,0.01,dr.sigma_loc]) # lower bounds for alpha,beta,mu,s,sigma

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
    sigma_mode = dr.sigma_mode

    modes = np.array([alpha_mode, beta_mode-2., mu_mode, s_mode, sigma_mode])

    print "modes:", modes
    
    # designed for priors to have modes at means of elkins data, but width is more important
    
    
    
    shapes = np.array([5.,2.5,7.5,2.5,dr.sigma_shape]) # must all be greater than 1
    scales = (modes-locs)/(shapes-1.)

    labels = [r'$\alpha$',r'$\beta$',r'$\mu$',r'$s$',r'$\sigma$']
    file_labels = ['alpha','beta','mu','s','sigma']

    # ranges to plot priors
    mins = [0,0,-5,0,0]
    maxs = [8,22,20,2,25]
    
    prior_xs = []
    priors = []
    
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
        prior_xs.append(x_prior)
        priors.append(prior)
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
    prior = st.gamma.pdf(x_prior,a=shapes[i],scale=scales[i],loc=locs[i])
    ax.plot(x_prior,prior,label='Gamma prior',lw=2)
    prior_xs.append(x_prior)
    priors.append(prior)
    
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
    thinning = args.thinning

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
        
    # save (alpha,mu) samples to be used as (Hill,pIC50) values in AP simulations
    # these are direct 'top-level' samples, not samples from the posterior predictive distributions
    indices = npr.randint(burn,saved_iterations,args.num_APs)
    samples_file = dr.alpha_mu_downsampling(drug,channel)
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
        
    total_axes = (6,4)
    fig = plt.figure(figsize=(6,7))
    for i in range(5): # have to do sigma separately
        if i==0:
            axloc = (0,0)
        elif i==1:
            axloc = (0,2)
        elif i==2:
            axloc = (2,0)
        elif i==3:
            axloc = (2,2)
        elif i==4:
            axloc = (4,0)
        ax = plt.subplot2grid(total_axes, axloc,colspan=2,rowspan=2)
        
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Probability density')
        
        ax.grid()
        if (i<4):
            min_sample = np.min(chain[burn:,i])
            max_sample = np.max(chain[burn:,i])
            ax.hist(chain[burn:,i],bins=50,normed=True,color='blue',edgecolor='blue')
        elif (i==4):
            min_sample = np.min(chain[burn:,-2])
            max_sample = np.max(chain[burn:,-2])
            ax.hist(chain[burn:,-2],bins=50,normed=True,color='blue',edgecolor='blue')  # -1 would be log-target
        ax.set_xlim(min_sample,max_sample)
        pts_in_this_range = np.where((prior_xs[i] >= min_sample) & (prior_xs[i] <= max_sample))
        x_in_this_range = prior_xs[i][pts_in_this_range]
        prior_in_this_range = priors[i][pts_in_this_range]
        line = ax.plot(x_in_this_range,prior_in_this_range,lw=2,color='red',label='Prior distributions')
        if (i==0 or i==3):
            plt.xticks(rotation=90)
        
    leg_ax = plt.subplot2grid(total_axes, (4,2),colspan=2,rowspan=2)
    leg_ax.axis('off')
    hist = mpatches.Patch(color='blue', label='Normalised histograms')
    leg_ax.legend(handles=line+[hist],loc="center",fontsize=12,bbox_to_anchor=[0.38,0.7])
    
    fig.tight_layout()
    fig.savefig(figs_dir+'all_prior_distributions_and_marginals.png')
    fig.savefig(figs_dir+'all_prior_distributions_and_marginals.pdf')
    plt.close()
        
    print "Marginal plots saved in", marginals_dir

    print "\n\n{} + {} complete!\n\n".format(drug,channel)
    
# single-level MCMC
def run_single_level(drug_channel):

    drug, channel = drug_channel
    
    seed = 100

    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(args.model, drug, channel, temperature)
    
    concs = np.array([])
    responses = np.array([])
    for i in xrange(num_expts):
        concs = np.concatenate((concs,experiments[i][:,0]))
        responses = np.concatenate((responses,experiments[i][:,1]))
        
    print experiments
    print concs
    print responses
    
    where_r_0 = responses==0
    where_r_100 = responses==100
    where_r_other = (0<responses) & (responses<100)
    
    print "where_r_0:", where_r_0
    print "where_r_100:", where_r_100
    print "where_r_other:", where_r_other
    
    pi_bit = dr.compute_pi_bit_of_log_likelihood(where_r_other)
    
    # plot priors
    for i in xrange(num_params):
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(dr.prior_xs[i], dr.prior_pdfs[i], color='blue', lw=2)
        ax.set_xlabel(dr.labels[i])
        ax.set_ylabel("Prior pdf")
        fig.tight_layout()
        fig.savefig(images_dir+dr.file_labels[i]+"_prior_pdf.pdf")
        plt.close()

    start = time.time()
    
    sigma0 = 0.1
    opts = cma.CMAOptions()
    opts['seed'] = seed
    if args.model==1:
        #x0 = np.array([2.5, 3.])
        x0 = np.array([2.5, 1.])
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            X = es.ask()
            #es.tell(X, [-dr.log_target(responses, where_r_0, where_r_100, where_r_other, concs, x**2 + [dr.pic50_exp_lower,dr.sigma_uniform_lower], temperature, pi_bit) for x in X])
            es.tell(X, [sum_of_square_diffs([x[0]**2+dr.pic50_exp_lower, 1.],concs,responses) for x in X])
            es.disp()
        res = es.result()
        #pic50_cur, sigma_cur = res[0]**2 + [dr.pic50_exp_lower, dr.sigma_uniform_lower]
        pic50_cur = res[0][0]**2 + dr.pic50_exp_lower
        hill_cur = 1
    elif args.model==2:
        #x0 = np.array([2.5, 1., 3.])
        x0 = np.array([2.5, 1.])
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            X = es.ask()
            #es.tell(X, [-dr.log_target(responses, where_r_0, where_r_100, where_r_other, concs, x**2 + [dr.pic50_exp_lower, dr.hill_uniform_lower, dr.sigma_uniform_lower], temperature, pi_bit) for x in X])
            es.tell(X, [sum_of_square_diffs(x**2+[dr.pic50_exp_lower, dr.hill_uniform_lower],concs,responses) for x in X])
            es.disp()
        res = es.result()
        #pic50_cur, hill_cur, sigma_cur = res[0]**2 + [dr.pic50_exp_lower, dr.hill_uniform_lower, dr.sigma_uniform_lower]
        pic50_cur, hill_cur = res[0]**2 + [dr.pic50_exp_lower, dr.hill_uniform_lower]
    

    sigma_cur = initial_sigma(len(responses),res[1])
    print "sigma_cur:", sigma_cur
    
    if args.model==1:
        theta_cur = np.array([pic50_cur,sigma_cur])
    elif args.model==2:
        theta_cur = np.array([pic50_cur,hill_cur,sigma_cur])
        
    print "theta_cur:", theta_cur
    
    best_params_file = images_dir+"{}_{}_best_fit_params.txt".format(drug, channel)
    with open(best_params_file, "w") as outfile:
        outfile.write("# CMA-ES best fit params\n")
        if args.model==1:
            outfile.write("# pIC50, sigma, (Hill=1, not included)\n")
        elif args.model==2:
            outfile.write("# pIC50, Hill, sigma\n")
        np.savetxt(outfile, [theta_cur])
    
    proposal_scale = 0.05

    mean_estimate = np.copy(theta_cur)
    cov_estimate = proposal_scale*np.diag(np.copy(np.abs(theta_cur)))
    
    cmaes_ll = dr.log_target(responses, where_r_0, where_r_100, where_r_other, concs, theta_cur, temperature, pi_bit)
    print "cmaes_ll:", cmaes_ll
        
    best_fit_fig = plt.figure(figsize=(5,4))
    best_fit_ax = best_fit_fig.add_subplot(111)
    best_fit_ax.set_xscale('log')
    best_fit_ax.grid()
    plot_lower_lim = int(np.log10(np.min(concs)))-1
    plot_upper_lim = int(np.log10(np.max(concs)))+2
    best_fit_ax.set_xlim(10**plot_lower_lim,10**plot_upper_lim)
    best_fit_ax.set_ylim(0,100)
    num_x_pts = 1001
    x_range = np.logspace(plot_lower_lim,plot_upper_lim,num_x_pts)
    best_fit_curve = dr.dose_response_model(x_range,hill_cur,dr.pic50_to_ic50(pic50_cur))
    best_fit_ax.plot(x_range,best_fit_curve,label='Best fit',lw=2)
    best_fit_ax.set_ylabel('% {} block'.format(channel))
    best_fit_ax.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    best_fit_ax.set_title(r'$pIC50 = {}, Hill = {}; SS = {}$'.format(np.round(pic50_cur,2),np.round(hill_cur,2),round(res[1],2)))
    best_fit_ax.plot(concs,responses,"o",color='orange',ms=10,label='Data',zorder=10)
    best_fit_ax.legend(loc=2)
    best_fit_fig.tight_layout()
    best_fit_fig.savefig(images_dir+'{}_{}_model_{}_CMA-ES_best_fit.png'.format(drug,channel,args.model))
    best_fit_fig.savefig(images_dir+'{}_{}_model_{}_CMA-ES_best_fit.pdf'.format(drug,channel,args.model))
    plt.close()
    
    if args.best_fit_only:
        print "\nStopping {}+{} after doing and plotting best fit\n".format(drug, channel)
        return None

    # let MCMC look around for a bit before adaptive covariance matrix
    # same rule (100*dimension) as in hierarchical case
    when_to_adapt = 1000*num_params

    log_target_cur = dr.log_target(responses, where_r_0, where_r_100, where_r_other, concs, theta_cur, temperature, pi_bit)
    
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
    
    
    
    # for reproducible results, otherwise select a new random seed
    seed = 25
    npr.seed(seed)

    # MCMC!
    t = 1
    start = time.time()
    while t <= total_iterations:
        theta_star = npr.multivariate_normal(theta_cur,np.exp(loga)*cov_estimate)
        accepted = 0
        log_target_star = dr.log_target(responses, where_r_0, where_r_100, where_r_other, concs, theta_star, temperature, pi_bit)
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
    
    burn_fraction = args.burn_in_fraction
    burn = saved_iterations/burn_fraction
    
    chain = chain[burn:,:]  # remove burn-in before saving
    with open(chain_file,'w') as outfile:
        outfile.write('# Nonhierarchical MCMC output for {} + {}: (Hill,pIC50,sigma,log-target)\n'.format(drug,channel))
        np.savetxt(outfile,chain)       


    best_ll_index = np.argmax(chain[:,num_params])
    best_ll_row = chain[best_ll_index,:]
    print "Best log-likelihood:", "\n", best_ll_row

    figs = []
    axs = []
    # plot all marginal posterior distributions
    for i in range(num_params):
        figs.append(plt.figure())
        axs.append([])
        axs[i].append(figs[i].add_subplot(211))
        axs[i][0].hist(chain[:,i], bins=40, normed=True, color='blue', edgecolor='blue')
        axs[i][0].legend()
        axs[i][0].set_title("MCMC marginal distributions")
        axs[i][0].set_ylabel("Normalised frequency")
        axs[i][0].grid()
        plt.setp(axs[i][0].get_xticklabels(), visible=False)
        axs[i].append(figs[i].add_subplot(212,sharex=axs[i][0]))
        axs[i][1].plot(chain[:,i],range(burn,saved_iterations))
        axs[i][1].invert_yaxis()
        axs[i][1].set_xlabel(dr.labels[i])
        axs[i][1].set_ylabel('Saved MCMC iteration')
        axs[i][1].grid()
        figs[i].tight_layout()
        figs[i].savefig(images_dir+'{}_{}_model_{}_{}_marginal.png'.format(drug,channel,args.model,dr.file_labels[i]))
        plt.close()

    # plot log-target path
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.plot(range(burn, saved_iterations), chain[:,-1])
    ax3.set_xlabel('MCMC iteration')
    ax3.set_ylabel('log-target')
    ax3.grid()
    fig2.tight_layout()
    fig2.savefig(images_dir+'log_target.png')
    plt.close()

    # plot scatterplot matrix of posterior(s)
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
                    axes[ij].hist(chain[:,i],bins=50,normed=True,color='blue', edgecolor='blue')
                elif j==0: # this column shares x-axis with top-left
                    axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes["00"])
                    counts, xedges, yedges, Image = axes[ij].hist2d(chain[:,j],chain[:,i],cmap='hot_r',bins=50,norm=norm)
                    maxcounts = np.amax(counts)
                    if maxcounts > colormax:
                        colormax = maxcounts
                    mincounts = np.amin(counts)
                    if mincounts < colormin:
                        colormin = mincounts
                else:
                    axes[ij] = matrix_fig.add_subplot(num_params,num_params,subplot_position,sharex=axes[str(j)+str(j)],sharey=axes[str(i)+"0"])
                    counts, xedges, yedges, Image = axes[ij].hist2d(chain[:,j],chain[:,i],cmap='hot_r',bins=50,norm=norm)
                    maxcounts = np.amax(counts)
                    if maxcounts > colormax:
                        colormax = maxcounts
                    mincounts = np.amin(counts)
                    if mincounts < colormin:
                        colormin = mincounts
                axes[ij].xaxis.grid()
                if (i!=j):
                    axes[ij].yaxis.grid()
                if i!=num_params-1:
                    hidden_labels.append(axes[ij].get_xticklabels())
                if j!=0:
                    hidden_labels.append(axes[ij].get_yticklabels())
                if i==j==0:
                    hidden_labels.append(axes[ij].get_yticklabels())
                if i==num_params-1:
                    axes[str(i)+str(j)].set_xlabel(dr.labels[j], fontsize=18)
                if j==0 and i>0:
                    axes[str(i)+str(j)].set_ylabel(dr.labels[i], fontsize=18)
                    
                plt.xticks(rotation=30)
        norm = matplotlib.colors.Normalize(vmin=colormin,vmax=colormax)
        count += 1

        
    plt.setp(hidden_labels, visible=False)
    
    matrix_fig.tight_layout()
    matrix_fig.savefig(images_dir+"{}_{}_model_{}_scatterplot_matrix.png".format(drug,channel,args.model))
    matrix_fig.savefig(images_dir+"{}_{}_model_{}_scatterplot_matrix.pdf".format(drug,channel,args.model))
    plt.close()


    print "\n\n{} + {} complete!\n\n".format(drug,channel)
    
    
if args.hierarchical:
    run = run_hierarchical
elif (not args.hierarchical): # assume single-level MCMC if hierarchical not specified
    run = run_single_level

drugs_channels = it.product(drugs_to_run,channels_to_run)
if (args.num_cores<=1) or (len(drugs_to_run)==1):
    for drug_channel in drugs_channels:
        run(drug_channel)
        
        # try/except is good when running multiple MCMCs and leaving them overnight,say
        # if one or more crash then the others will survive!
        # however, if you need more "control", comment out the try/except, and uncomment the other run(drug_channel) line
        #try:
        #    run(drug_channel)
        #except Exception,e:
        #    print e
        #    print "Failed to run {} + {}!".format(drug_channel[0],drug_channel[1])
# run multiple MCMCs in parallel
elif (args.num_cores>1):
    import multiprocessing as mp
    num_cores = min(args.num_cores, mp.cpu_count()-1)
    pool = mp.Pool(processes=num_cores)
    pool.map_async(run,drugs_channels).get(99999)
    pool.close()
    pool.join()


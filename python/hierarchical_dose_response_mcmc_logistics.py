import doseresponse as dr
import numpy as np
import numpy.random as npr
import itertools as it
import time
import os
import argparse
import scipy.stats as st
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
#matplotlib.rcParams.update({'font.size': 22})
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, help="number of MCMC iterations",default=1000000)
parser.add_argument("-n", "--num_params", type=int, help="number of parameters to fit (or fix Hill=1)",default=1000000)
parser.add_argument("-a", "--all", action='store_true', help='run hierarchical MCMC on all drugs and channels', default=False)
parser.add_argument('-ppp', '--plot-parameter-paths', action='store_true', help='plot the path taken by each parameter through the (thinned) MCMC',default=False)
parser.add_argument("-p", "--predictions", type=int, help="number of prediction curves to plot",default=1000)
args = parser.parse_args()
num_params = args.num_params+1

drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)

def log_likelihood(measurements,doses,theta):
    if num_params==3:
        hill = theta[0]
        pIC50 = theta[1]
        sigma = theta[2]
    elif num_params==2:
        hill = 1
        pIC50 = theta[0]
        sigma = theta[1]
    IC50 = dr.pic50_to_ic50(pIC50)
    return -len(measurements) * np.log(sigma) - np.sum((measurements-dr.dose_response_model(doses,hill,IC50))**2)/(2.*sigma**2)
    # as usual in our MCMC, omitted the -n/2*log(2pi) term from the log-likelihood, as this is always cancelled out
    
def sum_of_square_diffs(unscaled_params,doses,responses):
    #hill = hill_prior[0] + 0.5*(hill_prior[1]-hill_prior[0]) * (1-np.cos(np.pi/2*unscaled_params[0]))
    #pIC50 = pic50_prior[0] + 0.5*(pic50_prior[1]-pic50_prior[0]) * (1-np.cos(np.pi/2*unscaled_params[1]))
    hill = unscaled_params[0]**2
    pIC50 = unscaled_params[1]**2-1
    IC50 = dr.pic50_to_ic50(pIC50)
    if num_params==2: # this is still quick, but maybe worth using different solver (CMA-ES doesn't do 1d) for loads of data points
        hill = 1
    test_responses = dr.dose_response_model(doses,hill,IC50)
    return np.sum((test_responses-responses)**2)

def initial_sigma(n,sum_of_squares):
    return np.sqrt(sum_of_squares/n)

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
        truncated_scale = np.sum(np.log(st.norm.cdf(100,model_responses,sigma)-st.norm.cdf(0,model_responses,sigma)))
        answer -= (num_expt_pts*np.log(sigma) + exp_bit + truncated_scale)
    if (answer == -np.inf):
        print "inf in log_data_likelihood"
    if np.isnan(answer):
        print "NaN from log_data_likelihood!"
        print "hill_is =", hill_is
        print "pic50_is =", pic50_is
        print "sigma =", sigma
        sys.exit()
    return answer

def log_param_i_likelihood(x,param_mu,param_sigma,param_lower_bound): # Normal only truncated on LHS
    answer = -np.log(1-st.norm.cdf(param_lower_bound,param_mu,param_sigma)) - np.log(param_sigma) - (x-param_mu)**2/(2*param_sigma**2)
    if (np.any(answer == -np.inf)):
        print "-inf from log_param_i_likelihood"
    if np.isnan(answer):
        print "NaN from log_param_i_likelihood!"
        sys.exit()
    return answer

def log_hill_i_log_logistic_likelihood(x,alpha,beta):
    answer = np.log(beta) - beta*np.log(alpha) + (beta-1.)*np.log(x) - 2*np.log(1+(x/alpha)**beta)
    if (np.any(answer == -np.inf)):
        print "-inf from log_hill_i_log_logistic_likelihood"
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
    if (np.any(answer == -np.inf)):
        print "-inf from log_pic50_i_logistic_likelihood"
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
        print "-inf from log_beta_prior"
        print "x =", x
        return -np.inf
    else:
        answer = (alpha-1)*np.log(x-a) + (beta-1)*np.log(b-x)
        if (np.any(answer==-np.inf)):
            print "-inf from log_beta_prior"
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
        print "-inf from log_gamma_prior"
        print "x =", x
        return -np.inf
    answer = (shape_param-1)*np.log(x-loc_params) - (x-loc_params)/scale_param
    if (np.any(answer==-np.inf)):
        print "-inf from log_beta_prior"
        print "x:", x
        print "shape_param:", shape_param
        print "scale_param:", scale_param
        print "loc_params:", loc_params
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
    alpha,beta,mu,s = theta[:4]
    hill_is = theta[4:-1:2]
    pic50_is = theta[5:-1:2]
    sigma = theta[-1]
    if (alpha<=locs[0]) or (beta<=locs[1]) or (s<=locs[2]) or np.any(hill_is<0) or np.any(pic50_is<-2) or (sigma <= locs[3]): # these are just checking if in support of prior, roughly
        return -np.inf
    
    total = log_data_likelihood(hill_is,pic50_is,sigma,experiments)
    total += np.sum(log_hill_i_log_logistic_likelihood(hill_is,alpha,beta))
    total += np.sum(log_pic50_i_logistic_likelihood(pic50_is,mu,s))
    total += np.sum(log_gamma_prior(theta[[0,1,3,-1]],shapes,scales,locs))
    total += log_pic50_mu_prior(theta[2],-2,7,scale=10)
    
    if total == -np.inf:
        print"log_target_distribution = -inf!"
        
    if np.isnan(total):
        print "NaN from log_target_distribution!"
        print "theta =", theta
        sys.exit()
    else:
        return total

def scale_indices(theta):
    indices_to_square = [0,1,3] + range(4,len(theta),2)+[-1]
    temp_theta = np.copy(theta)
    temp_theta[indices_to_square] = temp_theta[indices_to_square]**2
    return temp_theta

def negative_log_target_distribution(experiments,theta,shapes,scales,locs):
    temp_theta = scale_indices(theta)
    return -log_target_distribution(experiments,temp_theta,shapes,scales,locs)

def log_logistic_mode(alpha,beta):
    return alpha * ((beta-1.)/(beta+1.))**(1./beta)

def log_logistic_variance(alpha,beta): # from Wikipedia
    return alpha**2 * (2*np.pi/(beta*np.sin(2*np.pi/beta)) - (np.pi/(beta*np.sin(np.pi/beta)))**2)

def logistic_mode(mu,s):
    return mu

def logistic_variance(mu,s): # from Wikipedia
    return np.pi**2 * s**2 / 3.

def log_pic50_mu_prior(x,uniform_lower_bound=-2,uniform_upper_bound=7,scale=10):
    k = 1./(9.+np.pi/(2.*scale))
    if (x<uniform_lower_bound):
        return -np.inf
    elif (uniform_lower_bound <= x < uniform_upper_bound):
        return np.log(k)
    else:
        return np.log(k) - (x-uniform_upper_bound)**2/scale


# In[26]:

def run(drug,channel):

    seed = 1
    num_params = 3

    try:
        import cma
    except:
        print "cma module not installed or something."
        sys.exit()
    
    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug,channel)
    drug, channel, output_dir, chain_dir, figs_dir, chain_file = dr.hierarchical_output_dirs_and_chain_file(drug,channel)

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
            #es.disp()
        res = es.result()
        #print "Time taken: {} s".format(time.time()-start)
        best_fits.append((res[0][0]**2,res[0][1]**2-1,initial_sigma(len(experiments[expt][:,0]),res[1])))

    best_fits = np.array(best_fits)
    print best_fits
    #print np.var(best_fits,axis=0)

    plt.close()
    fig = plt.figure(figsize=(10,8))
    #fig.canvas.mpl_connect('pick_event',onpick)
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
    xmax = int(np.log10(xmax))+2
    num_x_pts = 101
    x = np.logspace(xmin,xmax,num_x_pts)
    colors = ['red','blue','orange','cyan','purple']
    for expt in experiment_numbers:
        print expt
        print colors[expt]
        ax.plot(x,dr.dose_response_model(x,best_fits[expt,0],dr.pic50_to_ic50(best_fits[expt,1])),color=colors[expt])#,label='Expt {} best fit'.format(expt+1))
        ax.scatter(experiments[expt][:,0],experiments[expt][:,1],label='Expt {}'.format(expt+1),color=colors[expt],s=100)
    ax.set_ylim(0,100)
    ax.set_xlim(min(x),max(x))
    ax.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    ax.set_ylabel('% {} block'.format(channel))
    ax.legend(loc=2)
    fig.tight_layout()
    fig.savefig(figs_dir+'{}_{}_cma-es_best_fits.png'.format(drug,channel))
    plt.close()
    
    locs = np.array([0.,2.,0.01,1e-3]) # basically lower bounds for alpha,beta,s,sigma (not mu!)

    sigma_cur = np.mean(best_fits[:,-1])
    if (sigma_cur <= locs[3]):
        sigma_cur = locs[3]+0.1
    print "sigma_cur =", sigma_cur

    print st.fisk.fit(best_fits[:,0],loc=0)
    beta_cur, _, alpha_cur = st.fisk.fit(best_fits[:,0],loc=0)
    #fig = plt.figure(figsize=(10,8))
    #ax = fig.add_subplot(111)
    #x2 = np.linspace(0,5,101)
    #ax.plot(x2,st.fisk.pdf(x2,c=beta_cur,scale=alpha_cur,loc=0))
    #for a in best_fits[:,0]:
        #ax.axvline(a)
        

    x0 = np.array([0.5,0.5]) # not fitting sigma by CMA-ES
    sigma0 = 0.1
    opts = cma.CMAOptions()
    opts['seed'] = 1
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X = es.ask()
        es.tell(X, [-np.product(st.fisk.pdf(best_fits[:,0],c=x[1],scale=x[0],loc=0)) for x in X])
        #es.disp()
    res = es.result()
    print res[0]
    #ax.plot(x2,st.fisk.pdf(x2,c=res[0][1],scale=res[0][0],loc=0),color='red')
    alpha_cur, beta_cur = res[0]
    if alpha_cur <= locs[0]:
        alpha_cur = locs[0]+0.1
    if beta_cur <= locs[1]:
        beta_cur = locs[1]+0.1
    print "alpha_cur,beta_cur =", alpha_cur,beta_cur

    mu_cur, s_cur = st.logistic.fit(best_fits[:,1])
    if s_cur <= locs[2]:
        s_cur = locs[2]+0.1
    fig = plt.figure(figsize=(10,8))

    first_iteration = np.concatenate(([alpha_cur,beta_cur,mu_cur,s_cur],best_fits[:,:-1].flatten(),[sigma_cur]))
    print first_iteration
    
    elkins_hill_alphas = np.array([1.188, 1.744, 1.530, 0.930, 0.605, 1.325, 1.179, 0.979, 1.790, 1.708, 1.586, 1.469, 1.429, 1.127, 1.011, 1.318, 1.063])
    elkins_hill_betas = 1./np.array([0.0835, 0.1983, 0.2089, 0.1529, 0.1206, 0.2386, 0.2213, 0.2263, 0.1784, 0.1544, 0.2486, 0.2031, 0.2025, 0.1510, 0.1837, 0.1677, 0.0862])



    elkins_pic50_mus = np.array([5.235,5.765,6.060,5.315,5.571,7.378,7.248,5.249,6.408,5.625,7.321,6.852,6.169,6.217,5.927,7.414,4.860])
    elkins_pic50_sigmas = np.array([0.0760,0.1388,0.1459,0.2044,0.1597,0.2216,0.1856,0.1560,0.1034,0.1033,0.1914,0.1498,0.1464,0.1053,0.1342,0.1808,0.0860])

    

    # these values are from Elkins paper specifically for CaV1.2 + Verapamil
    alpha_mode = np.mean(elkins_hill_alphas)
    beta_mode = np.mean(elkins_hill_betas)
    mu_mode = np.mean(elkins_pic50_mus)
    s_mode = np.mean(elkins_pic50_sigmas)
    sigma_mode = 6.

    modes = np.array([alpha_mode, beta_mode-2., s_mode, sigma_mode])

    print "modes:", modes

    shapes = np.array([5.,2.5,2.5,5.]) # must all be greater than 1, I think

    scales = (modes-locs)/(shapes-1.)

    labels = [r'$\alpha$',r'$\beta$',r'$s$',r'$\sigma$']

    mins = [0,0,0,0]
    maxs = [8,22,2,25]

    fig = plt.figure(figsize=(20,40))
    for i in range(len(labels)):
        ax = fig.add_subplot(5,2,2*i+1)
        x_prior = np.linspace(mins[i],maxs[i],501)
        ax.plot(x_prior,st.gamma.pdf(x_prior,a=shapes[i],scale=scales[i],loc=locs[i]),label='Gamma prior')
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Probability density')
        ax.set_xlim(mins[i],maxs[i])
        ax.grid()
        if (i==1):
            ax = fig.add_subplot(5,2,2*i+2)
            x_prior = np.linspace(0,1,501)
            ax.plot(x_prior,(st.gamma.pdf(1./x_prior,a=shapes[i],scale=scales[i],loc=locs[i]))/(x_prior**2))
            ax.set_xlabel(r'$1/\beta$')
            ax.set_ylabel('Probability density')
            ax.set_xlim(0,1)
            ax.hist(1./elkins_hill_betas,normed=True,bins=10,label='Elkins data')
            ax.legend()
            ax.grid()
    ax = fig.add_subplot(5,2,9)
    x_prior = np.linspace(-5,20,501)
    pdf = np.zeros(len(x_prior))
    for i in range(len(pdf)):
        pdf[i] = np.exp(log_pic50_mu_prior(x_prior[i]))
    ax.plot(x_prior,pdf)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel('Probability density')
    ax.grid()
    fig.tight_layout()
    fig.savefig(figs_dir+'prior_distributions.png')
    plt.close()
    
    with open(chain_file,'w') as outfile:
        outfile.write("# Hill ~ log-logistic(alpha,beta), pIC50 ~ logistic(mu,s)\n")
        outfile.write("# alpha, beta, mu, s, hill_1, pic50_1, hill_2, pic50_2, ..., hill_Ne, pic50_Ne, sigma\n")



    first_cov = np.diag(0.01*np.abs(first_iteration))

    mean_estimate = np.copy(first_iteration)

    dim = len(first_iteration)
    print dim
    when_to_adapt = 100*dim

    positive_indices = [0,2,3]+range(4,4+2*num_expts+1,2)
    print positive_indices

    other_indices = [1]+range(5,5+2*num_expts,2)
    print other_indices

    theta_cur = np.copy(first_iteration)
    cov_cur = np.copy(first_cov)

    print "theta_cur =", theta_cur

    log_target_cur = log_target_distribution(experiments,theta_cur,shapes,scales,locs)

    print "initial log_target_cur =", log_target_cur
    

    loga = 0.
    acceptance = 0.
    target_acceptance = 0.25

    thinning = 5

    try:
        total_iterations = args.iterations
    except:
        total_iterations = 200000
    status_when = 10000
    saved_iterations = total_iterations/thinning+1
    pre_thin_burn = total_iterations/4
    burn = saved_iterations/4

    chain = np.zeros((saved_iterations,dim+1))
    chain[0,:] = np.copy(np.concatenate((first_iteration,[log_target_cur])))



    start = time.time()
    t = 1
    first_nonpos_cov = 0
    while t <= total_iterations:
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            theta_star = npr.multivariate_normal(theta_cur,np.exp(loga)*cov_cur)
            # Verify some things
            if len(w) > 0:
                x = w
                #print "Seems proposal covariance matrix becomes non positive etc. at iteration {}".format(t)
                if first_nonpos_cov == 0:
                    first_nonpos_cov = t
                    bad_loga = loga
                    break

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
            if i==0:
                y_label = r'$\alpha$'
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
        
    marginals_dir = figs_dir+'marginals/'
    if not os.path.exists(marginals_dir):
        os.makedirs(marginals_dir)
    for i in range(dim):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.hist(chain[burn:,i],bins=50,normed=True)
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
        plt.close()
    
    approx_total_to_plot = args.predictions
    fraction_to_plot = approx_total_to_plot / num_expts
    fig = plt.figure(figsize=(10,24))

    print "About to plot horizontal and vertical dose-resposne prediction figures."

    x = np.logspace(xmin,xmax,num_x_pts)
    try:
        approx_total_to_plot = args.predictions
    except:
        approx_total_to_plot = 1000
    fraction_to_plot = approx_total_to_plot / num_expts
    fig = plt.figure(figsize=(5,10))
    fig_2 = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(311) # vertical
    ax1_2 = fig_2.add_subplot(131) # horizontal
    ax1.set_xscale('log')
    ax1_2.set_xscale('log')
    ax1.set_ylim(0,100)
    ax1_2.set_ylim(0,100)
    ax1.set_xlim(10**xmin,10**xmax)
    ax1_2.set_xlim(10**xmin,10**xmax)
    markersize=50
    index_samples = npr.randint(burn,saved_iterations,num_expts*fraction_to_plot)
    for i in range(num_expts):
        ax1.scatter(experiments[i][:,0],experiments[i][:,1],s=markersize,color=colors[i],zorder=10,label='Experiment {}'.format(i+1))
        ax1_2.scatter(experiments[i][:,0],experiments[i][:,1],s=markersize,color=colors[i],zorder=10,label='Experiment {}'.format(i+1))
        #index_samples = npr.randint(burn,saved_iterations,fraction_to_plot)
        for index in index_samples[i*fraction_to_plot:(i+1)*fraction_to_plot]:
            prediction_curve = dr.dose_response_model(x,chain[index,4+i*2],dr.pic50_to_ic50(chain[index,5+i*2]))
            ax1.plot(x,prediction_curve,color=colors[i],alpha=0.01)
            ax1_2.plot(x,prediction_curve,color=colors[i],alpha=0.01)
    ax1.set_ylabel('% {} block'.format(channel))
    ax1_2.set_ylabel('% {} block'.format(channel))
    ax1_2.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    ax1.set_ylim(0,100)
    ax1_2.set_ylim(0,100)
    ax1.legend(loc=2,fontsize=12)
    ax1_2.legend(loc=2,fontsize=12)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = fig.add_subplot(312,sharex=ax1)
    ax2_2 = fig_2.add_subplot(132,sharey=ax1_2)
    ax2.set_xscale('log')
    ax2_2.set_xscale('log')
    for i in range(num_expts):
        ax2.scatter(experiments[i][:,0],experiments[i][:,1],s=markersize,color=colors[i],zorder=10)
        ax2_2.scatter(experiments[i][:,0],experiments[i][:,1],s=markersize,color=colors[i],zorder=10)
    #index_samples = npr.randint(burn,saved_iterations,num_expts*fraction_to_plot)
    for index in index_samples:
        hill = st.fisk.rvs(c=chain[index,1],scale=chain[index,0])
        pic50 = npr.logistic(chain[index,2],chain[index,3])
        prediction_curve = dr.dose_response_model(x,hill,dr.pic50_to_ic50(pic50))
        ax2.plot(x,prediction_curve,color='black',alpha=0.01)
        ax2_2.plot(x,prediction_curve,color='black',alpha=0.01)
    ax2.set_ylabel('% {} block'.format(channel))
    ax2_2.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2_2.get_yticklabels(), visible=False)
    ax2.set_ylim(0,100)
    ax2_2.set_ylim(0,100)
    ax2.set_xlim(10**xmin,10**xmax)
    ax2_2.set_xlim(10**xmin,10**xmax)

    ax3 = fig.add_subplot(313,sharex=ax1)
    ax3_2 = fig_2.add_subplot(133,sharey=ax1_2)
    ax3.set_xscale('log')
    ax3_2.set_xscale('log')
    for i in range(num_expts):
        ax3.scatter(experiments[i][:,0],experiments[i][:,1],s=markersize,color=colors[i],zorder=10)
        ax3_2.scatter(experiments[i][:,0],experiments[i][:,1],s=markersize,color=colors[i],zorder=10)
    #index_samples = npr.randint(burn,saved_iterations,num_expts*fraction_to_plot)
    for index in index_samples:
        hill_mode = log_logistic_mode(chain[index,0],chain[index,1])
        pic50_mode = logistic_mode(chain[index,2],chain[index,3])
        prediction_curve = dr.dose_response_model(x,hill_mode,dr.pic50_to_ic50(pic50_mode))
        ax3.plot(x,prediction_curve,color='black',alpha=0.01)
        ax3_2.plot(x,prediction_curve,color='black',alpha=0.01)
    ax3.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    ax3_2.set_xlabel(r'{} concentration ($\mu$M)'.format(drug))
    ax3.set_ylabel('% {} block'.format(channel))
    plt.setp(ax3_2.get_yticklabels(), visible=False)
    ax3.set_ylim(0,100)
    ax3_2.set_ylim(0,100)
    ax3.set_xlim(10**xmin,10**xmax)
    ax3_2.set_xlim(10**xmin,10**xmax)
    print "Finished plotting, about to save figures."

    fig.tight_layout()
    fig.savefig(figs_dir+'{}_{}_dose_response_predictions_vertical.png'.format(drug,channel))
    fig_2.tight_layout()
    fig_2.savefig(figs_dir+'{}_{}_dose_response_predictions_horizontal.png'.format(drug,channel))
    plt.close()
    print "Saved figures."

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    x_range = np.linspace(0,3,1001)
    for index in index_samples:
        ax.plot(x_range,st.fisk.pdf(x_range,c=chain[index,1],scale=chain[index,0],loc=0),alpha=0.1)
    ax.set_xlabel('Hill')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    fig.savefig(figs_dir+'sampled_hill_log_logistics.png')
    plt.close()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    x_range = np.linspace(-2,12,1001)
    for index in index_samples:
        ax.plot(x_range,st.logistic.pdf(x_range,chain[index,2],chain[index,3]),alpha=0.1)
    ax.set_xlabel('pIC50')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    fig.savefig(figs_dir+'sampled_pic50_logistics.png')
    plt.close()



    print "\n\n{} + {} complete!\n\n".format(drug,channel)


# In[29]:

for drug,channel in it.product(drugs_to_run,channels_to_run):
    run(drug,channel)
    """try:
        run(drug,channel)
    except:
        print "Failed to run {} + {}!".format(drug,channel)"""


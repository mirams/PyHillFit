import doseresponse as dr
import argparse
import numpy as np
import sys
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as st
import itertools as it

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--iterations", type=int, help="number of MCMC iterations",default=500000)
parser.add_argument("-t", "--thinning", type=int, help="how often to thin the MCMC, i.e. save every t-th iteration",default=5)
parser.add_argument("-b", "--burn-in-fraction", type=int, help="given N saved MCMC iterations, discard the first N/b as burn-in",default=4)
parser.add_argument("-a", "--all", action='store_true', help='run hierarchical MCMC on all drugs and channels', default=False)
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
num_params = dr.num_params

dr.setup(args.data_file)

drugs_to_run, channels_to_run = dr.list_drug_channel_options(args.all)


def do_mcmc(temperature):#, theta0):
    print "Starting chain"
    
    print "\nnum_params: {}\n".format(num_params)

    #theta_cur = np.copy(theta0)
    theta_cur = np.ones(num_params)
    log_target_cur = dr.log_target(responses, concs, theta_cur, num_pts, temperature, pi_bit)

    total_iterations = args.iterations
    thinning = args.thinning
    num_saved = total_iterations / thinning + 1
    burn = num_saved / args.burn_in_fraction

    chain = np.zeros((num_saved, num_params+1))
    chain[0, :] = np.concatenate((theta_cur, [log_target_cur]))

    loga = 0.
    acceptance = 0.

    mean_estimate = np.copy(theta_cur)
    cov_estimate = np.eye(num_params)

    status_when = 5000
    adapt_when = 1000*num_params

    t = 1
    s = 1
    while t <= total_iterations:
        theta_star = npr.multivariate_normal(theta_cur, np.exp(loga)*cov_estimate)
        """try:
            theta_star = npr.multivariate_normal(theta_cur, np.exp(loga)*cov_estimate)
        except Warning as e:
            print str(e)
            print "Iteration:", t
            print "temperature:", temperature
            print "theta_cur:", theta_cur
            print "loga:", loga
            print "cov_estimate:", cov_estimate
            sys.exit()"""
        log_target_star = dr.log_target(responses, concs, theta_star, num_pts, temperature, pi_bit)
        u = npr.rand()
        if np.log(u) < log_target_star - log_target_cur:
            accepted = 1
            theta_cur = theta_star
            log_target_cur = log_target_star
        else:
            accepted = 0
        acceptance = (t-1.)/t * acceptance + 1./t * accepted
        if t % thinning == 0:
            chain[t/thinning,:] = np.concatenate((theta_cur, [log_target_cur]))
        if t % status_when == 0:
            #pass
            print t/status_when, "/", total_iterations/status_when
            print "acceptance =", acceptance
        if t == adapt_when:
            mean_estimate = np.copy(theta_cur)
        if t > adapt_when:
            gamma_s = 1./(s+1.)**0.6
            temp_covariance_bit = np.array([theta_cur-mean_estimate])
            cov_estimate = (1-gamma_s) * cov_estimate + gamma_s * np.dot(np.transpose(temp_covariance_bit),temp_covariance_bit)
            mean_estimate = (1-gamma_s) * mean_estimate + gamma_s * theta_cur
            loga += gamma_s*(accepted-0.25)
            s += 1
        t += 1
    # discard burn-in before saving chain, just to save space mostly
    return chain[burn:, :]


for drug,channel in it.product(drugs_to_run, channels_to_run):

    num_expts, experiment_numbers, experiments = dr.load_crumb_data(drug, channel)

    concs = np.array([])
    responses = np.array([])
    for i in xrange(num_expts):
        concs = np.concatenate((concs, experiments[i][:, 0]))
        responses = np.concatenate((responses, experiments[i][:, 1]))

    num_pts = len(responses)
    pi_bit = dr.compute_pi_bit_of_log_likelihood(responses)

    #model = 2  #int(sys.argv[1])

    n = 2
    c = 3
    temperatures = (np.arange(n+1.)/n)**c
    print "\nDoing temperatures: {}\n".format(temperatures)

    for temperature in temperatures:
        drug,channel,chain_file,images_dir = dr.nonhierarchical_chain_file_and_figs_dir(args.model, drug, channel, temperature)
        print "chain_file:", chain_file
        chain = do_mcmc(temperature)
        np.savetxt(chain_file, chain)
        saved_iterations, num_params_plus_one = chain.shape
        
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
            axs[i][1].plot(chain[:,i],range(saved_iterations))
            axs[i][1].invert_yaxis()
            axs[i][1].set_xlabel(dr.labels[i])
            axs[i][1].set_ylabel('Saved MCMC iteration')
            axs[i][1].grid()
            figs[i].tight_layout()
            figs[i].savefig(images_dir+'{}_{}_{}_marginal.png'.format(drug,channel,dr.file_labels[i]))
            plt.close()

        # plot log-target path
        fig2 = plt.figure()
        ax3 = fig2.add_subplot(111)
        ax3.plot(range(saved_iterations), chain[:,-1])
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
                        axes[str(i)+str(j)].set_xlabel(dr.labels[j])
                    if j==0 and i>0:
                        axes[str(i)+str(j)].set_ylabel(dr.labels[i])
                        
                    plt.xticks(rotation=30)
            norm = matplotlib.colors.Normalize(vmin=colormin,vmax=colormax)
            count += 1

            
        plt.setp(hidden_labels, visible=False)
    
        matrix_fig.tight_layout()
        matrix_fig.savefig(images_dir+"{}_{}_temp_{}_scatterplot_matrix.png".format(drug,channel,temperature))
        #matrix_fig.savefig(images_dir+"{}_{}_temp_{}_scatterplot_matrix.pdf".format(drug,channel,temperature))
        plt.close()

import os
import pandas as pd
import numpy as np
import scipy.stats as st
#import warnings
#warnings.filterwarnings("error")

beta = 2.
alpha = ((beta+1.)/(beta-1.))**(1./beta)  # for mode at 1
mu = 4.
s = 2.
sigma_uniform_lower = 1e-3
sigma_uniform_upper = 50.
pic50_exp_rate = 0.2
pic50_exp_scale = 1./pic50_exp_rate
pic50_exp_lower = -3.
hill_uniform_lower = 0.
hill_uniform_upper = 10.
log_hill_uniform_const = -np.log(hill_uniform_upper-hill_uniform_lower)
log_sigma_uniform_const = -np.log(sigma_uniform_upper-sigma_uniform_lower)

sigma_shape = 5. # for sigma Gamma prior
sigma_mode = 6.
sigma_loc = 1e-3
sigma_scale = scales = (sigma_mode-sigma_loc)/(sigma_shape-1.)

n = 40
c = 3


def setup(given_file):
    global file_name, dir_name, df, drugs, channels
    file_name = given_file
    dir_name = given_file.split('/')[-1][:-4]
    df = pd.read_csv(file_name, names=['Drug','Channel','Experiment','Concentration','Inhibition'],skiprows=1)
    drugs = df.Drug.unique()
    channels = df.Channel.unique()
    

def list_drug_channel_options(args_all):
    if not args_all:
        print "\nDrugs:\n"
        for i in range(len(drugs)):
            print "{}. {}".format(i+1,drugs[i])
        drug_indices = [x-1 for x in map(int,raw_input("\nSelect drug numbers: ").split())]
        assert(0 <= len(drug_indices) <= len(drugs))
        drugs_to_run = [drugs[drug_index] for drug_index in drug_indices]
        print "\nChannels:\n"
        for i in range(len(channels)):
            print "{}. {}".format(i+1,channels[i])
        channel_indices = [x-1 for x in map(int,raw_input("\nSelect channel numbers: ").split())]
        assert(0 <= len(channel_indices) <= len(channels))
        channels_to_run = [channels[channel_index] for channel_index in channel_indices]
    else:
        drugs_to_run = drugs
        channels_to_run = channels
    return drugs_to_run, channels_to_run
    
    
def load_crumb_data(drug,channel):
    experiment_numbers = np.array(df[(df['Drug'] == drug) & (df['Channel'] == channel)].Experiment.unique())
    num_expts = max(experiment_numbers)
    experiments = []
    for expt in experiment_numbers:
        experiments.append(np.array(df[(df['Drug'] == drug) & (df['Channel'] == channel) & (df['Experiment'] == expt)][['Concentration','Inhibition']]))
    experiment_numbers -= 1
    return num_expts, experiment_numbers, experiments
    
    
def hierarchical_output_dirs_and_chain_file(drug,channel,Ne=0):
    if ('/' in drug):
        drug = drug.replace('/','_')
    if ('/' in channel):
        channel = channel.replace('/','_')
    output_dir = 'output/{}/hierarchical/{}/{}/{}_expts/'.format(dir_name,drug,channel,Ne)
    chain_dir = output_dir+'chain/'
    figs_dir = output_dir+'figures/'
    for directory in [output_dir,chain_dir,figs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    chain_file = chain_dir+'{}_{}_{}_hierarchical_chain.txt'.format(dir_name,drug,channel)
    return drug, channel, output_dir, chain_dir, figs_dir, chain_file
    
def dose_response_model(dose,hill,IC50):
    return 100. * ( 1. - 1./(1.+(1.*dose/IC50)**hill) )
    
def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)
    
def ic50_to_pic50(ic50): # IC50 in uM
    return 6-np.log10(ic50)
    
def hierarchical_posterior_predictive_cdf_files(drug,channel,Ne):
    cdf_dir = 'output/{}/hierarchical/{}/{}/{}_expts/cdfs/'.format(dir_name,drug,channel,Ne)
    if not os.path.exists(cdf_dir):
        os.makedirs(cdf_dir)
    hill_cdf_file = cdf_dir+'{}_{}_{}_posterior_predictive_hill_cdf.txt'.format(dir_name,drug,channel)
    pic50_cdf_file = cdf_dir+'{}_{}_{}_posterior_predictive_pic50_cdf.txt'.format(dir_name,drug,channel)
    return hill_cdf_file, pic50_cdf_file
    
def hierarchical_hill_and_pic50_samples_for_AP_file(drug,channel):
    output_dir = 'output/{}/hierarchical/posterior_predictive_hill_pic50_samples/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '{}_{}_{}_hill_pic50_samples.txt'.format(dir_name,drug,channel)
    return output_file
    
def hierarchical_downsampling_folder_and_file(drug,channel):
    output_dir = 'output/{}/hierarchical/downsampling/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '{}_{}_downsampled_alpha_beta_mu_s.txt'.format(drug,channel)
    return output_file

def nonhierarchical_chain_file_and_figs_dir(model, drug, channel, temperature):
    if ('/' in drug):
        drug = drug.replace('/','_')
    if ('/' in channel):
        channel = channel.replace('/','_')
    output_dir = 'output/{}/single-level/{}/{}/model_{}/temperature_{}/'.format(dir_name, drug, channel, model, temperature)
    chain_dir = output_dir+'chain/'
    images_dir = output_dir+'figures/'
    dirs = [output_dir,chain_dir,images_dir]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    chain_file = chain_dir+'{}_{}_model_{}_temp_{}_chain_single-level.txt'.format(drug, channel, model, temperature)
    return drug,channel,chain_file,images_dir
    
def alpha_mu_downsampling(drug,channel):
    output_dir = 'output/{}/hierarchical/alpha_mu_samples/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + '{}_{}_hill_pic50_samples.txt'.format(drug,channel)
    return output_file
    
def all_predictions_dir(drug,channel):
    main_dir = 'output/{}/all_prediction_curves/{}/{}/'.format(dir_name,drug,channel)
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    return main_dir

def log_hill_log_logistic_likelihood(x):
    return (beta-1.)*np.log(x) - 2.*np.log(1.+(x/alpha)**beta)


def log_pic50_logistic_likelihood(x):
    return -x/s - 2.*np.log(1 + np.exp((mu-x)/s))


def log_pic50_exponential(x):
    """Omitted constant bits like log(scale) and scale*lower"""
    if x < pic50_exp_lower:
        return -np.inf
    else:
        return -pic50_exp_rate*x
        
        
def log_sigma_uniform(x):
    if (x < sigma_uniform_lower) or (x > sigma_uniform_upper):
        return -np.inf
    else:
        return log_sigma_uniform_const      


def log_priors_model_1(params):
    pic50, sigma = params
    #if (sigma < sigma_uniform_lower) or (sigma > sigma_uniform_upper):
    #    return -np.inf
    #else:
    #    return log_pic50_exponential(pic50)
    return log_pic50_exponential(pic50) + log_gamma_prior(sigma, sigma_shape, sigma_scale, sigma_loc)


def log_priors_model_2(params):
    pic50, hill, sigma = params
    #if (sigma < sigma_uniform_lower) or (sigma > sigma_uniform_upper) or (hill < hill_uniform_lower) or (hill > hill_uniform_upper):
    #    return -np.inf
    #else:
    #    return log_pic50_exponential(pic50)
    if (hill < hill_uniform_lower) or (hill > hill_uniform_upper):
        return -np.inf
    else:
        return log_pic50_exponential(pic50) + log_gamma_prior(sigma, sigma_shape, sigma_scale, sigma_loc)


def log_target(y, where_y_0, where_y_100, where_y_other, concs, params, t, pi_bit):
    answer = log_data_likelihood(y, where_y_0, where_y_100, where_y_other, concs, params, t, pi_bit) + log_priors(params)
    return answer


def trapezium_rule(x, y):
    return 0.5 * np.sum((x[1:]-x[:-1]) * (y[1:]+y[:-1]))
    
    
def define_log_py_file(model, drug, channel):
    temp_dir = "../output/{}/{}/model_{}/log_pys/".format(drug, channel, model)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir+"{}_{}_model_{}_log_pys.txt".format(drug, channel, model)


def log_data_likelihood_model_1_capped(y, where_y_0, where_y_100, where_y_other, concs, params, t, pi_bit):
    if t == 0:
        return 0
    """
    Compute log likelihood of data.
    Note that pi_bit is a constant, so is not needed for MH, but when we're approximating log(p(y)), we need to include it.
    Of course, we could just add it on after, but I just left it there so I can reuse the exact same code.
    """
    pic50, sigma = params
    if sigma <= sigma_uniform_lower:
        # the -inf actually comes from the prior, but just to fix log of negative
        return -np.inf
    predicted_responses = dose_response_model(concs, 1, pic50_to_ic50(pic50))
    #if sigma_sq <= 0:
    #    print params
    y_0_sum = np.sum(st.norm.logcdf(0, predicted_responses[where_y_0], sigma))
    y_100_sum = np.sum(st.norm.logsf(100, predicted_responses[where_y_100], sigma))
    temp_1 = where_y_other.sum() * np.log(sigma)
    temp_2 = np.sum((y[where_y_other]-predicted_responses[where_y_other])**2/(2.*sigma**2))
    answer = t * (y_0_sum + y_100_sum - pi_bit - temp_1 - temp_2)
    if np.isnan(answer):
        print params
        print predicted_responses
    return answer


def log_data_likelihood_model_2_capped(y, where_y_0, where_y_100, where_y_other, concs, params, t, pi_bit):
    if t == 0:
        return 0
    """
    Compute log likelihood of data.
    Note that pi_bit is a constant, so is not needed for MH, but when we're approximating log(p(y)), we need to include it.
    Of course, we could just add it on after, but I just left it there so I can reuse the exact same code.
    """
    pic50, hill, sigma = params
    if sigma <= sigma_uniform_lower:
        # the -inf actually comes from the prior, but just to fix log of negative
        return -np.inf
    predicted_responses = dose_response_model(concs, hill, pic50_to_ic50(pic50))
    #if sigma_sq <= 0:
    #    print params
    y_0_sum = np.sum(st.norm.logcdf(0, predicted_responses[where_y_0], sigma))
    y_100_sum = np.sum(st.norm.logsf(100, predicted_responses[where_y_100], sigma))
    temp_1 = where_y_other.sum() * np.log(sigma)
    temp_2 = np.sum((y[where_y_other]-predicted_responses[where_y_other])**2/(2.*sigma**2))
    return t * (y_0_sum + y_100_sum - pi_bit - temp_1 - temp_2)
    
def define_model(model):
    """Choose whether to fix Hill = 1 (#1) or allow Hill to vary (#2)"""
    global log_data_likelihood, log_priors, num_params, file_labels, labels, prior_xs, prior_pdfs
    num_prior_pts = 1001
    pic50_lower = -4.
    pic50_upper = 14.
    hill_lower = 0.
    hill_upper = 6.
    if model == 1:
        num_params = 2
        log_data_likelihood = log_data_likelihood_model_1_capped
        log_priors = log_priors_model_1
        labels = [r"$pIC50$", r"$\sigma$"]
        file_labels =  ['pIC50','sigma']
        #prior_xs = [np.linspace(pic50_lower, pic50_upper, num_prior_pts),
        #            np.linspace(sigma_uniform_lower,sigma_uniform_upper,num_prior_pts)]
        prior_xs = [np.linspace(pic50_exp_lower-2, pic50_exp_lower+23, num_prior_pts),
                    np.linspace(0, 25, num_prior_pts)]
        #prior_pdfs = [st.logistic.pdf(prior_xs[0], loc=mu, scale=s),
        #              np.ones(num_prior_pts)/(1.*sigma_uniform_upper-sigma_uniform_lower)]
        #prior_pdfs = [st.expon.pdf(prior_xs[0], loc=pic50_exp_lower, scale=pic50_exp_scale),
        #              np.concatenate(([0,0],np.ones(num_prior_pts)/(1.*sigma_uniform_upper-sigma_uniform_lower),[0,0]))]
        prior_pdfs = [st.expon.pdf(prior_xs[0], loc=pic50_exp_lower, scale=pic50_exp_scale),
                      st.gamma.pdf(prior_xs[1], sigma_shape, loc=sigma_loc, scale=sigma_scale)]
    elif model == 2:
        num_params = 3
        log_data_likelihood = log_data_likelihood_model_2_capped
        log_priors = log_priors_model_2
        labels = [r"$pIC50$", r"$Hill$", r"$\sigma$"]
        file_labels =  ['pIC50','Hill','sigma']
        #prior_xs = [np.linspace(pic50_lower, pic50_upper, num_prior_pts),
        #            np.linspace(hill_lower, hill_upper, num_prior_pts),
        #            np.linspace(sigma_uniform_lower,sigma_uniform_upper,num_prior_pts)]
        prior_xs = [np.linspace(pic50_exp_lower-2, pic50_exp_lower+23, num_prior_pts),
                    np.concatenate(([hill_uniform_lower-2,hill_uniform_lower],
                                    np.linspace(hill_uniform_lower, hill_uniform_upper, num_prior_pts),
                                    [hill_uniform_upper,hill_uniform_upper+2])),
                    np.linspace(0, 25, num_prior_pts)]
        #prior_pdfs = [st.logistic.pdf(prior_xs[0],loc=mu,scale=s),
        #              st.fisk.pdf(prior_xs[1],c=beta,scale=alpha),
        #              np.ones(num_prior_pts)/(1.*sigma_uniform_upper-sigma_uniform_lower)]
        #prior_pdfs = [st.expon.pdf(prior_xs[0], loc=pic50_exp_lower, scale=pic50_exp_scale),
        #              np.concatenate(([0,0],np.ones(num_prior_pts) / (1. * hill_uniform_upper - hill_uniform_lower),[0,0])),
        #              np.concatenate(([0, 0], np.ones(num_prior_pts) / (1. * sigma_uniform_upper - sigma_uniform_lower), [0, 0]))]
        prior_pdfs = [st.expon.pdf(prior_xs[0], loc=pic50_exp_lower, scale=pic50_exp_scale),
                      np.concatenate(([0,0],np.ones(num_prior_pts) / (1. * hill_uniform_upper - hill_uniform_lower),[0,0])),
                      st.gamma.pdf(prior_xs[2], sigma_shape, loc=sigma_loc, scale=sigma_scale)]


def compute_pi_bit_of_log_likelihood(y):
    num_pts = len(y)
    return 0.5 * num_pts * np.log(2 * np.pi)
    

def log_gamma_prior(x,shape_param,scale_param,loc_params):
    # hierarchical prior for noise sigma
    if np.any(x<loc_params):
        return -np.inf
    answer = (shape_param-1)*np.log(x-loc_params) - (x-loc_params)/scale_param
    if np.any(np.isnan(answer)):
        print "NaN from dr.log_gamma_prior!"
        print "x =", x
        print "shape_param =", shape_param
        print "scale_param =", scale_param
        print "loc_param =", loc_params
        sys.exit()
    else:
        return answer
        

def samples_file(drug, channel, model, hierarchical, num_samples, temperature):
    if hierarchical:
        output_dir = 'output/{}/hierarchical/{}/{}/model_{}/temperature_{}/'.format(dir_name, drug, channel, model, temperature)
    else:
        output_dir = 'output/{}/single-level/{}/{}/model_{}/temperature_{}/'.format(dir_name, drug, channel, model, temperature)
    samples_dir = output_dir+'chain/{}_samples/'.format(num_samples)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    samples_file = samples_dir+'{}_{}_model_{}_{}_samples.txt'.format(drug, channel, model, num_samples)
    samples_png = samples_dir+'{}_{}_model_{}_{}_samples.png'.format(drug, channel, model, num_samples)
    samples_pdf = samples_dir+'{}_{}_model_{}_{}_samples.pdf'.format(drug, channel, model, num_samples)
    return samples_file, samples_png, samples_pdf

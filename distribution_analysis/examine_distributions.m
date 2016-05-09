close all
clear all

% Data files are in the format of row for each MCMC iteration with the following
% columns: [Hill, IC50, sigma, log-likelihood]

data = load('example2.txt');

Hills = data(:,1);
ic50s = data(:,2);
noise = data(:,3);

figure
subplot(3,1,1)
hist(Hills,100)
subplot(3,1,2)
hist(ic50s,100)
subplot(3,1,3)
hist(noise,100)

figure
scatter(ic50s, Hills)

%% Fit a probability distribution
prob_distbn = fitdist(ic50s,'Loglogistic');

x = [min(ic50s): 0.2: max(ic50s)];
matlab_pdf = pdf(prob_distbn,x);
matlab_cdf = cdf(prob_distbn,x);

% Check we can get parameters out that we can work with too
matlab_params = prob_distbn.ParameterValues;

% Matlab works with the mu and sigma from the Logistic as if
% Ln(x) ~ logistic(mu, sigma)
% i.e. the parameters for the pIC50 distribution,
% so we need the following scalings:

alpha = exp(matlab_params(1)); % alpha = exp(mu)
beta = 1.0./(matlab_params(2)); % beta = 1/sigma

% These are the Wikipedia PDF and CDF expressions
our_pdf = ((beta/alpha).*(x/alpha).^(beta-1.0))./((1.0+(x/alpha).^beta).^2.0);
our_cdf = (x.^beta)./(alpha.^beta + x.^beta);

figure
subplot(1,2,1)
plot(x,matlab_pdf,'b-')
hold on
plot(x,our_pdf,'r--')
xlabel('IC50')
ylabel('PDF')
subplot(1,2,2)
plot(x,matlab_cdf,'b-')
hold on
plot(x,our_cdf,'r--')
xlabel('IC50')
ylabel('CDF')
legend('Matlab','Ours')



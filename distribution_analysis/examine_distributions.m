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

dfittool
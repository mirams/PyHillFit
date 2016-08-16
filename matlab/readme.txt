A script to run Matlab's built in nonlinear mixed effects method on the Crumb data.

In Matlab, define the variables drug and channel as strings, and call the function run_NLME(drug,channel). e.g.

*******

drug = 'Amiodarone';
channel = 'hERG';

run_NLME(drug,channel)

*******

Be sure to spell the drug and channel as they appear in python_input_data.csv, including any weird punctuation etc.

Also included in this directory are a few Matlab function files for the dose-response model and one for loading the data.
Make sure python_input_data.csv is in the working directory!

Matlab page on NLME: http://uk.mathworks.com/help/stats/mixed-effects-models.html

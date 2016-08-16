A script to run Matlab's built in nonlinear mixed effects method on the Crumb data.

In Matlab, just call run_NLME and select drug and channel from the list provided.

Also included in this directory are a few Matlab function files for the dose-response model and one for loading the data.
Make sure python_input_data.csv is in the working directory!

Matlab page on NLME: http://uk.mathworks.com/help/stats/mixed-effects-models.html

Also, the Matlab nlmefit function only works when each experiment has the same number of observations in each experiment.
There's a check in run_NLME to make sure this is the case, but if not then it will just quit with a failed assertion and it should give a little error message.
I don't currently know if there's a way to overcome this or if it's a property of NLME in general.

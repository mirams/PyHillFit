function synthetic_NLME(num_expts,num_params)

seed = 1
rng(seed)

log_min_conc = -3
log_max_conc = 1
num_concs = 4

sigma = 1

responses = zeros(num_expts,num_concs);

true_hill = 1.5;
hill_var = 0.1;
initial_hill = 1;

%true_ic50 = 1;
true_pic50 = 6; % so that IC50 = 1
pic50_var = 0.01;
initial_pic50 = 4;



if num_params==2
    BETA = [true_hill,true_pic50] % "true" [Hill,pIC50]
    beta0 = [initial_hill,initial_pic50] % initial guess
    PSI = diag([hill_var,pic50_var]) % covariance matrix
    model = @(PHI,c) dose_response_model(PHI(:,1),PHI(:,2),c)
elseif num_params==1 % assuming Hill = 1 and only fitting pIC50; not sure if we should keep Hill = 1 for synthetic data generating, however...
    BETA = true_pic50 % pIC50
    beta0 = initial_pic50 % initial guess
    PSI = pic50_var % covariance matrix
    model = @(PHI,c) dose_response_model(ones(size(PHI)),PHI,c)
end

b_is = mvnrnd(zeros(1,num_params),PSI,num_expts)

PHI_is = bsxfun(@plus,BETA,b_is)

concs = logspace(log_min_conc,log_max_conc,num_concs);

if num_params==2
    Hills = PHI_is(:,1)
    pIC50s = PHI_is(:,2)
elseif num_params==1
    Hills = ones(num_expts,1)
    pIC50s = PHI_is
end

for i=1:num_concs
    responses(:,i) = dose_response_model(Hills,pIC50s,concs(i));
end

responses = responses + sigma^2 * randn(size(responses));
responses(responses<0) = 0;
responses(responses>100) = 100

CONC = repmat(concs,num_expts,1)
NUMS = repmat((1:num_expts)',size(concs))

h = plot(concs,responses,'o','LineWidth',2);
xlabel('Concentration (\muM)')
ylabel('% block')
set(gca,'xscale','log')
%xlim([10^xmin 10^xmax])
legend([repmat('Expt ',num_expts,1),num2str((1:num_expts)')],...
       'Location','NW')
grid on
hold on

[beta1,PSI1,stats1,b1] = nlmefit(CONC(:),responses(:),NUMS(:),...
                              [],model,beta0)
variances = diag(PSI1);


PHI = repmat(beta1,1,num_expts) + ...          % Fixed effects
             b1;    % Random effects

cplot = logspace(log_min_conc-1,log_max_conc+1);
xlim([10^(log_min_conc-1),10^(log_max_conc+1)])
for i = 1:num_expts
    if num_params==2
        fitted_model=@(c) dose_response_model(PHI(1,i),PHI(2,i),c);
    elseif num_params==1
        fitted_model=@(c) dose_response_model(1,PHI(i),c);
    end
    plot(cplot,fitted_model(cplot),'Color',h(i).Color, ...
         'LineWidth',2)
end

end

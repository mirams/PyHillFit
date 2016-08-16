function run_NLME(drug,channel)

    [expts,concs,responses] = load_drug_channel(drug,channel);

    num_expts = max(expts);
    assert(num_expts>1,...
           'Only one experiment number for this drug and channel combination!');

    num_first_expt_pts = sum(expts==1);
    for i = 2:num_expts
        assert(sum(expts==i)==num_first_expt_pts,...
               'There must be the same number of data points in each experiment for NLME to work!');
    end

    responses = reshape(responses,[num_expts,num_first_expt_pts])

    xmin = -4;
    xmax = 4;
    conc = unique(concs)'
    h = plot(conc,responses,'o','LineWidth',2);
    xlabel('Concentration (\muM)')
    ylabel('% block')
    set(gca,'xscale','log')
    xlim([10^xmin 10^xmax])
    legend([repmat('Expt ',num_expts,1),num2str((1:num_expts)')],...
           'Location','NW')
    grid on
    hold on

    model = @(PHI,c) dose_response_model(PHI(:,1),PHI(:,2),c)

    CONC = repmat(conc,num_expts,1);
    NUMS = repmat((1:num_expts)',size(conc));

    beta0 = [1,2] % Initial guess for [Hill,pIC50]

    [beta1,PSI1,stats1,b1] = nlmefit(CONC(:),responses(:),NUMS(:),...
                                  [],model,beta0)


    PHI = repmat(beta1,1,num_expts) + ...          % Fixed effects
          b1;    % Random effects

    cplot = logspace(xmin,xmax);
    for i = 1:num_expts
      fitted_model=@(c) dose_response_model(PHI(1,i),PHI(2,i),c);
      plot(cplot,fitted_model(cplot),'Color',h(i).Color, ...
	       'LineWidth',2)
    end
    
end

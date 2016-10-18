close all
clear all

% Get the list of drugs we are using
d = importdata('drug_list.txt',' ',1);
drug_names = d.textdata(2:end,1);
risks = d.data(:,2);
colours = [1 0 0      % Red
           1 0.5 0    % Orange
           0 0.8 0];  % Green
       
colour_order = distinguishable_colors(length(drug_names));
       
for drug_idx = 1:length(drug_names)
    
    % Open the data file for each drug
    filename = [getenv('CHASTE_TEST_OUTPUT') '/CrumbDataStudy_model_6_num_params_2_num_samples_501/' drug_names{drug_idx} '_apd90_results.dat']
    d = importdata(filename,'\t',1);
    
    % If it ran OK there will be a structure d.data
    if (~isfield(d,'data'))
        continue
    end
    
    % Plot histogram
    figure(1)
    subplot(2,1,1)
    h(drug_idx) = histogram(d.data(:,3),20);
    hold all
    set(h(drug_idx),'FaceColor',colours(risks(drug_idx),:)) 
    
    subplot(2,1,2)
    h2(drug_idx) = histogram(d.data(:,3),20);
    hold all
    [max_val, i] = max(h2(drug_idx).Values);
    x = h2(drug_idx).BinEdges(i);
    y = 1.05*max_val;
    text_obj = text(x,y,drug_names{drug_idx});
    set(text_obj, 'rotation', 45)
    set(h2(drug_idx),'FaceColor',colour_order(drug_idx,:)) 
    
    values = h(drug_idx).Values;
    tmp = h(drug_idx).BinEdges;    
    for i=1:length(values)
        midpoints(i) = mean(tmp([i i+1]));
    end    
    p = findobj(gca,'Type','patch');
    set(p,'facealpha',0.5)        
    
    figure(2)
    h2(drug_idx) = plot(midpoints,values,'-','Color',colours(risks(drug_idx),:),'LineWidth',3);
    hold on
        
end

order = [1 3 2];

figure(1)
subplot(2,1,1)
xlabel('APD90 (ms)')
ylabel('Frequency')
% Grag three drugs and put legend in a sensible order (first three happen
% to be one of each risk, handily).
legend(h(order),{'High Risk','Intermediate Risk','Low Risk'},'Location','east')
set(gca,'FontSize',16)

subplot(2,1,2)
xlabel('APD90 (ms)')
ylabel('Frequency')
% Grag three drugs and put legend in a sensible order (first three happen
% to be one of each risk, handily).
legend(drug_names,'Location','eastoutside')

figure(2)
xlabel('APD90 (ms)')
% Grag three drugs and put legend in a sensible order (first three happen
% to be one of each risk, handily).
legend(h2(order),{'High Risk','Intermediate Risk','Low Risk'})


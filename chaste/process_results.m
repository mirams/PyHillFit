close all
clear all

% Get the list of drugs we are using
d = importdata('drug_list.txt',' ',1);
drug_names = d.textdata(2:end,1);
       
colour_order = distinguishable_colors(length(drug_names));
       
for drug_idx = 1:length(drug_names)
    
    % Open the data file for each drug
    
    % Use this to load data from the repo
    filename = ['results/' drug_names{drug_idx} '_apd90_results_num_params_2.dat']
    
    % Or uncomment this to load data that Chaste program generated.
    %filename = [getenv('CHASTE_TEST_OUTPUT') '/CrumbDataStudy_num_params_2_num_samples_500/' drug_names{drug_idx} '_apd90_results_num_params_2.dat']
    d = importdata(filename,'\t',1);
    
    % If it ran OK there will be a structure d.data
    if (~isfield(d,'data'))
        continue
    end
    
    % Plot histogram
    figure(1)
    h(drug_idx) = histogram(d.data(:,3),20);
    hold all
    if strcmp(drug_names{drug_idx},'Quinidine')
        sort(d.data(:,3))
    end
    
    [max_val, i] = max(h(drug_idx).Values);
    x = h(drug_idx).BinEdges(i);
    y = 1.05*max_val;
    %text_obj = text(x,y,drug_names{drug_idx});
    %set(text_obj, 'rotation', 45)
    set(h(drug_idx),'FaceColor',colour_order(drug_idx,:)) 
    
    values = h(drug_idx).Values;
    tmp = h(drug_idx).BinEdges;    
    for i=1:length(values)
        midpoints(i) = mean(tmp([i i+1]));
    end    
    p = findobj(gca,'Type','patch');
    set(p,'facealpha',0.5)
end

figure(1)
xlabel('APD90 (ms)')
ylabel('Frequency')
% Grag three drugs and put legend in a sensible order (first three happen
% to be one of each risk, handily).
legend(drug_names,'Location','eastoutside')



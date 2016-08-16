function [expts,concs,responses] = load_drug_channel(drug,channel)

    table = readtable('python_input_data.csv',...
            'Delimiter',',',...
            'ReadVariableNames',false,...
            'Format','%s%s%u%f%f');
    table.Properties.VariableNames = {'Drug', 'Channel', 'Experiment', 'Concentration', 'Inhibition'};

    rows = strcmp(table.Drug,drug) & strcmp(table.Channel,channel);

    expt_conc_response = table(rows,3:end);

    expts = expt_conc_response.Experiment;
    concs = expt_conc_response.Concentration;
    responses = expt_conc_response.Inhibition;

end

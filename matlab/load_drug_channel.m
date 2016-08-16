function [expts,concs,responses] = load_drug_channel()

    table = readtable('python_input_data.csv',...
            'Delimiter',',',...
            'ReadVariableNames',false,...
            'Format','%s%s%u%f%f');
    table.Properties.VariableNames = {'Drug', 'Channel', 'Experiment', 'Concentration', 'Inhibition'};

    drugs = unique(table.Drug);
    channels = unique(table.Channel);

    choose_drug_msg = sprintf('\nChoose a drug:\n');
    disp(choose_drug_msg)
    for i = 1:length(drugs)
        S = sprintf('%u. %s',i,drugs{i});
        disp(S)
    end
    drug_input = input('\n');

    choose_channel_msg = sprintf('\nChoose a channel:\n');
    disp(choose_channel_msg)
    for i = 1:length(channels)
        S = sprintf('%u. %s',i,channels{i});
        disp(S)
    end
    channel_input = input('\n');

    drug = drugs{drug_input}
    channel = channels{channel_input}

    rows = strcmp(table.Drug,drug) & strcmp(table.Channel,channel);

    expt_conc_response = table(rows,3:end);

    expts = expt_conc_response.Experiment;
    concs = expt_conc_response.Concentration;
    responses = expt_conc_response.Inhibition;

end

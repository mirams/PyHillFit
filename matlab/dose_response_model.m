function [response] = dose_response_model(hill,pic50,conc)

    ic50 = 10.^(6-pic50);

    response = 100 ./ ( 1 + (ic50./conc).^hill );

end

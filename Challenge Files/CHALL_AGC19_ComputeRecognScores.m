
function FR_score = CHALL_AGC19_ComputeRecognScores(...
    AutomSTR, AGC19_Challenge3_STR)
%
% Compute face recognition score
%
% FR_score = CHALL_AGC19_ComputeRecognScores(...
%     AutomSTR, AGC19_Challenge2_STR)
%
% INPUTS
%   - AutomSTR: A structure with the results of the automatic face
%   recognition algorithm, stored as an integer in the 'id' field
%
%   - AGC19_Challenge2_STR: The ground truth structure (e.g.
%   AGC19_Challenge2_TRAINING or AGC19_Challenge2_TEST).
%
% OUTPUt
%   - FR_score:     The final recognition score
% 
% --------------------------------------------------------------------
% AGC Challenge 2019 
% Universitat Pompeu Fabra
%

auto_ids = zeros(1, length( AutomSTR ));
true_ids = zeros(1, length( AGC19_Challenge3_STR ));

if length( auto_ids ) ~= length( true_ids )
    error('Inputs must be of the same length');
end

for j = 1 : length( auto_ids )
    auto_ids(j) = AutomSTR(j).id;
    true_ids(j) = AGC19_Challenge3_STR(j).id;    
end

f_beta = 1;
nTP = length( find(...
    (auto_ids == true_ids) & (true_ids ~= -1) ));
nTN = length( find(...
    (auto_ids == -1) & (true_ids == -1) ));
nFP = length( find(...
    (auto_ids ~= true_ids) & auto_ids ~= -1 ));
nFN = length( find(...
    (auto_ids == -1) & true_ids ~= -1));

fprintf("TP: %d, TN: %d, FP: %d, FN: %d\n", nTP, nTN, nFP, nFN);
FR_score = (1+f_beta^2)*nTP /(...
    (1+f_beta^2)*nTP + f_beta^2 * nFN + nFP);
end



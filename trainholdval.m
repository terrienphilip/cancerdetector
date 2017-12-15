function [ trainInd, holdInd, valInd ] = trainholdval(X, trainNum, holdNum)
%trainholdval Returns a set of trainInd, holdInd, valInd for data set.
% 
%   [ trainInd, holdInd, valInd ] = trainholdval(X, trainNum, holdNum)
% 
%   The number of training and holdout sets are put into the function,
%   which then returns randomly generated exclusive indices for each set.
%   The remaining indices not specified by the number of train or hold our
%   are returned as validation set indices The function assumes that
%   different samples are in the rows and features are in the columns.

ind = 1:size(X,1);

% Find training samples
trainInd = randperm(length(ind), trainNum);

% Find hold out samples from remaining indices
indRemain = ind(~ismember(ind, trainInd));
tempInd = randperm(length(indRemain), holdNum);
holdInd = indRemain(tempInd);

% Find validation samples from remaining indices
indRemain2 = indRemain;
valInd = indRemain2(~ismember(indRemain, holdInd));
end


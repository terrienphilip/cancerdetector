function [ ypredict, yValTrans ] = logclassify( upThresh, lowThresh, yhatLog, yVal )
%logclassify Binary classification based on set probability thresholds.
% 
%   [ ypredict, yValTrans ] = logclassify( upThresh, lowThresh, yhatLog,
%   yVal )
% 
% The user passes in a upper threshold and lower threshold, which specify
% the probabilities that must be met for a "confident" classification. The
% full range of test samples are taken in and only the classifications that
% are outside the specified threshold range are returned as ypredict. yVal
% is then transformed to yValTrans to match the sample indices that were
% returned in ypredict.

% create placeholder for ypredict
ypredict = zeros(length(yhatLog),1);

% classify predictions outside of threshold as 0 or 1
ypredict(yhatLog > upThresh) = 1;
ypredict(yhatLog < lowThresh) = 0;

% classify predictions within threshold as 0.5
ypredict((lowThresh <= yhatLog) & (yhatLog <= upThresh)) = 0.5;

% remove indices for 0.5 or "not confident" predictions
yValTrans = yVal(ypredict ~= 0.5);
ypredict = ypredict(ypredict ~= 0.5);

end


function [ TP, TN, FP, FN ] = analysis( y, yhat, yes, no )
%analysis Finds true positives, true negatives, false positives, and false
%negatives given yhat and y. Specify yes as integer for positive
%classification and no for integer as negative classification
% 
%     [ TP TN FP FN ] = analysis( y, yhat, yes, no )

trueValues = find(yhat == y);
TP = length(find(yhat(trueValues) == yes));
TN = length(find(yhat(trueValues) == no));

falseValues = find(yhat ~= y);
FP = length(find(yhat(falseValues) == yes));
FN = length(find(yhat(falseValues) == no));

end
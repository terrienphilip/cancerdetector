%% ECE 532 - Final Project
%% Alex Scharp and Philip Terrien

%% Load Data
clc; clear all; close all;

% -1 for Benign and +1 for Malignant
load('data.csv')

y = data(:,2);
X = data(:,3:end);

%% Pre-processing Data
% normalize data
X = (X - mean(X,1))./std(X,0,1);

% add a bias term
X = [ones(size(X,1),1), X];

%% Linear Regression w/ Tikhonov Regularization by Stochastic Gradient Descent
tow = 1e-4;       % descent step size
lamda = 0.1;    % regularization factor
epsilon = 1e-6; % threshold for convergance
aveIter = 1000;    % number of iterations to average results

% placeholders for average accuracy, sensitivity, and specificity
accurLin = zeros(1,aveIter);
senseLin = zeros(1,aveIter);
specLin = zeros(1,aveIter);
pprLin = zeros(1,aveIter);
nprLin = zeros(1,aveIter);

for aveInd = 1:aveIter
    % randomly select training and validation set
    [train, hold, val] = trainholdval(X, 400, 0);
    
    % initialize weights
    wprev = zeros(size(X,2),1);
    
    % update weights by SGD
    for i = 1:10000
        % select random index, ik, for SGD
        ind = randperm(length(train),1);
        ik = train(ind);
        
        grad = -2*(y(ik) - X(ik,:) * wprev) * X(ik,:)' + 2*lamda*wprev/length(train);
        wnxt = wprev - (tow/2) * grad;
        
        % check for convergence
        if norm(wnxt-wprev) < epsilon
            break
        else
            wprev=wnxt;
        end
    end
    
    % check and display warning if solution never converged
    if wnxt == wprev
        'Warning: Gradient did not converge'
    end
    
    w = wnxt;
    
    % make prediction on validation data
    yhatLin = sign(X(val,:) * w);
    
    % return the TP, TN, FP, and FN based on results
    [TP TN FP FN] = analysis(y(val), yhatLin, 1, -1);
    
    % calculate stats for particular set
    accurLin(aveInd) = (TP + TN) / (TP + TN + FP + FN);
    senseLin(aveInd) = TP / (TP + FN);
    specLin(aveInd) = TN / (TN + FP);
    pprLin(aveInd) = TP / (TP + FP);
    nprLin(aveInd) = TN / (TN + FN);
end

% Linear Regression stat averages
['Accuracy Linear = ' num2str(mean(accurLin))]
['Sensitivity Linear = ' num2str(mean(senseLin))]
['Specicity Linear = ' num2str(mean(specLin))]
['PPR Linear = ' num2str(mean(pprLin))]
['NPR Linear = ' num2str(mean(nprLin))]

%% Re-proccess Output
% change y values for benign tumor from -1 to 0
y(y==-1) = 0;

%% Logistic Regression Using Stochastic Gradient Descent
%% Learn the Best Thresholds
tow = 1e-4;       % descent step size
epsilon = 1e-7;   % threshold for convergance
aveIter = 1000;    % number of iterations to average results

% place holders for stats and percent of confident classifications of lower
% threshold
accurLow = zeros(aveIter, length(0:0.001:0.5));
npr = zeros(aveIter, length(0:0.001:0.5));
classPercLow = zeros(aveIter, length(0:0.001:0.5));

% place holders for stats and percent of confident classifications of upper
% threshold
accurUp = zeros(aveIter, length(0.5:0.001:1));
ppr = zeros(aveIter, length(0.5:0.001:1));
classPercUp = zeros(aveIter, length(0.5:0.001:1));

for aveInd = 1:aveIter
    % randomly select training and validation set
    [train, hold, val] = trainholdval(X, 400, 0);
    
    % initialize weights
    wprev = zeros(size(X,2),1);
    
    % update weights by SGD (increased iterations from 10000 to 100000)
    for i = 1:100000
        % select random index, ik, for SGD
        ind = randperm(length(train),1);
        ik = train(ind);
        
        grad = (logsig(X(ik,:)*wprev) - y(ik)) * X(ik,:);
        wnxt = wprev - tow*grad';
        
        % check for convergence
        if norm(wnxt-wprev) < epsilon
            break
        else
            wprev=wnxt;
        end
    end
    
    % check and display warning if solution never converged
    if wnxt == wprev
        'Warning: Gradient did not converge'
    end
    w = wnxt;
    
    yhatLog = logsig(X(val,:)*w);
    
    % Learn the Best Lower Threshold Bound
    upThresh = 1;
    ind = 1;
    for lowThresh = 0 : 0.001 : 0.5
        yVal = y(val);
        [ypredict, yVal] = logclassify(upThresh, lowThresh, yhatLog, yVal);
        
        % return the TP, TN, FP, and FN based on results
        [TP TN FP FN] = analysis(yVal, ypredict, 1, 0);
        
        accurLow(aveInd, ind) = (TP + TN) / (TP + TN + FP + FN);
        npr(aveInd, ind) = TN / (TN + FN);
        
        classPercLow(aveInd, ind) = length(ypredict)/nnz(y(val)==0);
        
        ind = ind + 1;
    end
    
    % Learn the Best Upper Threshold Bound
    lowThresh = 0;
    ind = 1;
    for upThresh = 0.5 : 0.001 : 1
        yVal = y(val);
        [ypredict, yVal] = logclassify(upThresh, lowThresh, yhatLog, yVal);
        
        % return the TP, TN, FP, and FN based on results
        [TP TN FP FN] = analysis(yVal, ypredict, 1, 0);
        
        accurUp(aveInd, ind) = (TP + TN) / (TP + TN + FP + FN);
        ppr(aveInd, ind) = TP / (TP + FP);
        
        classPercUp(aveInd, ind) = length(ypredict)/nnz(y(val)==1);
        
        ind = ind + 1;
    end
end

% Plot the negative predictivity rate vs. percent of confident
% classifications
aveNPR = mean(npr,1);
aveNPR(isnan(aveNPR)) = 1;
aveNPR = aveNPR*100;
aveClassPercLow = mean(classPercLow,1)*100;
figure();
plot(aveClassPercLow, aveNPR);
set(gca,'FontSize',20);
xlabel('% Classified from Benign Classifications in Validation Set');
xlim([0,100]);
ylim([96,100]);
ylabel('Negative Predictive Rate');
title('NPR vs. % Classified from Benign Classification in Validation Set');

% Plot the positive predictive rate vs. number of confident
% classifications
avePPR = mean(ppr,1);
avePPR(isnan(avePPR)) = 1;
avePPR = avePPR*100;
aveClassPercUp = mean(classPercUp,1)*100;
figure();
plot(aveClassPercUp, avePPR);
set(gca,'FontSize',20);
xlabel('% Classified from Malignant Classifications in Validation Set');
xlim([0,100]);
ylim([96,100]);
ylabel('Positive Predictive Rate');
title('PPR vs. % Classified from Malignant Classifications in Validation Set');

% Find the best thresholds where classifications are max and 99% predictive
% rate is passed
upThreshRange = 0.5:0.001:1;
lowThreshRange = 0:0.001:0.5;

% Best index is at the lowest threshold which is the first index
bestUpInd = find(avePPR >= 0.99);
bestUpInd = bestUpInd(1);
bestUpThresh = upThreshRange(bestUpInd);

% Best index is at the highest threshold which is the last index
bestLowInd = find(aveNPR >= 0.99);
bestLowInd = bestLowInd(end);
bestLowThresh = lowThreshRange(bestLowInd);

%% Use Best Thresholds to Perform Statistics for 1000 Classification Models
tow = 1e-4;       % descent step size
epsilon = 1e-7;   % threshold for convergance
aveIter = 1000;    % number of iterations to average results

accurLog = zeros(aveIter, 1);
pprLog = zeros(aveIter, 1);
nprLog = zeros(aveIter, 1);
senseLog = zeros(aveIter, 1);
specLog = zeros(aveIter, 1);
classPerc = zeros(aveIter, 1);

for aveInd = 1:aveIter
    % randomly select training and validation set
    [train, hold, val] = trainholdval(X, 400, 0);
    
    % initialize weights
    wprev = zeros(size(X,2),1);
    
    % update weights by SGD (increased iterations from 10000 to 100000)
    for i = 1:100000
        % select random index, ik, for SGD
        ind = randperm(length(train),1);
        ik = train(ind);
        
        grad = (logsig(X(ik,:)*wprev) - y(ik)) * X(ik,:);
        wnxt = wprev - tow*grad';
        
        % check for convergence
        if norm(wnxt-wprev) < epsilon
            break
        else
            wprev=wnxt;
        end
    end
    
    % check and display warning if solution never converged
    if wnxt == wprev
        'Warning: Gradient did not converge'
    end
    w = wnxt;
    
    yhatLog = logsig(X(val,:)*w);
    
    yVal = y(val);
    [ypredict, yVal] = logclassify(0.5, 0.5, yhatLog, yVal);
    
    % return the TP, TN, FP, and FN based on results
    [TP TN FP FN] = analysis(yVal, ypredict, 1, 0);
    
    accurLog(aveInd) = (TP + TN) / (TP + TN + FP + FN);
    nprLog(aveInd) = TN / (TN + FN);
    pprLog(aveInd) = TP / (TP + FP);
    senseLog(aveInd) = TP / (TP + FN);
    specLog(aveInd) = TN / (TN + FP);
    classPerc(aveInd) = length(ypredict)/length(y(val));
    
end

% Logistic Regression for best thresholds
['Accuracy Logistic = ' num2str(mean(accurLog))]
['Sensitivity Logistic = ' num2str(mean(senseLog))]
['Specificity Logistic = ' num2str(mean(specLog))]
['PPR Logistic = ' num2str(mean(pprLog))]
['NPR Logistic = ' num2str(mean(nprLog))]
['Percent Classified of the Validation Set = ' num2str(mean(classPerc))]

figure();
plot(sort(yhatLog))
xlabel('YHat', 'FontSize', 20);
ylabel('Probability', 'FontSize', 20);
title('Sigmoid Graph for YHat', 'FontSize', 20);

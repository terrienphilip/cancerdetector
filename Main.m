%% ECE 532 - Final Project
%% Alex Scharp and Philip Terrien

%% Load Data
% -1 for Benign and +1 for Malignant
clc; clear all; close all;

load('data.csv')

y = data(:,2);
X = data(:,3:end);

%% Pre-processing Data
% normalize data
X = (X - mean(X,1))./std(X,0,1);

% add a bias term
X = [ones(size(X,1),1), X];

%% Linear Regression w/ Tikhonov Regularization by Stochastic Gradient Descent

% placeholders for average accuracy, sensitivity, and specificity
accurLin = zeros(1,100);
senseLin = zeros(1,100);
specLin = zeros(1,100);

for aveInd = 1:100
[train, hold, val] = trainholdval(X, 480, 0);
tow = 1e-4;       % descent step size
lamda = 0.1;    % regularization factor
epsilon = 1e-6; % threshold for convergance

% initialize weights
wprev = zeros(size(X,2),1);

% update weights by SGD
for i = 1:10000
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

if wnxt == wprev
    'Warning: Gradient did not converge'
end
w = wnxt;

yhatLin = sign(X(val,:) * w);
[TP TN FP FN] = analysis(y(val), yhatLin, 1, -1);

accurLin(aveInd) = (TP + TN) / (TP + TN + FP + FN);
senseLin(aveInd) = TP / (TP + FN);
specLin(aveInd) = TN / (TN + FP);
end

mean(accurLin)
std(accurLin)

%% Re-proccess Output
% change y values of -1 to 0
y(y==-1) = 0;

%% Logistic Regression Using Stochastic Gradient Descent

% placeholders for average accuracy, sensitivity, and specificity
accurLog = zeros(1,100);
senseLog = zeros(1,100);
specLog = zeros(1,100);

for aveInd = 1:100
    [train, hold, val] = trainholdval(X, 400, 0);
    tow = 1e-4;       % descent step size
    epsilon = 1e-7; % threshold for convergance
    
    % initialize weights
    wprev = zeros(size(X,2),1);
    
    % update weights by SGD
    for i = 1:100000
        ind = randperm(length(train),1);
        
        ik = train(ind);
        grad = (1/(1+exp(-X(ik,:)*wprev)) - y(ik)) * X(ik,:);

        wnxt = wprev - tow*grad';
        
        % check for convergence
        if norm(wnxt-wprev) < epsilon
            break
        else
            wprev=wnxt;
        end
    end
    
    if wnxt == wprev
        'Warning: Gradient did not converge'
    end
    w = wnxt;

    yhatLog = 1 ./ (1 + exp(-X(val,:)*w));
    
    upThresh = 0.7;
    lowThresh = 0.3;
    
    ypredict = zeros(length(yhatLog),1);
    ypredict(yhatLog > upThresh) = 1;
    ypredict(yhatLog < lowThresh) = 0;
    ypredict((lowThresh <= yhatLog) & (yhatLog <= upThresh)) = 0.5;
    
    yVal = y(val);
    yVal = yVal(ypredict ~= 0.5);
    ypredict = ypredict(ypredict ~= 0.5);
    
    [TP TN FP FN] = analysis(yVal, ypredict, 1, 0);
    
    accurLog(aveInd) = (TP + TN) / (TP + TN + FP + FN);
    senseLog(aveInd) = TP / (TP + FN);
    specLog(aveInd) = TN / (TN + FP);
end

mean(accurLog)
plot(sort(yhatLog))

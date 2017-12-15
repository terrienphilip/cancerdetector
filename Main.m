%% ECE 532 - Final Project
%% Alex Scharp and Philip Terrien
clc; clear all; close all;

load('data.csv')

y = data(:,2);
X = data(:,3:end);

[train, hold, val] = trainholdval(X, 300, 0);

%% Linear Regression Using All Features

wLin = (X(train,:)' * X(train,:)) \ (X(train,:)' * y(train));
yhatLin = X(val,:) * wLin;

[TP TN FP FN] = analysis(y(val), yhatLin);

accurLin = (TP + TN) / (TP + TN + FP + FN);

%% Linear Regression Using PCA First Feature


%% Linear Regression Using ICA
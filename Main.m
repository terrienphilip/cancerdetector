%% ECE 532 - Final Project
%% Alex Scharp and Philip Terrien
clc; clear all; close all;

load('data.csv')

y = data(:,2);
X = data(:,3:end);

[train, hold, val] = trainholdval(X, 300, 100);

[U,S,V] = svd(X,'econ');




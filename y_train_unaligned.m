%% This file loads the .mat files from the unaligned X_train and generates the surface coordinates with snakes

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library for NIFTI_TOOLBOX
addpath(genpath('NIFTI_TOOLBOX'));

%% Define MyPath to our local Raw Data
MyPath = 'Data'; 
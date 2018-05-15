%% Load NIFTI Files

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.

%% Add library for NIFTI_TOOLBOX
addpath(genpath('NIFTI_TOOLBOX'));

%% Define MyPath to our local Raw Data
MyPath = 'Data'; 

%% Import Nifti Images
B = load_nii(fullfile([MyPath, '\patient001\patient001_frame12.nii.gz']));

%% view
view_nii(B);

%% view single image
figure;
% frame 15, at depth 5
image(A.img(:,:,5,15)/8); 
colormap gray;
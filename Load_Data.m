%% Load NIFTI Files

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library to create NPY outputs
addpath(genpath('NIFTI_TOOLBOX'));

%% Define MyPath to our local Raw Data
MyPath = 'C:\Users\dario\Dropbox\3D Vision\Data';
% MyPath = ;  % Define Cornels working folder
 MyPath = 'C:\Users\Saphi\Desktop\8th Semester\3D Vision\project\training';  % Define Michelles working folder

%% Import Nifti Images
A = load_nii(fullfile([MyPath, '\patient001\patient001_4d.nii.gz']));
B = load_nii(fullfile([MyPath, '\patient001\patient001_frame12.nii.gz']));

%% view
view_nii(A)
%% Alignment of images towards centre of image. 

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.


%% HARDCODED INPUTS
LoadPath_GT = 'GT_Unaligned\';
LoadPath_MRI = 'MRI_Unaligned\';
StorePath_GT = 'GT_Aligned\';
StorePath_MRI = 'MRI_Aligned\';
filePattern_GT = fullfile(LoadPath_GT, '*.png');
filePattern_MRI = fullfile(LoadPath_MRI, '*.mat');

%% Add library for NIFTI_TOOLBOX
addpath(genpath('NIFTI_TOOLBOX'));

%% Define MyPath to our local Raw Data
if ~isdir(LoadPath_GT)
	errorMessage = sprintf('Error: The following folder does not exist:\n%s', LoadPath_GT);
	uiwait(warndlg(errorMessage));
	return;
end

if ~isdir(LoadPath_MRI)
	errorMessage = sprintf('Error: The following folder does not exist:\n%s', LoadPath_MRI);
	uiwait(warndlg(errorMessage));
	return;
end

if ~isdir(StorePath_MRI)
    mkdir(StorePath_MRI);
end

if ~isempty(dir(fullfile(StorePath_MRI, '/*.png')))
    which_dir = StorePath_MRI;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end

if ~isdir(StorePath_GT)
    mkdir(StorePath_GT);
end

if ~isempty(dir(fullfile(StorePath_GT, '/*.png')))
    which_dir = StorePath_GT;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end

%% Load Data in loop

Files_GT = dir(filePattern_GT);
Files_MRI = dir(filePattern_MRI);

for k = 1:length(Files_GT)
	Filename_GT = fullfile(LoadPath_GT, Files_GT(k).name);
    Filename_MRI = fullfile(LoadPath_MRI, Files_MRI(k).name);
    disp(['Loading now: ', Filename_GT, '  and  ', Filename_MRI]);
    
    I = imread(Filename_GT);
    [row, col] = size(I);
    disp([row, col]);

    figure;
    I = imtranslate(I, [row/2,col/2]);
    imshow(I);
end



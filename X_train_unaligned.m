%% This file loads the nifti-files and generates the unaligned X_train data by storing the images as .mat files in enumerated sequence. 

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library for NIFTI_TOOLBOX
addpath(genpath('NIFTI_TOOLBOX'));

%% Define MyPath to our local Raw Data
LoadPath = 'Data\';
if ~isdir(LoadPath)
	errorMessage = sprintf('Error: The following folder does not exist:\n%s', LoadPath);
	uiwait(warndlg(errorMessage));
	return;
end

StorePath = 'X_train_unaligned\';
if ~isdir(StorePath)
    mkdir(StorePath);
end

if ~isempty(dir(fullfile(StorePath, '/*.mat')))
    which_dir = StorePath;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end

%% Load data in loop

filePattern = fullfile(LoadPath, 'patient*');
Files = dir(filePattern);
q=1; %enumerator 

for k = 1:length(Files)
	Filename_dillated = fullfile(LoadPath, Files(k).name, [Files(k).name '_frame01.nii.gz']);
    disp(Filename_dillated);
%     Filename_contracted = fullfile(myFolder, Files(k).name, 'patient001_frame1*.nii.gz');
%     disp(Filename_contracted);
    
    Img_dillated = load_nii(Filename_dillated);
    for i=2:5
        img = Img_dillated.img(:,:,i);  
        save(fullfile([StorePath, num2str(q) '_MRIMAT.mat']),'img');
        q=q+1;
    end
end

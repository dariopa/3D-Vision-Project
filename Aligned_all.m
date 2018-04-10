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

%% Define MyPath to our local Raw Data
% Check path for ground truth data
if ~isdir(LoadPath_GT)
	errorMessage = sprintf('Error: The following folder does not exist:\n%s', LoadPath_GT);
	uiwait(warndlg(errorMessage));
	return;
end

% Check path for MRI data
if ~isdir(LoadPath_MRI)
	errorMessage = sprintf('Error: The following folder does not exist:\n%s', LoadPath_MRI);
	uiwait(warndlg(errorMessage));
	return;
end

% Create Store Folder if not already existing
if ~isdir(StorePath_MRI)
    mkdir(StorePath_MRI);
end

% When running this code, delete all content in that folder
if ~isempty(dir(fullfile(StorePath_MRI, '/*.png')))
    which_dir = StorePath_MRI;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end

% Create Store Folder if not already existing
if ~isdir(StorePath_GT)
    mkdir(StorePath_GT);
end

% When running this code, delete all content in that folder
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
    % Load data
	Filename_GT = fullfile(LoadPath_GT, Files_GT(k).name);
    Filename_MRI = fullfile(LoadPath_MRI, Files_MRI(k).name);
    disp(['Loading now: ', Filename_GT, '  and  ', Filename_MRI]);
    
    %% Shift images
    % Load original image
    I = imread(Filename_GT);
    [row, col] = size(I);
    
    % generate thresholded image
    img = I;
    img(img < 255) = 0; 
    img = bwareaopen(img,2000);

    % find the centroid of the objects (use regionprops() ) & plot the original image with the centroid
    stat=regionprops(img,'Centroid');
    centroid=cat(2, stat.Centroid);
    
    % Calculate shift of images:
    x_mid = int16(row/2);
    y_mid = int16(col/2);
    
    % Shift images and surface points
    I_GT = imtranslate(I, [x_mid-stat.Centroid(1), y_mid-stat.Centroid(2)]);
    load(Filename_MRI);
    I_MRI = imtranslate(image_data, [x_mid-stat.Centroid(1), y_mid-stat.Centroid(2)]);

    % Store shifted images and surface points
    imwrite(I_GT, fullfile(StorePath_GT,['image' num2str(k) '_GT.png']));
    save(fullfile(StorePath_MRI,['image' num2str(k) '_MRIMAT.mat']),'I_MRI')
        
    %% Plot shifted image
    if k == 1
        figure;
        subplot(2,2,1)
        imshow(I)
        xlabel('Original ground truth image');

        subplot(2,2,2)
        imshow(img);
        xlabel('Thresholded Image');

        subplot(2,2,3)
        imshow(img)
        hold on
        plot(centroid(:,1),centroid(:,2),'rX','linewidth', 3, 'MarkerSize', 15)
        xlabel('Centroid in thresholded image');
        hold off

        subplot(2,2,4)
        imshow(I_GT);
        xlabel('Shifted image')
    end    
end



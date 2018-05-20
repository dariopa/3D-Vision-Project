%% Alignment of images towards centre of image. 

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
% imtool close all;	% Close all figure windows created by imtool.


%% HARDCODED INPUTS
LoadPath_GT = 'GT_Unaligned/';
LoadPath_MRI = 'MRI_Unaligned/';
LoadPath_SURF = 'SURF_Unaligned/';
StorePath_GT = 'GT_Aligned_cropped/';
StorePath_MRI = 'MRI_Aligned_cropped/';
StorePath_SURF = 'SURF_Aligned_cropped/';
filePattern_GT = fullfile(LoadPath_GT, '*.png');
filePattern_MRI = fullfile(LoadPath_MRI, '*.mat');
filePattern_SURF = fullfile(LoadPath_SURF, '*.asc');

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

% Check path for SURF data
if ~isdir(LoadPath_SURF)
	errorMessage = sprintf('Error: The following folder does not exist:\n%s', LoadPath_SURF);
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

% Create Store Folder if not already existing
if ~isdir(StorePath_SURF)
    mkdir(StorePath_SURF);
end

% When running this code, delete all content in that folder
if ~isempty(dir(fullfile(StorePath_SURF, '/*.asc')))
    which_dir = StorePath_SURF;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end

%% Load Data in loop

Files_GT = dir(filePattern_GT);
Files_MRI = dir(filePattern_MRI);
Files_SURF = dir(filePattern_SURF);

for k = 1:length(Files_GT)
    % Load data
    Filename_GT = fullfile(LoadPath_GT, ['image' num2str(k) '_GT.png']);
    Filename_MRI = fullfile(LoadPath_MRI, ['image' num2str(k) '_MRIMAT.mat']);
    Filename_SURF = fullfile(LoadPath_SURF, ['image' num2str(k) '_surf.asc']);
    disp(['Loading now: ', Filename_GT, '  and  ', Filename_MRI, '  and  ', Filename_SURF]);
    
    %% Shift images
    % Load original image
    I = imread(Filename_GT);
    [row, col] = size(I);
    
    % generate thresholded image
    img = I;
    img(img < 255) = 0; 
    img = imbinarize(img);

    % find the centroid of the objects (use regionprops() ) & plot the original image with the centroid
    stat=regionprops(img,'Centroid');
    centroid=cat(2, stat.Centroid);
    
    % Calculate shift of images:
    x_mid = int16(row/2);
    y_mid = int16(col/2);
    
    % Shift images and surface points
    % Ground Truth Image
    I_GT = imtranslate(I, [x_mid-stat.Centroid(1), y_mid-stat.Centroid(2)]);
    I_GT = I_GT(57:156, 57:156); % crop
    % MRI Image
    load(Filename_MRI);
    image_data = imtranslate(image_data, [x_mid-stat.Centroid(1), y_mid-stat.Centroid(2)]);
    image_data = image_data(57:156, 57:156); % crop
    % Surface Points
    surf = load(Filename_SURF);
    surf = surf(2:end,:);
    surf_y = surf(:,1); % ATTENTION HERE! FIRST COLUMN IS Y-POSITION!
    surf_x = surf(:,2); % ATTENTION HERE! SECOND COLUMN IS X-POSITION!
    surf_shifted_x = surf_x + cast(repelem(x_mid-stat.Centroid(1), length(surf))','double');
    surf_shifted_x = surf_shifted_x - 56.0; % crop
    surf_shifted_y = surf_y + cast(repelem(y_mid-stat.Centroid(2), length(surf))','double');
    surf_shifted_y = surf_shifted_y - 56.0; % crop
    
    % Store shifted images and surface points
    % Ground Truth Image
    imwrite(I_GT, fullfile(StorePath_GT,['image' num2str(k) '_GT.png']));
    % MRI Image
    save(fullfile(StorePath_MRI,['image' num2str(k) '_MRIMAT.mat']),'image_data');
    % Surface Points
    coordname = fullfile(StorePath_SURF,['image' num2str(k),'_surf.asc']);
    fileID = fopen(coordname,'w');
    fprintf(fileID,'%d %d\n',length(surf),length(surf));
    for j=1:length(surf)
        fprintf(fileID,'%f %f\n',surf_shifted_y(j),surf_shifted_x(j));
    end
    fclose(fileID);
        
    %% Plot shifted image
    if k == 1
        figure;
        subplot(2,2,1)
        imshow(I)
        hold on
        plot(surf_x, surf_y, 'r*', 'LineWidth', 2, 'MarkerSize', 2);
        xlabel('Original ground truth image');
        hold off
        
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
        hold on;
        plot(surf_shifted_x, surf_shifted_y, 'r*', 'LineWidth', 2, 'MarkerSize', 2);
        xlabel('Shifted image')
        hold off;
    end    
end



%% This file loads the .mat files from the unaligned X_train and generates the surface coordinates with snakes

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library for Snakes-Toolbox
addpath(genpath('snakes'));

%% Options for snakes

Options=struct;
Options.Verbose=true;
Options.Iterations=600;
Options.nPoints=50;
Options.Wedge=2;
Options.Wline=0;
Options.Wterm=0;
Options.Kappa=4;
Options.Sigma1=1;
Options.Sigma2=6;
Options.Alpha=0.1;
Options.Beta=1;
Options.Mu=0.2;
Options.Delta=-0.2;

%% Define pathes
LoadPath_GT = 'GT_Unaligned\'; 
StorePath = 'SURF_Unaligned\';

% Create Store Folder if not already existing
if ~isdir(StorePath)
    mkdir(StorePath);
end
% When running this code, delete all content in that folder
if ~isempty(dir(fullfile(StorePath, '/*.png')))
    which_dir = StorePath;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end

%% Read png images
filePattern_GT = fullfile(LoadPath_GT, '*.png');
Files_GT = dir(filePattern_GT);

for k = 1:length(Files_GT)
    % Load files
    Filename_GT = fullfile(LoadPath_GT, ['image' num2str(k) '_GT.png']);
    disp(['Loading now: ', Filename_GT]);
    I = imread(Filename_GT);
    [row, col] = size(I);
    y=[0 0 row col];
    x=[0 col row 0];
    P=[x(:) y(:)];
    I(I<255) = 0;
    I = im2double(I);
    
%     figure, imshow(I); [x,y] = getpts;
%     imshow(I);
    
    % run snakes
    [O,J] = Snake2D(I, P, Options);
    
    imshow(I); 
    hold on;
    img = plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    %% Store coordinates
    coordname = fullfile(StorePath,['image' num2str(k),'_surf.asc']);
    fileID = fopen(coordname,'w');
    fprintf(fileID,'%d %d\n',50,50);
    for j=1:50
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    saveas(img, fullfile(StorePath,['image' num2str(k) '_GT.png']));
end

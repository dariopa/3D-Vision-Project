%% This file loads the .mat files from the unaligned X_train and generates the surface coordinates with snakes

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library for Snakes-Toolbox
addpath(genpath('snakes'));

%% Options for snakes
Options=struct;
Options.Verbose=false;
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
    landmark = findlandmark(I); % find landmark
    [row, col] = size(I);
    y=[0 0 row col];
    x=[0 col row 0];
    P=[x(:) y(:)];
    I(I<255) = 0;
    I = im2double(I);
    
    % run snakes
    [O,J] = Snake2D(I, P, Options);
    hold off;
    imshow(I); 
    hold on;
    % visualize snakes
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    % visualize landmark
    img = plot(landmark(2), landmark(1), '-o', 'MarkerEdge','black', 'MarkerFaceColor','red');
    
    %% Sort coordinates 
    % with the help of the landmark
    
    % find index of O with shortest distance to the landmark
    distvec = O-landmark;
    [min_dist, index] = min(sqrt(distvec(:,1).^2 + distvec(:,2).^2));
    
    % re-order O
    % the new order starts from index and goes clock-wise
    O = [O(index:end,:); O(1:index-1,:)];
    
    %% Store coordinates
    coordname = fullfile(StorePath,['image' num2str(k),'_surf.asc']);
    fileID = fopen(coordname,'w');
    fprintf(fileID,'%d %d\n',Options.nPoints,Options.nPoints);
    for j=1:Options.nPoints
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    saveas(img, fullfile(StorePath,['image' num2str(k) '_GT.png']));
end

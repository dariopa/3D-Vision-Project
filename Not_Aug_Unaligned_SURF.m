%% This file loads the .mat files from the unaligned X_train and generates the surface coordinates with snakes

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.

%% Add library for Snakes-Toolbox
addpath(genpath('Toolbox/snakes'));

%% Distance threshold
thresh = 2;
%% Options for snakes
Options=struct;
Options.Verbose=false;
Options.Iterations=200;
Options.nPoints=50;
Options.Wedge=2;
Options.Wline=0;
Options.Wterm=0;
Options.Kappa=4;
Options.Sigma1=1;
Options.Sigma2=0.5;
Options.Alpha=0.1;
Options.Beta=0.1;
Options.Mu=0.2;
Options.Delta=0.2;

%% Define pathes
LoadPath_GT = 'Not_Aug_GT_Unaligned/'; 
StorePath = 'Not_Aug_SURF_Unaligned/';

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
    I(I<255) = 0;
    I = im2double(I);
    
    I_binarized = imbinarize(I);
    stat=regionprops(I_binarized,'Centroid');
    centroid=cat(2, stat.Centroid);
    y_center = stat.Centroid(1);
    x_center = stat.Centroid(2);
    
    y=[y_center-thresh y_center+thresh y_center+thresh y_center-thresh];
    x=[x_center-thresh x_center-thresh x_center+thresh x_center+thresh];
    P=[x(:) y(:)];

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
    
    if Options.Verbose==true
        close all;
    end
end

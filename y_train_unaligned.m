%% This file loads the .mat files from the unaligned X_train and generates the surface coordinates with snakes

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library for NIFTI_TOOLBOX
addpath(genpath('NIFTI_TOOLBOX'));
addpath(genpath('snakes'));

%% Define MyPath to our local Raw Data
MyPath = 'Data'; 

%% read hdf5 file
HDF5_Reading

no_img = size(MASKS,3);

fig = figure;
set(gca, 'Units', 'normal', 'Position', [0 0 1 1]);

for i = 1 : no_img
    I_original = MASKS(:,:,i);
    I = I_original;
    [x,y] = find(I == 3);
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    
    %imshow(I', []);
    %colormap gray;
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=3;
    [O,J] = Snake2D(I, [x,y], Options);
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %% save information
    StorePath = 'unaligned_snakes\';
    if ~isdir(StorePath)
        mkdir(StorePath);
    end
    filepath = 'unaligned_snakes\image';
    filepath = strcat(filepath,num2str(i));
    
    % save coordinates
    coordname = strcat(filepath,'_surf.asc');
    fileID = fopen(coordname,'w');
    fprintf(fileID,'%d %d\n',50,50);
    for j=1:50
        fprintf(fileID,'%d %d\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
end
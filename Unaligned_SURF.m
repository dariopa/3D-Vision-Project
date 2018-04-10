%% This file loads the .mat files from the unaligned X_train and generates the surface coordinates with snakes

clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

%% Add library for NIFTI_TOOLBOX
addpath(genpath('NIFTI_TOOLBOX'));
addpath(genpath('snakes'));

%% Define pathes
MyPath = 'Data\'; 
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

%% read hdf5 file
filename = fullfile(MyPath, 'column_full_UKBB_data_2D_size_212_212_res_1.36719_1.36719_sl_2_5_onlytrain.hdf5');
MASKS = h5read(filename,'/masks_train');

no_img = size(MASKS,3);

fig = figure;
set(gca, 'Units', 'normal', 'Position', [0 0 1 1]);

disp('doing preliminary round of snakes...');
for i = 1 : no_img
    I_original = MASKS(:,:,i);
    I = I_original;
    [x,y] = find(I == 3); % THIS DOESN'T WORK LIKE THAT!!!
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    
    %imshow(I', []);
    %colormap gray;
    
    Options=struct;
    Options.Iterations=1000;
    Options.Verbose=false;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=3;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %% save information
    filepath = fullfile(StorePath,'image');
    filepath = strcat(filepath,num2str(i));
    
    % save coordinates
    coordname = strcat(filepath,'_surf.asc');
    fileID = fopen(coordname,'w');
    fprintf(fileID,'%d %d\n',50,50);
    for j=1:50
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
end

disp('preliminary round finish, fixing bad ones...');
%% manuel snakes
%% image 3,4 %%%
for i = [3,4]
    I_original = MASKS(:,:,i);
    I = I_original;
    [x,y] = find(I == 2);
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=2;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
end
 %% image 11
    i = 11;
    I_original = MASKS(:,:,i);
    I = I_original;
    [x,y] = find(I ==3);
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    
    Options=struct;
    Options.Iterations=1000;
    Options.nPoints=50;
    Options.Alpha=0.18;
    Options.Beta=1;
    Options.Wedge=6;
    Options.Wterm=0.2;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');

%% image 21
    i=21;
    I_original = MASKS(:,:,i);
    I = I_original;
    x = [74.3635 70.9984 63.9317 60.2302 61.2397 64.6048 69.9889 76.3825 80.0841 83.4492 85.4683 85.4683 85.1317 80.7571]';
    y = [36.0016 36.0016 38.3571 44.7508 47.7794 53.8365 58.2111 59.8937 59.8937 57.8746 53.8365 49.1254 44.0778 40.0397]';
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    
    Options=struct;
    Options.Iterations=1000;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=4.5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
    
 %% image 22
 i=22;
 I_original = MASKS(:,:,i);
    I = I_original;
    x = [87.8238 83.4492 82.1032 85.8048 98.2556 108.6873 122.8206 127.5317 130.5603 128.5413 119.7921 111.3794 96.5730]';
    y = [77.3921 90.8524 103.9762 114.0714 123.8302 124.8397 121.1381 112.0524 100.6111 87.1508 79.0746 76.0460 73.3540]';
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);

    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.18;
    Options.Beta=1;
    Options.Wedge=5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
    
%% image 23
i = 23;
I_original = MASKS(:,:,i);
    I = I_original;
    x = [111.7159 100.2746 95.2270 98.5921 103.6397 115.4175 127.8683 136.9540 138.3000 135.6079 130.5603 121.8111]';
    y = [148.3952 151.7603 164.2111 175.3159 182.7190 188.7762 188.4397 181.0365 168.9222 164.2111 154.1159 149.4048]';
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);

    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');

 %% image 24
    i = 24;
    I_original = MASKS(:,:,i);
    I = I_original;
    x = [95.5635 87.8238 87.8238 90.1794 94.2175 102.2937 107.3413 114.4079 120.8016 123.8302 124.8397 117.7730 110.3698 99.2651]';
    y = [80.0841 88.1603 99.6016 104.9857 111.0429 115.7540 117.1000 115.0810 110.7063 102.6302 94.5540 87.4873 83.7857 82.4397]';
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    
    Options=struct;
    Options.Iterations=1000;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
    
 %% image 25, 26
 for i = [25, 26]
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    [x,y] = find(I_original==2 );
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.16;
    Options.Beta=1;
    Options.Wedge=7;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
 end
 
 %% image 27, 28, 29, 30, 31, 32, 55, 56, 57, 58, 64, 91, 92, 95, 183, 234
for i = [27:32, 55:58, 64, 91:92, 95, 183, 234]
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    [x,y] = find(I_original==2);
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.16;
    Options.Beta=1;
    Options.Wedge=6;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
end

%% image 43
    i = 43;
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    x = [113.7349 95.2270 72.6810 57.8746 57.8746 82.1032 99.6016 113.7349 116.4270]';
    y = [36.3381 28.9349 29.6079 37.3476 51.8175 66.9603 64.9413 58.2111 42.0587]';
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');

%% image 63
    i = 63;
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    [x,y] = find(I_original==3);
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.23;
    Options.Beta=1;
    Options.Wedge=3.5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');

%% image 96
    i = 97;
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    [x,y] = find(I_original==2);
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.16;
    Options.Beta=1;
    Options.Wedge=5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');

%% image 171
    i = 171;
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    [x,y] = find(I_original==2);
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.19;
    Options.Beta=1;
    Options.Wedge=6;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
    
%% image 175
    i = 175;
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    [x,y] = find(I_original==2);
    
    Options=struct;
    Options.Iterations=1000;
    Options.nPoints=50;
    Options.Alpha=0.3;
    Options.Beta=0.2;
    Options.Wedge=2.5;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');

%% image 237, 239, 240, 285, 286, 287, 288
for i = [237, 239, 240, 285:288]
    I_original = MASKS(:,:,i);
    I = I_original;
    I(I==1) = 0;
    I(I==2) = 0;
    I = im2double(I);
    [x,y] = find(I_original==2);
    
    Options=struct;
    Options.Iterations=1000;
    %Options.Verbose=true;
    Options.nPoints=50;
    Options.Alpha=0.2;
    Options.Beta=1;
    Options.Wedge=7;
    [O,J] = Snake2D(I, [x,y], Options);
    
    imshow(I_original, []); hold on;
    plot([O(:,2);O(1,2)],[O(:,1);O(1,1)]);
    
    
    %save information
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
        fprintf(fileID,'%f %f\n',O(j,1),O(j,2));
    end
    fclose(fileID);
    
    % save png file
    imgname = strcat(filepath,'_GT');
    print(imgname,'-dpng','-noui');
end

disp('done');
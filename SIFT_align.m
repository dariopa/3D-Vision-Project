clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.

%% HARDCODED INPUTS
LoadPath_MRI = 'MRI_Unaligned/';
StorePath_MRI = 'MRI_SIFT_Aligned/';
filePattern_MRI = fullfile(LoadPath_MRI, '*.mat');

%% Define MyPath to our local Raw Data
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

%% create centered myocordium
% get one of the images to shift and crop
I = imread('SIFT/image7_MRIMAT.png');

% shift
I_shifted = imtranslate(I, [-40, 4]);

% off-sets
offx = 60;
offy = 70;

% crop
I_cropped =I_shifted(offx:140,offy:160);

% save
imwrite(I_cropped, fullfile('SIFT/',['centered_img.png']));
center_img = imread('SIFT/centered_img.png');

%% SIFT in loop
Files_MRI = dir(filePattern_MRI);

shifted_files = 0;
not_shifted_files = 0;
for k = 1:(length(Files_MRI) / 2)
    if mod(k+1,2)
        % Load data
        Filename_MRI = fullfile(LoadPath_MRI, ['image' num2str(k) '_MRIMAT.mat']);
        disp(['Loading now: ', Filename_MRI]);
        
        %% Shift images
        % Load original image
        A = load(Filename_MRI);
        image_data = A.image_data;
        MRI_png = uint8(image_data*255);
        
        [clmnX1, clmnX2, rowY1, rowY2] = SIFT_FP(MRI_png, center_img, offx, offy);
        
        if ~isempty(clmnX1)
            xdiff = mean(clmnX1-clmnX2);
            ydiff = mean(rowY1-rowY2);
            image_data = imtranslate(image_data, [xdiff, ydiff]);
            MRI_png = imtranslate(MRI_png, [xdiff, ydiff]);
            
            % Store shifted images
            % MRI Image
            save(fullfile(StorePath_MRI,['image' num2str(k) '_MRIMAT.mat']),'image_data');
            
            FileName_png = strcat(StorePath_MRI,'image', num2str(k), '_MRIMAT.png');
            imwrite(MRI_png,FileName_png);
            shifted_files = shifted_files + 1;
        else
            % Store shifted images
            % MRI Image
            save(fullfile(StorePath_MRI,['image' num2str(k) '_MRIMAT_not_shifted.mat']),'image_data');
            
            FileName_png = strcat(StorePath_MRI,'image', num2str(k), '_MRIMAT_not_shifted.png');
            imwrite(MRI_png,FileName_png);
            not_shifted_files = not_shifted_files + 1;
        end
    end
    
end

disp([num2str(shifted_files) ' files have been shifted.' ]);
disp([num2str(not_shifted_files) ' files were not shifted.']);




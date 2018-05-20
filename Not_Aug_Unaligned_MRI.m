close all;
clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CHANGE THIS IF YOU WANT TO EXPORT THE WHOLE DATA SET %%%%
export_all = 1;     % 1 - Export All
                    % 0 - Export 4 images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% Trainingsset einlesen
filename = 'preproc_data_augmented/column_full_UKBB_data_2D_size_212_212_res_1.36719_1.36719_sl_2_5_onlytrain.hdf5';

%h5disp(filename);      % Function to inspect the hdf5 file
info = h5info(filename);

M = h5read(filename,'/images_train');   % Assigns all Training Images from the HDF5-File to Variable M
MASKS = h5read(filename,'/masks_train');

FirstPic = M(:,:,19);
FirstMask = MASKS(:,:,19);

figure;
imshow(FirstPic,[]);
figure;
imshow(FirstMask,[]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EXPORT MRI IMAGES MIT NUMMERIERUNG %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%From data.py
%img_file = file_name[idx] + "_MRIMAT.mat"

StorePath = 'Not_Aug_MRI_Unaligned/';
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

if (export_all == 1)
export_size = size(M,3);    % number of pictures
else
    export_size = 4;
end

for k=1:export_size
    i = k;
    if mod(i,2) == 0
        FileName = strcat(StorePath,'image', num2str(i/2), '_MRIMAT.mat');
        FileName_png = strcat(StorePath,'image', num2str(i/2), '_MRIMAT.png');
        image_data = M(:,:,k);
        MRI_png = uint8(image_data*255);
        imwrite(MRI_png,FileName_png);

        save(FileName,'image_data');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EXPORT GROUND TRUTH IMAGES MIT NUMMERIERUNG %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pngPath = 'Not_Aug_GT_Unaligned/';
if ~isdir(pngPath)
    mkdir(pngPath);
end

if ~isempty(dir(fullfile(pngPath, '/*.png')))
    which_dir = pngPath;
    dinfo = dir(which_dir);
    dinfo([dinfo.isdir]) = [];   %skip directories
    filenames = fullfile(which_dir, {dinfo.name});
    delete( filenames{:} )
end

% We have to get Values from 0 to 255 for png images!!
% Ground Truth image has Intensitiy Values {0,1,2,3}
ScalingFactor = 255/3;
FirstMask = FirstMask * ScalingFactor;

for i = 1:export_size
    j = i;
    if mod(i,2) == 0
        %%From data.py
        %GT_file = file_name[idx] + "_GT.png"
        pngName = strcat(pngPath,'image', num2str(j/2), '_GT.png');
        png_data = MASKS(:,:,i)*ScalingFactor;
        imwrite(png_data,pngName);
    end
end
disp('Job terminated!')
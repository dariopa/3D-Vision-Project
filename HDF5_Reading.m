close all;
clear all;
clc;

%%%% Trainingsset einlesen
filename = 'Data/column_full_UKBB_data_2D_size_212_212_res_1.36719_1.36719_sl_2_5_onlytrain.hdf5';

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

%%From data.py
%img_file = file_name[idx] + "_MRIMAT.mat"

StorePath = 'MRI_Export\';
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

%%% Export mit Nummerierung

%%%Export ALL
%for k=1:size(M,3)
for k=1:4
FileName = strcat(StorePath,'image', num2str(k), '_MRIMAT.mat');
disp(FileName);
image_data = M(:,:,k);
save(FileName,'image_data');
end




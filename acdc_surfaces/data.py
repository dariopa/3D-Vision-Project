import utils

import os
import glob
import fnmatch

import numpy as np

import h5py
from scipy.io import loadmat


class DataMakerSynth:

    def __init__(self, out_file_name, base_path_MRI, base_path_GT, base_path_SURF):
        self.base_path_MRI = base_path_MRI
        self.base_path_GT = base_path_GT
        self.base_path_SURF = base_path_SURF
        self.out_file_name = out_file_name


    def run(self):

        lab_list = {'test': [], 'val': [], 'train': []}
        img_list = {'test': [], 'val': [], 'train': []}
        patient_id_list = {'test': [], 'val': [], 'train': []}
        GT_list = {'test': [], 'val': [], 'train': []}

        num_images = len(fnmatch.filter(os.listdir(self.base_path_GT), '*.png'))

        num_train = int(num_images * 0.8)
        num_val = int(num_images * 0.1)
        num_test = num_images - num_train - num_val

        for idx in range(num_images):

            if (idx % 5 == 0):
                train_val_test = 'test'
            elif (idx % 5 == 1):
                train_val_test = 'val'
            else:
                train_val_test = 'train'

            img_file = self.base_path_MRI + 'image' + str(idx+1) + "_MRIMAT.mat"
            img = loadmat(img_file)['image_data']

            lab_file = self.base_path_SURF + 'image' + str(idx+1) + "_surf.asc"
            lab = utils.load_vrtx(lab_file)

            GT_file = self.base_path_GT + 'image' + str(idx+1) + "_GT.png"
            GT = utils.load_png(GT_file)

            img_list[train_val_test].append(img)
            lab_list[train_val_test].append(lab)
            patient_id_list[train_val_test].append(idx+1)
            GT_list[train_val_test].append(GT)

        hdf5_file = h5py.File(self.out_file_name, "w")
        print(patient_id_list)
        for tt in ['train', 'val', 'test']:
            hdf5_file.create_dataset('images_%s' % tt, data=np.asarray(img_list[tt], dtype=np.float32))
            hdf5_file.create_dataset('labels_%s' % tt, data=np.asarray(lab_list[tt], dtype=np.float32))
            hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint32))
            hdf5_file.create_dataset('GT_%s' % tt, data=np.asarray(GT_list[tt], dtype=np.float32))

        hdf5_file.close()



if __name__ == '__main__':

# Unaligned data
    # base_path_MRI = '../Data_Final/MRI_Unaligned/'
    # base_path_GT = '../Data_Final/GT_Unaligned/'
    # base_path_SURF = '../Data_Final/SURF_Unaligned/'
    # out_file_name = '../preproc_data_augmented/Unaligned_Data.hdf5'

# Aligned data
    base_path_MRI = '../Data_Final/MRI_Aligned/'
    base_path_GT = '../Data_Final/GT_Aligned/'
    base_path_SURF = '../Data_Final/SURF_Aligned/'
    out_file_name = '../preproc_data_augmented/Aligned_Data.hdf5'


    data_maker = DataMakerSynth(out_file_name, base_path_MRI, base_path_GT, base_path_SURF)

    data_maker.run()

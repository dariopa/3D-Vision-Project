import utils

import os
import glob

import numpy as np

import h5py
from scipy.io import loadmat


class DataMakerSynth:

    def __init__(self, out_file_name, base_path):
        self.base_path = base_path
        self.out_file_name = out_file_name


    def run(self):

        lab_list = {'test': [], 'val': [], 'train': []}
        img_list = {'test': [], 'val': [], 'train': []}
        patient_id_list = {'test': [], 'val': [], 'train': []}
        GT_list = {'test': [], 'val': [], 'train': []}
        # img_list = {'test': [], 'train': [] }
        # patient_id_list = {'test': [], 'train': []}

        file_list = sorted(glob.glob('{}/*_MRI.png'.format(self.base_path)))
        #subject_ids = [ele.split('/')[-1].split('_')[0] for ele in file_list]

        #base_id = '10000'
        #subject_ids.remove(base_id)

        subject_ids = [ele.split('/')[-1].split('_')[0].split('t')[-1] for ele in file_list]

        file_name = [ele.rsplit('_', 1)[0] for ele in file_list]

        num_subjects = len(subject_ids)
        num_train = int(num_subjects * 0.8)
        num_val = int(num_subjects * 0.1)
        num_test = num_subjects - num_train - num_val


        for idx in range(0, len(subject_ids)):

            #if idx < num_train:
            #    train_val_test = 'train'
            #elif idx < num_train + num_val:
            #    train_val_test = 'val'
            #else:
            #    train_val_test = 'test'
            if (int(subject_ids[idx]) % 5 == 0):
                train_val_test = 'test'
            elif (int(subject_ids[idx])  % 5 == 1):
                train_val_test = 'val'
            else:
                train_val_test = 'train'

            print('------------------------')
            print(idx, ', ', subject_ids[idx])
            print(train_val_test)

            img_file = file_name[idx] + "_MRIMAT.mat"
            img = loadmat(img_file)['I_bw']
            print(img.shape)

            #img_file = file_name[idx] + "_MRI.png"
            #img = utils.load_png(img_file)
            #print(img.shape)

            lab_file = file_name[idx] + "_surf.asc"
            lab = utils.load_vrtx(lab_file)


            GT_file = file_name[idx] + "_GT.png"
            GT = utils.load_png(GT_file)

            img_list[train_val_test].append(img)
            lab_list[train_val_test].append(lab)
            patient_id_list[train_val_test].append(subject_ids[idx])
            GT_list[train_val_test].append(GT)

        hdf5_file = h5py.File(self.out_file_name, "w")

        for tt in ['train', 'val', 'test']:
            hdf5_file.create_dataset('images_%s' % tt, data=np.asarray(img_list[tt], dtype=np.float32))
            hdf5_file.create_dataset('labels_%s' % tt, data=np.asarray(lab_list[tt], dtype=np.float32))
            hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint32))
            hdf5_file.create_dataset('GT_%s' % tt, data=np.asarray(GT_list[tt], dtype=np.float32))

        hdf5_file.close()



if __name__ == '__main__':

    base_path = '/home/tothovak/work/CSR/augmented_full_UKBB_data_2D_size_200_200_res_1.8269_sl_4_4/'
    out_file_name = '/home/tothovak/work/CSR/augmented_full_UKBB_data_2D_size_200_200_res_1.8269_sl_4_4/augmented_full_UKBB_data_2D_size_200_200_res_1.8269_sl_4_4.hdf5'


    data_maker = DataMakerSynth(out_file_name, base_path)

    data_maker.run()

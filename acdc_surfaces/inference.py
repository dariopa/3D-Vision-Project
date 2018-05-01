import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
import config.system as sys_config
import numpy as np
import utils
import scipy.misc as misc
import os.path
import model as model
from scipy.io import loadmat


### EXPERIMENT CONFIG FILE #############################################################
# Set the config file of the experiment you want to run here:
from experiments import Unaligned_Data as exp_config
########################################################################################
DATAFILE = '../preproc_data_augmented/Unaligned_Data.hdf5'
SESSION = '../acdc_logdir/Unaligned_Data1000Epoch_0.0001LR16Batch1keep0WD/'
#log_dir_name = exp_config.experiment_name + str(exp_config.max_epochs) +'Epoch_' + str(exp_config.learning_rate) + 'LR' + str(exp_config.batch_size) + 'Batch'

log_dir_name = SESSION.split("/")[-2]

log_dir = os.path.join(sys_config.log_root, log_dir_name)


def run_inference():
    # Load data
    data_file = DATAFILE
    data = h5py.File(data_file, 'r')
    # the following are HDF5 datasets, not numpy arrays
    images_val = data['images_val']
    labels_val = data['labels_val']

    images_test = data['images_test']
    labels_test = data['labels_test']



    patientID_test = data['patient_id_test']
    GT_test = data['GT_train']
    GT_test = np.reshape(GT_test, (GT_test.shape[0], exp_config.image_size[0], exp_config.image_size[1]))
    num_samples = GT_test.shape[0]

    if hasattr(exp_config, 'do_pca'):
        do_pca = exp_config.do_pca
       # if (do_pca == True):
        PCA_U_file_path = os.path.join(sys_config.preproc_folder, 'data_PCA_U.mat')
        PCA_mean_file_path = os.path.join(sys_config.preproc_folder, 'data_PCA_mean.mat')
        PCA_sqrtsigma_file_path = os.path.join(sys_config.preproc_folder, 'data_PCA_sqrtsigma.mat')
        PCA_U = loadmat(PCA_U_file_path)['U']
        PCA_U = PCA_U[:, 0:exp_config.nlabels]
        PCA_mean = loadmat(PCA_mean_file_path)['mean']
        PCA_sqrtsigma = loadmat(PCA_sqrtsigma_file_path)['sqrtsigma']
        PCA_sqrtsigma = PCA_sqrtsigma[0:exp_config.nlabels, 0:exp_config.nlabels]
    else:
        do_pca = False

    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.

        #image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
        #mask_tensor_shape = [exp_config.batch_size, 1, exp_config.num_vertices, 2]
        #image_tensor_shape = [None] + list(exp_config.image_size) + [1]
        #mask_tensor_shape = [None, 1, exp_config.num_vertices, 2]

        #image_tensor_shape = [num_samples] + list(exp_config.image_size) + [1]
        #mask_tensor_shape = [num_samples, 1, exp_config.num_vertices, 2]
        image_tensor_shape = [1] + list(exp_config.image_size) + [1]
        mask_tensor_shape = [1, 1, exp_config.num_vertices, 2]

        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        labels_pl = tf.placeholder(tf.float32, shape=mask_tensor_shape, name='labels')

        if do_pca:
            PCA_U_pl = tf.placeholder(tf.float32, shape=PCA_U.shape, name='PCA_U')
            PCA_mean_pl = tf.placeholder(tf.float32, shape=PCA_mean.shape, name='PCA_mean')
            PCA_sqrtsigma_pl = tf.placeholder(tf.float32, shape=PCA_sqrtsigma.shape, name='PCA_sqrtsigma')

        #training_pl = tf.placeholder(tf.bool, shape=[])
        #training = tf.constant(False, dtype=tf.bool)

        # Build a Graph that computes predictions from the inference model.
        if do_pca:
            logits = model.inferencePCA(images_pl, exp_config, PCA_U_pl, PCA_mean_pl, PCA_sqrtsigma_pl, tf.constant(1,dtype=tf.float32), tf.constant(False, dtype=tf.bool))
        else:
            logits = model.inference(images_pl, exp_config, tf.constant(1,dtype=tf.float32), tf.constant(False, dtype=tf.bool))

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        sess.run(init)

        saver.restore(sess, tf.train.latest_checkpoint(SESSION))

        images_test = np.expand_dims(images_test, axis=3)
        print(images_test.shape)


        DICEall = np.zeros((images_test.shape[0], 1))

        # save predictions
        res_path = os.path.join(sys_config.out_data_root, log_dir_name)
        if not tf.gfile.Exists(res_path):
            tf.gfile.MakeDirs(res_path)
        for i in range(images_test.shape[0]):

            if do_pca:
                pred = sess.run(logits, feed_dict={images_pl: np.expand_dims(images_test[i, :, :, :], axis=0), PCA_U_pl: PCA_U, PCA_mean_pl: PCA_mean, PCA_sqrtsigma_pl: PCA_sqrtsigma})
            else:
                pred = sess.run(logits, feed_dict={images_pl: np.expand_dims(images_test[i, :, :, :], axis=0)})
            print(pred.shape)
            pred = np.squeeze(pred, axis=(1,))
            print(pred.shape)

            res = np.squeeze(pred, axis=(0,))
            #res = np.asarray(pred[i, :, :])
            # print(res)

            # save result
            outFile = os.path.join(res_path, "pred" + str(i) + ".csv")
            np.savetxt(outFile, res, delimiter=" ")

            # saveplot
            outFilePng = os.path.join(res_path, "pred" + str(i) + ".png")
            # utils.view_plot(res)
            axis_limits = [0, exp_config.image_size[0], 0, exp_config.image_size[1]]
            utils.save_plot(res, outFilePng, axis_limits)

            # save intensity image
            outFileBW = os.path.join(res_path, "pred" + str(i) + "BW.png")
            outImageBW = images_test[i, :, :, :]
            print(outImageBW.shape)
            outImageBW = np.squeeze(outImageBW, axis=(2,))
            misc.imsave(outFileBW, outImageBW)

            # save GT segmentation
            outFileGTBW = os.path.join(res_path, "pred" + str(i) + "GTBW.png")
            outImageGTBW = GT_test[i, :, :]
            print(outImageGTBW.shape)
            misc.imsave(outFileGTBW, outImageGTBW)

            # save true label
            outFileGT = os.path.join(res_path, "pred" + str(i) + "GT.png")
            print(labels_test.shape)
            outGT = labels_test[i, :, :]

            utils.save_plot(outGT, outFileGT, axis_limits)
            utils.save_plot

            # save GT and pred in one plot
            outFileGTpred = os.path.join(res_path, "pred" + str(i) + "predGT.png")
            outGT = labels_test[i, :, :]

            utils.save_plot(outGT, outFileGT, axis_limits)
            # utils.save_plot
            utils.save_two_plots(outGT, res, outFileGTpred, axis_limits)

            # save overlay plot
            outOverlay = os.path.join(res_path, "pred" + str(i) + "overlay.png")
            utils.overlay_img_2plots(outGT, res, outImageBW, outOverlay, axis_limits)



            segmentation = utils.create_segmentation(res, exp_config.image_size)
            # # save segmentation image
            # outFileSegm = os.path.join(res_path, "pred" + str(i) + "segm.png")
            # # print(segmentation)
            # misc.imsave(outFileSegm, segmentation)

            # compute DICE coefficient
            DICEall[i], _, _, _, _ = utils.computeDICE(segmentation.astype(int), GT_test[i, :, :] / 255)
            print(i, " ; ", patientID_test[i] ,": ", DICEall[i])

        print("**Global stats**")
        print("Avg Dice: ", DICEall.mean())
        print("Std Dice: ", DICEall.std())

        with open(os.path.join(out_data_root,'Results.csv'), 'w+') as fp:
            fp.write('Average Dice:  ' + ',' + str(DICEall.mean() + '\n'))
            fp.write('Standard Dice:  ' + ',' + str(DICEall.std()))

        sess.close()
    data.close()


def main():

    run_inference()

if __name__ == '__main__':
    main()

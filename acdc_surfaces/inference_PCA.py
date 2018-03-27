import h5py
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

#from experiments import unet2D_bn_ssd as exp_config
#from experiments import shallow2D_ssd as exp_config
#from experiments import FCN8_bn_ssd as exp_config
#from experiments import PCA_main as exp_config
#from experiments import PCA as exp_config
#from experiments import CL9_DL1 as exp_config
#from experiments import CL5_DL3 as exp_config
#from experiments import CL5_DL1 as exp_config
#from experiments import CL9_nobias_DL1_nobias as exp_config
#from experiments import CL9_DL1_UKBB as exp_config
#from experiments import shallow_FCN_UKBB as exp_config
from experiments import shallow_FCN_small_UKBB as exp_config
########################################################################################
DATAFILE = '/home/tothovak/work/CSR/Code/acdc_segmenter/preproc_data/column_augmented_UKBB_data_2D_size_60_60_res_1.8_1.8_sl_4_4_corrected.hdf5'
#DATAFILE = '/home/tothovak/work/CSR/Code/acdc_segmenter/preproc_data/column_synt_data_2D_size_212_212_res_1.36719_1.36719_sl_2_5.hdf5'
#SESSION = '/home/tothovak/work/CSR/Code/acdc_segmenter/acdc_logdir/unet2D_bn_ssd2500Epoch_0.001LR5Batch/'
#SESSION = '/home/tothovak/work/CSR/Code/acdc_segmenter/acdc_logdir/shallow2D_ssd200Epoch_0.0001LR5Batch/'
#SESSION = '/home/tothovak/work/CSR/Code/acdc_segmenter/acdc_logdir/FCN8_bn_ssd2000Epoch_0.001LR5Batch/'
#SESSION = '/home/tothovak/work/CSR/Code/acdc_segmenter/acdc_logdir/PCA_main2500Epoch_0.0001LR5Batch/'
SESSION = '/home/tothovak/work/CSR/Code/acdc_segmenter/acdc_logdir/shallow_FCN_small_UKBB_bn111111Epoch_0.0001LR5Batch1keep0.0WD100PCA/'
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

    images_test = data['images_train']
    labels_test = data['labels_train']



    patientID_test = data['patient_id_train']
    GT_test = data['GT_train']
    GT_test = np.reshape(GT_test, (GT_test.shape[0], exp_config.image_size[0], exp_config.image_size[1]))
    num_samples = GT_test.shape[0]


    if hasattr(exp_config, 'do_pca'):
        do_pca = exp_config.do_pca
       # if (do_pca == True):
        PCA_U_file_path = os.path.join(sys_config.preproc_folder, 'UKBB_PCA_U.mat')
        PCA_mean_file_path = os.path.join(sys_config.preproc_folder, 'UKBB_PCA_mean.mat')
        PCA_sqrtsigma_file_path = os.path.join(sys_config.preproc_folder, 'UKBB_PCA_sqrtsigma.mat')
        PCA_U = loadmat(PCA_U_file_path)['U']
        PCA_U = PCA_U[:, 0:exp_config.nlabels]
        PCA_mean = loadmat(PCA_mean_file_path)['mean']
        PCA_sqrtsigma = loadmat(PCA_sqrtsigma_file_path)['sqrtsigma']
        PCA_sqrtsigma = PCA_sqrtsigma[0:exp_config.nlabels, 0:exp_config.nlabels]
    else:
        do_pca = False

    if hasattr(exp_config, 'multi_offset_prediction'):
        multi_offset = exp_config.multi_offset_prediction
        if (multi_offset == True):
            multi_offset_matrix = utils.get_offset_matrix(exp_config)
    else:
        multi_offset = False


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
        elif multi_offset:
            multi_offset_matrix_pl = tf.placeholder(tf.float32, shape = multi_offset_matrix.shape, name='multi_offset_matrix')

        # Build a Graph that computes predictions from the inference model.
        if do_pca:
            [logits, weights, shift] = model.inference_multi_PCA(images_pl, exp_config, PCA_U_pl, PCA_mean_pl, PCA_sqrtsigma_pl, tf.constant(1,dtype=tf.float32), tf.constant(False, dtype=tf.bool))
        elif multi_offset:
                logits = model.inference_multi_offset(images_pl, exp_config, multi_offset_matrix_pl, tf.constant(1,dtype=tf.float32), tf.constant(False, dtype=tf.bool))
        else:
            logits = model.inference_multi(images_pl, exp_config, tf.constant(1,dtype=tf.float32), tf.constant(False, dtype=tf.bool))

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
                [pred, predweights, predshift] = sess.run([logits, weights, shift],
                                                          feed_dict={images_pl: np.expand_dims(images_test[i, :, :, :], axis=0),
                                                                     PCA_U_pl: PCA_U, PCA_mean_pl: PCA_mean, PCA_sqrtsigma_pl: PCA_sqrtsigma})
            elif multi_offset:
                pred = sess.run(logits, feed_dict={images_pl: np.expand_dims(images_test[i, :, :, :], axis=0), multi_offset_matrix_pl: multi_offset_matrix})
            else:
                pred = sess.run(logits, feed_dict={images_pl: np.expand_dims(images_test[i, :, :, :], axis=0)})
            print(pred.shape)

            final_pred = utils.mean_prediction(np.squeeze(pred), exp_config)
            final_pred = np.array(np.squeeze(final_pred))
            print(np.shape(final_pred))

        #save result
            outFinal = os.path.join(res_path, "pred" + str(i) +"_final.csv")
            np.savetxt(outFinal, final_pred, delimiter=" ")

            outFile = os.path.join(res_path, "pred" + str(i) + ".csv")
            res = np.squeeze(pred)

            np.savetxt(outFile, res, delimiter=" ")

            if do_pca:
                outWeights = os.path.join(res_path, "pred" + str(i) + "Weights.csv")
                w = np.squeeze(predweights)
                np.savetxt(outWeights, w, delimiter = " ")

                outShift = os.path.join(res_path, "pred" + str(i) + "Shift.csv")
                s = np.squeeze(predshift)
                np.savetxt(outShift, s, delimiter=" ")

        # save GT
            outFileGT = os.path.join(res_path, "pred" + str(i) + "GT.csv")
            outGT = labels_test[i, :, :]
            outGT = np.transpose(outGT)
            outGT = np.reshape(outGT,(2, exp_config.num_vertices))
            outGT = np.transpose(outGT)

            np.savetxt(outFileGT, np.squeeze(outGT), delimiter=" ")


         #   outGT = np.concatenate((outGT[:,1:exp_config.num_vertices], outGT[:,exp_config.num_vertices+1]), axis=1)
        #     pred = np.squeeze(pred, axis=(1,))
        #     print(pred.shape)
        #
        #     res = np.squeeze(pred, axis=(0,))
        #     #res = np.asarray(pred[i, :, :])
        #     # print(res)
        #
        #     # save result
        #     outFile = os.path.join(res_path, "pred" + str(i) + ".csv")
        #     np.savetxt(outFile, res, delimiter=" ")
        #
         # saveplot
            outFilePng = os.path.join(res_path, "pred" + str(i) + ".png")
        #   # utils.view_plot(res)
            axis_limits = [0, exp_config.image_size[0], 0, exp_config.image_size[1]]
            utils.save_plot(final_pred, outFilePng, axis_limits)
        #
        #     # save intensity image
        #     outFileBW = os.path.join(res_path, "pred" + str(i) + "BW.png")
            outImageBW = images_test[i, :, :, :]
        #     print(outImageBW.shape)
            outImageBW = np.squeeze(outImageBW, axis=(2,))
        #    misc.imsave(outFileBW, outImageBW)
        #
            # save GT segmentation
        #    outFileGTBW = os.path.join(res_path, "pred" + str(i) + "GTBW.png")
        #    outImageGTBW = GT_test[i, :, :]
        #    print(outImageGTBW.shape)
        #    misc.imsave(outFileGTBW, outImageGTBW)
        #
        #     # save true label
        #     outFileGT = os.path.join(res_path, "pred" + str(i) + "GT.png")
        #     print(labels_test.shape)
        #     outGT = labels_test[i, :, :]
        #
        #     utils.save_plot(outGT, outFileGT, axis_limits)
        #     utils.save_plot
        #
        # save GT and pred in one plot
        #    outFileGTpred = os.path.join(res_path, "pred" + str(i) + "predGT.png")
        #    outGT = labels_test[i, :, :]
        #
        #     utils.save_plot(outGT, outFileGT, axis_limits)
        #     # utils.save_plot
        #     utils.save_two_plots(outGT, res, outFileGTpred, axis_limits)
        #
        # save overlay plot
            mean = np.reshape(PCA_mean,(exp_config.num_vertices,2),order='F')
            outOverlay = os.path.join(res_path, "pred" + str(i) + "overlay.png")
            utils.overlay_img_3plots(outGT, final_pred, mean, outImageBW, outOverlay, axis_limits)
        #
        #
        #
        #     segmentation = utils.create_segmentation(res, exp_config.image_size)
        #     # save segmentation image
        #     outFileSegm = os.path.join(res_path, "pred" + str(i) + "segm.png")
        #     # print(segmentation)
        #     misc.imsave(outFileSegm, segmentation)
        #
        #     # compute DICE coefficient
        #     DICEall[i], _, _, _, _ = utils.computeDICE(segmentation.astype(int), GT_test[i, :, :] / 255)
        #     print(i, " ; ", patientID_test[i] ,": ", DICEall[i])
        #
        # print("**Global stats**")
        # print("Avg Dice: ", DICEall.mean())
        # print("Std Dice: ", DICEall.std())
        sess.close()
    data.close()


def main():

    run_inference()

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     description="Train a neural network.")
    # parser.add_argument("CONFIG_PATH", type=str, help="Path to config file (assuming you are in the working directory)")
    # args = parser.parse_args()

    main()

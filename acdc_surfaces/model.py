# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import tensorflow as tf
from tfwrapper import losses
import numpy as np


import tensorflow.examples.tutorials.mnist


def inference(images, exp_config, keep_prob, training):
    '''
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module. 
    '''
    return exp_config.model_handle(images, training, keep_prob, nlabels=exp_config.nlabels)




def inference_multi(images, exp_config, keep_prob, training):
    '''
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module.
    '''
    tmp = exp_config.model_handle(images, training, keep_prob, nlabels=exp_config.nlabels)
    tmp = tf.reshape(tmp, (tf.shape(tmp)[0], tf.shape(tmp)[1] * tf.shape(tmp)[2], exp_config.nlabels))
    return tmp




def inference_multi_offset(images, exp_config, multi_offset_matrix, keep_prob, training):
    tmp = exp_config.model_handle(images, training, keep_prob, nlabels=exp_config.nlabels)
    pred = tf.add(tmp, multi_offset_matrix)
    return tf.reshape(pred, (tf.shape(tmp)[0], tf.shape(tmp)[1] * tf.shape(tmp)[2], exp_config.nlabels))




def inference_multi_PCA(images, exp_config, PCA_U, PCA_mean, PCA_sqrtsigma, keep_prob, training):
    '''
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module.
    '''
    tmp = exp_config.model_handle(images, training, keep_prob, nlabels=exp_config.nlabels)

    weights = tf.slice(tmp, [0, 0, 0, 0], [exp_config.batch_size, tf.shape(tmp)[1], tf.shape(tmp)[2],  exp_config.nlabels], name="pca_weights")
    shift = tf.slice(tmp, [0, 0, 0, exp_config.nlabels], [exp_config.batch_size, tf.shape(tmp)[1], tf.shape(tmp)[2], 2])

    pred_list = []

    for k in range(0,exp_config.batch_size):
        w = tf.slice(weights,[k,0,0,0], [1, tf.shape(tmp)[1], tf.shape(tmp)[2],  exp_config.nlabels], name="batch_weight")
        w = tf.squeeze(w)
        w = tf.reshape(w, (tf.shape(tmp)[1] * tf.shape(tmp)[2], exp_config.nlabels)) #FIXME ????
        w = tf.transpose(w)
        w = tf.matmul(PCA_sqrtsigma, w)
        pred = tf.matmul(PCA_U, w)
        pred = tf.add(pred, PCA_mean)

        s = tf.slice(shift,[k,0,0,0], [1, tf.shape(tmp)[1], tf.shape(tmp)[2], 2], name="slice_shift")
        s = tf.squeeze(s)
        s = tf.reshape(s,(tf.shape(tmp)[1] * tf.shape(tmp)[2], 2))

        PCA_shift1 = tf.tile(tf.slice(s, [0, 0], [tf.shape(tmp)[1] * tf.shape(tmp)[2], 1]), [1, exp_config.num_vertices], name="pca_shift_x")
        PCA_shift2 = tf.tile(tf.slice(s, [0, 1], [tf.shape(tmp)[1] * tf.shape(tmp)[2], 1]), [1, exp_config.num_vertices], name="pca_shift_y")

        PCA_shift = tf.concat([PCA_shift1, PCA_shift2], 1, name="batch_pca_shift")

        pca_output = tf.add(pred, tf.transpose(PCA_shift), name="pca_output")

        pred_list.append(tf.transpose(tf.squeeze(pca_output)))

    return tf.stack(pred_list), w, s




def inferencePCA(images, exp_config, PCA_U, PCA_mean, PCA_sqrtsigma, keep_prob, training):
    '''
    Wrapper function to provide an interface to a PCA model from the model_zoo inside of the model module.
    '''
    tmp = exp_config.model_handle(images, training, keep_prob, nlabels=exp_config.nlabels)
    weights = tf.slice(tmp, [0,0], [exp_config.nlabels, exp_config.batch_size], name="pca_weights")
    weights = tf.matmul(PCA_sqrtsigma, weights)

    shift = tf.slice(tmp, [exp_config.nlabels, 0], [2, exp_config.batch_size])

    PCA_shift1 = tf.tile(tf.slice(shift,[0,0], [1, exp_config.batch_size]), [exp_config.num_vertices, 1], name="pca_shift_x")
    PCA_shift2 = tf.tile(tf.slice(shift, [1, 0], [1, exp_config.batch_size]), [exp_config.num_vertices, 1], name="pca_shift_y")
    PCA_shift = tf.concat([PCA_shift1, PCA_shift2], 0, name="pca_shift")

    PCA_output = tf.matmul(PCA_U, weights)
    pca_output_mean = tf.add(PCA_output, PCA_mean)
    pca_output = tf.add(pca_output_mean, PCA_shift, name="pca_output")

    pca_output_x = tf.slice(pca_output, [0,0], [exp_config.num_vertices, exp_config.batch_size], name="pca_output_x")
    pca_output_y = tf.slice(pca_output, [exp_config.num_vertices,0], [exp_config.num_vertices, exp_config.batch_size], name="pca_output_y")

    a = tf.reshape(tf.transpose(pca_output_x), [tf.shape(pca_output_x)[1], 1, -1, 1], name="coordinate_x")
    b = tf.reshape(tf.transpose(pca_output_y), [tf.shape(pca_output_y)[1], 1, -1, 1], name="coordinate_y")

    out = tf.concat((a, b), axis=-1, name="vertices")

    #pca_output = tf.reshape(pca_output, [tf.shape(pca_output)[1], 1, exp_config.num_vertices, 2])

    return out


def loss(logits, labels, exp_config, nlabels, loss_type, weight_decay=0.0):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'weighted_crossentropy'/'crossentropy'/'dice'/'crossentropy_and_dice'
    :param weight_decay: The weight for the L2 regularisation of the network paramters
    :return: The total loss including weight decay, the loss without weight decay, only the weight decay 
    '''

    #labels = tf.one_hot(labels, depth=nlabels)

    with tf.variable_scope('weights_norm') as scope:

        weights_norm = tf.reduce_sum(
            input_tensor = weight_decay*tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

    if loss_type == 'weighted_crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_weighted(logits, labels,
                                                                          class_weights=[0.1, 0.3, 0.3, 0.3])
    elif loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
    elif loss_type == 'crossentropy_and_dice':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels) + 0.2*losses.dice_loss(logits, labels)
    elif loss_type == 'ssd':
        segmentation_loss =  losses.sum_of_squared_dist(logits, labels)
    elif loss_type == 'multi_ssd':
        segmentation_loss = losses.multi_sum_of_squared_dist(logits,labels, exp_config)
    elif loss_type == 'avg_ssd':
        segmentation_loss = losses.average_sum_of_squared_dist(logits,labels, exp_config)
    else:
        raise ValueError('Unknown loss: %s' % loss_type)


    total_loss = tf.add(segmentation_loss, weights_norm)

    return total_loss, segmentation_loss, weights_norm


def predict(images, exp_config):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    logits = exp_config.model_handle(images, training=tf.constant(False, dtype=tf.bool), nlabels=exp_config.nlabels)
    softmax = tf.nn.softmax(logits)
    mask = tf.arg_max(softmax, dimension=-1)

    return mask, softmax


def training_step(loss, optimizer_handle, learning_rate, **kwargs):
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param learning_rate: Learning rate
    :param momentum: Optionally, you can also pass a momentum term to the optimiser. 
    :return: The training operation
    '''


    if 'momentum' in kwargs:
        momentum = kwargs.get('momentum')
        optimizer = optimizer_handle(learning_rate=learning_rate, momentum=momentum)
    else:
        optimizer = optimizer_handle(learning_rate=learning_rate)

    # The with statement is needed to make sure the tf contrib version of batch norm properly performs its updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    return train_op




def evaluation(logits, labels, exp_config, images, nlabels, loss_type):
    '''
    A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the 
    current foreground Dice score, and also writes example segmentations and imges to to tensorboard.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input image mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss without weight decay, the foreground dice of a minibatch
    '''

    #mask = tf.arg_max(tf.nn.softmax(logits, dim=-1), dimension=-1)  # was 3
    #mask_gt = labels

    #tf.summary.image('example_gt', prepare_tensor_for_summary(mask_gt, mode='mask', nlabels=nlabels))
    #tf.summary.image('example_pred', prepare_tensor_for_summary(mask, mode='mask', nlabels=nlabels))
    #tf.summary.image('example_zimg', prepare_tensor_for_summary(images, mode='image'))

    total_loss, nowd_loss, weights_norm = loss(logits, labels, exp_config, nlabels=nlabels, loss_type=loss_type)

    #cdice_structures = losses.per_structure_dice(logits, tf.one_hot(labels, depth=nlabels))
    #cdice_foreground = tf.slice(cdice_structures, (0,1), (-1,-1))

    #cdice = tf.reduce_mean(cdice_foreground)

    return nowd_loss #, cdice


def prepare_tensor_for_summary(img, mode, idx=0, nlabels=None):
    '''
    Format a tensor containing imges or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':

        if img.get_shape().ndims == 3:
            V = tf.slice(img, (idx, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (idx, 0, 0, 10), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':

        if img.get_shape().ndims == 3:
            V = tf.slice(img, (idx, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (idx, 0, 0, 0), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else:
        raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]

    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

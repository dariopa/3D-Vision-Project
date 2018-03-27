# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
import numpy as np

def dice_loss(logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
    '''
    Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
     denotes background and the remaining labels are foreground. 
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :param from_label: First label to evaluate 
    :param to_label: Last label to evaluate
    :return: Dice loss
    '''

    with tf.name_scope('dice_loss'):

        prediction = tf.nn.softmax(logits)

        intersection = tf.multiply(prediction, labels)
        intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

        l = tf.reduce_sum(prediction, axis=[1, 2])
        r = tf.reduce_sum(labels, axis=[1, 2])

        dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

        loss = 1 - tf.reduce_mean(tf.slice(dices_per_subj, (0, from_label), (-1, to_label)))

    return loss


def foreground_dice(logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
    '''
    Pseudo-dice calculated from all voxels (from all subjects) and all non-background labels
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: scalar Dice
    '''

    struct_dice = per_structure_dice(logits, labels, epsilon)
    foreground_dice = tf.slice(struct_dice, (0, from_label),(-1, to_label))

    return tf.reduce_mean(foreground_dice)


def per_structure_dice(logits, labels, epsilon=1e-10):
    '''
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1])
    '''

    ndims = logits.get_shape().ndims

    prediction = tf.nn.softmax(logits)
    hard_pred = tf.one_hot(tf.argmax(prediction, axis=-1), depth=tf.shape(prediction)[-1])

    intersection = tf.multiply(hard_pred, labels)

    if ndims == 5:
        reduction_axes = [1,2,3]
    else:
        reduction_axes = [1,2]

    intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=reduction_axes)  # was [1,2]

    l = tf.reduce_sum(hard_pred, axis=reduction_axes)
    r = tf.reduce_sum(labels, axis=reduction_axes)

    dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

    return dices_per_subj


def pixel_wise_cross_entropy_loss(logits, labels):
    '''
    Simple wrapper for the normal tensorflow cross entropy loss 
    '''

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss


def pixel_wise_cross_entropy_loss_weighted(logits, labels, class_weights):
    '''
    Weighted cross entropy loss, with a weight per class
    :param logits: Network output before softmax
    :param labels: Ground truth masks
    :param class_weights: A list of the weights for each class
    :return: weighted cross entropy loss
    '''

    n_class = len(class_weights)

    flat_logits = tf.reshape(logits, [-1, n_class])
    flat_labels = tf.reshape(labels, [-1, n_class])

    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    weight_map = tf.multiply(flat_labels, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)

    return loss

def sum_of_squared_dist(logits, labels):
    return tf.reduce_sum(tf.pow(logits - labels, 2))

def average_sum_of_squared_dist(logits, labels, exp_config):
    sum = tf.reduce_sum(tf.pow(logits - labels, 2))
    (n, m) = exp_config.image_size
    batches = tf.shape(logits)[0]
    dev = tf.to_float(tf.multiply(n,m))
    dev = tf.multiply(tf.to_float(batches), dev)
    return tf.divide(sum, dev)

def multi_sum_of_squared_dist(logits, labels, exp_config): #FIXME


    sum = 0
    for i in range(0, len(logits)):
        sum = sum + tf.reduce_sum(tf.pow(np.squeeze(logits[:,i,:,:]) - labels, 2))



import model_zoo
import tensorflow as tf

experiment_name = 'shallow_FCN_UKBB'

# Model settings
model_handle = model_zoo.shallow_FCN
do_pca = 0

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (200,200)#(212, 212)#(60,60)#
target_resolution =  (1.8269,1.8269) #(1.36719, 1.36719)(1.8,1.8)#
nlabels = 100
num_vertices = 50
start_slice = 4
end_slice = 4


# Training settings
batch_size = 5
learning_rate = 0.0001
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'avg_ssd'  # crossentropy/weighted_crossentropy/dice/ssd for vertices
keep_probability = 1
do_pca = 1
multi_pixel_prediction = 0
multi_offset_prediction = 0

# Augmentation settings !!!!!!!!!!!!!!to use augmentation, one needs to augment the vertices correspondingly - write procedures
augment_batch = False
do_rotations = True
do_scaleaug = False
do_fliplr = False

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_epochs = 3333333322
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100
import model_zoo
import tensorflow as tf

experiment_name = 'CL5_DL1'

# Model settings
model_handle = model_zoo.CL5_DL1

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (212, 212)
target_resolution = (1.36719, 1.36719)
nlabels = 50
num_vertices = 50
start_slice = 2
end_slice = 5

# Training settings
batch_size = 5
learning_rate = 0.0001
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0
momentum = None
loss_type = 'ssd'  # crossentropy/weighted_crossentropy/dice/ssd for vertices
keep_probability = 1

# Augmentation settings !!!!!!!!!!!!!!to use augmentation, one needs to augment the vertices correspondingly - write procedures
augment_batch = False
do_rotations = True
do_scaleaug = False
do_fliplr = False

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_epochs = 7000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100
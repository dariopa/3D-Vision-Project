# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

project_root = '../'
data_root = '../ACDC_challenge_20170617/'
test_data_root = '../ACDC_challenge_testdata/'
local_hostnames = ['bmicdl03']  # used to check if on cluster or not

##################################################################################

log_root = os.path.join(project_root, 'acdc_logdir/')
if not os.path.isdir(log_root):
    os.makedirs(log_root)
out_data_root = os.path.join(project_root, 'Prediciton_unaligned_data/')
if not os.path.isdir(out_data_root):
    os.makedirs(out_data_root)
    
preproc_folder = os.path.join(project_root,'preproc_data_augmented')

def setup_GPU_environment():

    logging.warning('No setup proceedure defined')

    # Local setup for Sun Grid Engine
    # hostname = socket.gethostname()
    # print('Running on %s' % hostname)
    # if not hostname in local_hostnames:
    #     logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
    #     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
    #     logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])


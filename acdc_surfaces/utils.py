# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import nibabel as nib
import numpy as np
import os
import glob
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
import matplotlib.path as pth
#import matplotlib.figure as fig
#import tfplot
import io
from scipy.io import loadmat
from sklearn.cluster import KMeans

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))

def load_png(file_path):
    return misc.imread(file_path)

def load_vrtx(file_path):

    tab = np.loadtxt(file_path)
    num_vrtx = int(tab[0][0])
    print('num_vrtx', num_vrtx)

    return tab[1:num_vrtx+1, :]

def load_vrtx_column(file_path):

    tab = np.loadtxt(file_path)
    num_vrtx = int(tab[0][0])
    print('num_vrtx', num_vrtx)

    column = tab[1:num_vrtx+1, :]
    column = np.transpose(column)
    column = np.reshape(column, (1, num_vrtx *2))

    return column #np.transpose(column)


def load_coeff(file_path, n_coeff):
    print('num_coeff', n_coeff)
    tab = loadmat(file_path)['coeff']

    return tab[1:n_coeff + 1]

def save_plot(points, outName, axis_limits):
    plt.figure()
    plt.plot(points[:,1], points[:,0])
    plt.axis(axis_limits)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(outName)

#def summary_plot(points):
#    fig, ax = tfplot.subplots()
#    ax.plot(points[:,1], points[:,0])
#    return tfplot.to_summary(fig, tag='Prediction')

def gen_plot(points, axis_limits):
    plt.figure()
    plt.plot(points[:, 1], points[:, 0])
    plt.axis(axis_limits)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def gen_2plot(points1, points2, axis_limits):
    plt.figure()
    plt.plot(points1[:, 1], points1[:, 0], 'r-', points2[:, 1], points2[:, 0], 'b-')
    plt.axis(axis_limits)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def gen_2plot_column(points1, points2, num_vertices, axis_limits):
    plt.figure()
    plt.plot(points1[0,0:num_vertices], points1[0,num_vertices:], 'r-', points2[0,0:num_vertices], points2[0,num_vertices:], 'b-')
    plt.axis(axis_limits)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def gen_bbox(image, GTlabel, prediction, axis_limits):
    plt.figure()

    plt.imshow(image)
    plt.plot([GTlabel[0,1], GTlabel[1,1], GTlabel[1,1], GTlabel[0,1], GTlabel[0,1]], [GTlabel[0,0], GTlabel[0,0], GTlabel[1,0], GTlabel[1,0],GTlabel[0,0] ], 'r-',
             [prediction[0, 1], prediction[1, 1],
              prediction[1, 1], prediction[0, 1],
              prediction[0, 1]], [prediction[0, 0], prediction[0, 0], prediction[1, 0], prediction[1, 0], prediction[0, 0]],  'b-')

    plt.axis(axis_limits)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def save_two_plots(points1, points2, outName, axis_limits):
    plt.figure()
    plt.plot(points1[:, 1], points1[:, 0],'r-',points2[:, 1], points2[:, 0], 'b-' )
    plt.axis(axis_limits)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(outName)

def overlay_img_2plots(points1, points2, img, outName, axis_limits):
    plt.figure()
    plt.plot(points1[:, 1], points1[:, 0], 'r-', points2[:, 1], points2[:, 0], 'b-')
    plt.axis(axis_limits)
    plt.imshow(img, zorder=0, cmap='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(outName)

def overlay_img_3plots(points1, points2, points3, img, outName, axis_limits):
    plt.figure()
    plt.plot(points1[:, 1], points1[:, 0], 'r-', points2[:, 1], points2[:, 0], 'b-', points3[:, 1], points3[:, 0], 'g-')
    plt.axis(axis_limits)
    plt.imshow(img, zorder=0, cmap='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(outName)

def overlay_bboxes(img, points, pointsGT, outName, axis_limits):
    plt.figure()
    plt.plot([pointsGT[0,1], pointsGT[1,1], pointsGT[1,1], pointsGT[0,1], pointsGT[0,1]], [pointsGT[0,0], pointsGT[0,0], pointsGT[1,0], pointsGT[1,0],pointsGT[0,0] ], 'r-',
             [points[0, 1], points[1, 1], points[1, 1], points[0, 1], points[0, 1]], [points[0, 0], points[0, 0], points[1, 0], points[1, 0], points[0, 0]],  'b-')
    plt.axis(axis_limits)
    plt.imshow(img, zorder=0, cmap='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(outName)


def adjust_image_range(image, min_r, max_r):

    img_o = np.float32(image.copy())

    min_i = img_o.min()
    max_i = img_o.max()

    img_o = (img_o - min_i)*((max_r - min_r)/(max_i - min_i)) + min_r

    return img_o

def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)

def normalise_image(image):
    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)

def resize_image(img, size, interp=cv2.INTER_LINEAR):

    img_resized = cv2.resize(img, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
    return img_resized

def rescale_image(img, scale, interp=cv2.INTER_LINEAR):

    curr_size = img.shape
    new_size = (int(float(curr_size[0])*scale[0]+0.5), int(float(curr_size[1])*scale[1]+0.5))
    img_resized = cv2.resize(img, (new_size[1], new_size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
    return img_resized


def create_segmentation(vertices, image_size):

    # create polygon
    polygon = pth.Path(vertices)

    # create mesh
    img_shape = image_size
    idx = np.indices(img_shape).reshape((2, -1))

    # create segmentation
    inpoints = polygon.contains_points(idx.T).reshape(img_shape)

    return inpoints


def computeDICE(segmentation, GT):
    xDim = segmentation.shape[0]
    yDim = segmentation.shape[1]

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for y in range(yDim):
        for x in range(xDim):
            x1 = GT[x,y]
            y1 = 1-x1
            x2 = segmentation[x,y]
            y2 = 1-x2

            tn = tn + min(y1,y2)
            if (x1 > x2):
                fn = fn + x1 - x2
            if (x2 > x1):
                fp = fp + x2 - x1
            tp = tp + min(x1, x2)

    return (2*tp/(2*tp + fp + fn)), tp, fp, tn, fn


def getPC():
    return 0

def get_offset_matrix(exp_config):
    (n,m) = exp_config.image_size
    i = np.linspace(0, n-1, n)
    j = np.linspace(0, m-1, m)
    jm, im = np.meshgrid(j,i)

    im = np.tile(im,(exp_config.num_vertices, 1 ,1))
    jm = np.tile(jm,(exp_config.num_vertices, 1, 1))

    out = np.concatenate((im,jm),axis = 0)

    return np.swapaxes(out,0, -1)

def mean_prediction(pred, exp_config):
    final_pred = []
    (point, vertices) = np.shape(pred)
    vertices = int(np.floor(vertices / 2))

    for i in range(0,vertices):
        x = np.expand_dims(pred[:,i], axis=1)
        y = np.expand_dims(pred[:,i+vertices], axis=1)
        mat = np.concatenate((x,y),axis=1 )
        kmeans = KMeans(n_clusters=1, random_state=0).fit(mat)
        final_pred.append(kmeans.cluster_centers_)
    return final_pred
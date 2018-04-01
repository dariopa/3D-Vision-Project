
import scipy.misc as misc
import numpy as np
import utils
import image_utils
import matplotlib.pyplot as plt

file = '/home/tothovak/work/CSR/Data/UKBB/1000376/tsc_sa_crop.nii.gz'


img_dat = utils.load_nii(file)
img = img_dat[0].copy()
img = np.array(img)
img = image_utils.normalise_image(img)


print(img_dat[2].structarr['srow_x'][0:3])
S = np.array((img_dat[2].structarr['srow_x'][0:3],
              img_dat[2].structarr['srow_y'][0:3],
              img_dat[2].structarr['srow_z'][0:3]))

#S = np.concatenate((srow_x, srow_y, srow_z), 1)
print(S)

mask_shift = np.array((img_dat[2].structarr['qoffset_x'],
              img_dat[2].structarr['qoffset_y'],
              img_dat[2].structarr['qoffset_z']))

print(mask_shift)

print (img_dat[2].structarr['dim'])
#
# for i in range(img.shape[2]):
#     print(i)
#     slice = np.squeeze(img[:,:,i,1])
#     print(slice.shape[0])
#     print(slice.shape[1])
#     slicename = str(i) +'.png'
#     print(slicename)
# #    misc.imsave(slice, slicename)
#     plt.imshow(slice)
#     plt.show()
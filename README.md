# 3D Vision Project: Surface Reconstruction in Medical Imaging: Data and CNNs
This repository contains code (Matlab) to preprocess MRI images for cardiac segmentation networks and a training framework proposed by Baumgarnter et al. which is publicly available on github. 

The original framework is available by typing

``` git clone https://github.com/baumgach/acdc_segmenter.git ```

To clone our repository, type

``` git clone https://github.com/dariopa/3D-Vision-Project.git```

## Installing required Python packages

Create an environment with Python 3.4. If you use virutalenv it 
might be necessary to first upgrade pip (``` pip install --upgrade pip ```).

Next, install the required packages listed in the `requirements.txt` file:

``` pip install -r requirements.txt ```

Then, install tensorflow:

``` pip install tensorflow==1.2 ```
or
``` pip install tensorflow-gpu==1.2 ```

depending if you are setting up your GPU environment or CPU environment. The code was also
tested with tensorflow 1.1 if for some reason you prefer that version. Tensorflow 1.3 is currently causing
trouble on our local machines, so we couldn't test this version yet. 

WARNING: Installing tensorflow before the requirements.txt will lead to weird errors while compiling `scikit-image` in `pip install -r requirements`. Make sure you install tensorflow *after* the requirements. 
If you run into errors anyways try running `pip install --upgrade cython` and then reruning `pip install -r requirements.txt`. 


## Download the ACDC challenge data

If you don't have access to the data already you can sign up and download it from this [webpage](http://acdc.creatis.insa-lyon.fr/#challenges).

The cardiac segmentation challenge and the data is described in detail [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

## How to run the code:

0) pip install -r requirements.txt in the folder acdc_surfaces
1) First, run acdc_data.py in the folder acdc_surfaces -> will generate data to import into matlab
2) Run Unaligned_MRI.m
3) Run Unaligned_SURF.m
4) Run Aligned_all.m
5) Run data.py in folder acdc_surfaces -> will generate data to import into NN
6) run train_*.py
7) run inference_*.py -> final score 

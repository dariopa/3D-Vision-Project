# 3D Vision Project: Surface Reconstruction in Medical Imaging: Data and CNNs
This repository contains code (Matlab) to preprocess MRI images for cardiac segmentation networks and a training framework (Python) proposed by Baumgarnter et al. which is publicly available on github. 

Authors:
- Dario Panzuto ([email](mailto:dario.panzuto@gmail.com))
- Cornel Frigoli ([email](mailto:cfrigoli@ethz.ch))
- Michelle Woon ([email](mailto:woonmi@ethz.ch))

The original framework is available by typing

``` git clone https://github.com/baumgach/acdc_segmenter.git ```

To clone our repository, type

``` git clone https://github.com/dariopa/3D-Vision-Project.git```

## Requirements 

- Python 3.4 (also tested with 3.6.3)
- Tensorflow >= 1.0 (tested with 1.1.0, and 1.2.0)
- The remainder of the requirements are given in `requirements.txt`

## Installing required Python packages

Create an environment with Python 3.6. If you use virutalenv it 
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

## Known issues

- If `pip install -r requirements.txt` fails while compiling `scikit-image`, try the following:
    - Make sure you install `tensorflow` _after_ the `requirements.txt`
    - If that didn't solve the issue, try running `pip install --upgrade cython` seperately, and then run `pip install -r requirements.txt` again.
     
- There seems to be an issue compiling scikit-image when using Python 3.5. If this is happening make sure you are using Python 3.4. 

## Download the ACDC challenge data

If you don't have access to the data already you can sign up and download it from this [webpage](http://acdc.creatis.insa-lyon.fr/#challenges).

Store the data in a folder called ``` Data ```. Check that the path is implemented correctly in the code-files. 

The cardiac segmentation challenge and the data is described in detail [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).

## How to run the code from your computer:
If you wish to compare the unaligned versus the aligned data, do the following:

1) Open the `config/system.py` and edit all the paths there to match your system.
2) First, type ```python acdc_data.py``` in the folder `acdc_surfaces`. This will generate a .hdf5-file.
3) Run Unaligned_MRI.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Unaligned_MRI;quit;" ``` in your shell. 
4) Run Unaligned_SURF.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Unaligned_SURF;quit;" ``` in your shell. 
5) Run Aligned_all.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Aligned_all;quit;" ``` in your shell. 
6) Type ```python data.py``` in folder `acdc_surfaces`. This will generate a new .hdf5-file with data ready to be fed in the network. Make sure the paths match the dataset you want to analyse (line 72 - 100). 
7) Open the `experiments/Aligned_*.py`, select the experiment you want to run, adapt the parameters and save the file. Example: `experiments/Aligned_CL9_DL1.py`
8) Now you can open start your training by typing ``` python train_*.py```. Example: ``` python train_aligned_CL9_DL1.py```. Make sure you comment line 10 (SGE_GPU environmental variable) if you run your code locally and not on a GPU. 

WARNING: When you run the code on CPU, you need around 12 GB of RAM. Make sure your system is up to the task. If not you can try reducing the batch size, or simplifying the network. 

7) run ```python inference_*.py```. Example: ```python inference_aligned.py```. Make sure you edited your parameters correctly (line 15 to line 27). 
8) Go in the folder `Prediction_Data`. There you will find your results. 

If you wish to analyse augmented versus unaugmented datasets, do the following: 

1) SAME AS ABOVE
2) SAME AS ABOVE
3) Run Not_Aug_Unaligned_MRI.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Not_Aug_Unaligned_MRI;quit;" ``` in your shell. 
4) Run Not_Aug_Unaligned_SURF.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Not_Aug_Unaligned_SURF;quit;" ``` in your shell. 
5) Run Not_Aug_Aligned_all.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Not_Aug_Aligned_all;quit;" ``` in your shell. 
6) SAME AS ABOVE
7) SAME AS ABOVE
8) SAME AS ABOVE


If you wish to analyse uncropped versus cropped datasets, do the following: 

1) SAME AS ABOVE
2) SAME AS ABOVE
3) Run Unaligned_MRI.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Unaligned_MRI;quit;" ``` in your shell. 
4) Run Unaligned_SURF.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Unaligned_SURF;quit;" ``` in your shell. 
5) Run Aligned_all_crop.m on Matlab. You can also type ``` matlab -nodesktop -nosplash -r "Aligned_all_crop;quit;" ``` in your shell. 
6) SAME AS ABOVE
7) SAME AS ABOVE
8) SAME AS ABOVE


## CRISM Classifier
### Official Repository for the CRISM Classifier VAE Hybrid Model developed by Platt et al.

## Introduction:
This repo contains the package to apply the Hybrid Model for Mineral Mapping on Mars (HM4) to denoise and classify CRISM L sensor cubes.
The code offers the following functionality:
* Load and preprocess CRISM L sensor cubes
* Train the Hybrid Mineral Mapper for Mars model (HM4)
* Apply the trained model to denoise and classify CRISM L sensor cubes
* Evaluate the model's performance
* Visualise classification results
* Visualise denoised results

#### Example classification
![The Western Fan of Jezero Crater as classified by HM4](https://github.com/rob-platt/HM4/blob/main/data/HRL_example_image.png?raw=true)

## CAML

If you don't want to mess around with code, and just want to use HM4 yourself, check out our companion application [CAML](https://rob-platt.github.io/CAML/)!

## Installation:
To install the package, clone the repository and run the following command in the root directory:

#### Using `uv`:
If you use [uv](https://github.com/astral-sh/uv) for dependency management, install the dependencies with:
```bash
uv sync
```

#### Using `pip`:
Alternatively, with pip, install the dependencies with:
```bash
pip install -r requirements.txt
```

#### Using `conda`:
Alternatively, with conda, install the dependencies with:
```bash
conda create -n hm4
conda activate hm4
conda install pip
pip install -r requirements.txt
```
## Usage:

To test out the capabilities of HM4 on your area/mineralogy of interest, try the tutorial in notebooks/HM4_tutorial.ipynb 

## Data

### Basic Usage

To use HM4, you will need the following data:
- The bland pixel dataset from [Plebani et al. (2022)](https://zenodo.org/records/13338091) is required to ratio images as a preprocessing step. This can be downloaded from the link above, and should be placed in the data folder with the following structure:  
| data/  
| ----/CRISM_ML/  
| -------- CRISM_bland_unratioed.mat  
The following bash script will do this for you:
    ```bash
    mkdir data
    cd data
    mkdir CRISM_ML
    cd CRISM_ML
    wget https://zenodo.org/records/13338091/files/CRISM_bland_unratioed.mat
    cd ..
    cd ..
    ```
- A CRISM L sensor image. Must be L sensor only, J (joint) or S (short) images are currently not supported. The standard geometric and photometric corrections, as well as the "volcano-scan" correction, must be applied before use. Data can be collected from the NASA PDS, and the corrections applied through ENVI/IDL CAT extension, or by using MarsSI. 

### Retraining
To retrain HM4 from scratch, you will need the following data:
- The CRISM_ML Toolkit ratioed training dataset (see above Zenodo repository)
- All of the CRISM images used in that training dataset, from either the PDS or MarsSI, with the standard corrections applied. Each image should be in a folder named after the image hexcode, e.g. for HRL000040FF  
    | data/  
    | ----/HRL000040FF  
    | --------/HRL000040FF_07_IF183L_TRR3.img  
    | --------/HRL000040FF_07_IF183L_TRR3.lbl  
    | --------/HRL000040FF_07_IF183L_TRR3.hdr

## Tests
The package includes a test suite that can be run using the following command:
```bash
pytest
```

## Training from Scratch
To train the HM4 model from scratch, first you must collate the required training and testing data.
This can be done by following these steps:
* Download both the mineral and bland pixel datasets from [Plebani et al. (2022)](https://zenodo.org/records/13338091). 
* Download the imagery used for the Plebani et al. (2022) datasets from [MarsSI](https://marssi.univ-lyon1.fr/wiki/Home). The _CAT_corr.img files must be used, but must be renamed to match the original .img filenames. 
* Run the bland_pixel_finder.py and mineral_data_collation.py scripts in the /scripts folder. This will extract all relevant pixels from the raw images and save them as a single .json file each.
* Then run the ratio_pixels.py file in the /scripts folder to ratio the images using the bland pixel data and mineral data .json files created in the previous step.
* Run the train_test_holdout_split.py script in the /scripts folder to split the ratioed data into training, validation, testing, and holdout sets.
* Run the train_HM4.py script in the /scripts folder to train the HM4 model.

## Benchmarking 
A notebook has been provided to demonstrate the performance of the Heirarchical Bayesian Model (HBM) from [1] on the same data as the HM4 model. This can be found in the /notebooks/HBM_benchmark folder.

For the Random Forest benchmark from [2], scripts for the additional preprocessing have been included under /scripts/dhoundiyal_benchmark. First run traintest_preprocessing.py, then holdout_preprocessing.py. Then run the mineral_data_aggregration.py. These will take the ratioed dataset from HM4, apply the denoising and continuum removal as in [2], and then split the data based on K-Medoids clustering into new training, and testing sets, again as in [2]. Training for the model can be done using the /notebooks/RandomForest_benchmark/original_training.ipynb notebook to match the results in [2], and then retrain_platt_data.ipynb to train on the same classes as HM4. The holdout_data_inference.ipynb notebook can then be used to evaluate the model on the holdout set.

## Acknowledgement
This code is part of Robert Platt's PhD work and you can [visit his GitHub repository](https://github.com/rob-platt) where the primary version of this code resides. The work was carried out under the supervision of [Cédric John](https://github.com/cedricmjohn) and all code from the research group can be found in the [John Lab GitHub repository](https://github.com/johnlab-research).

<a href="https://www.john-lab.org">
<img src="https://www.john-lab.org/wp-content/uploads/2023/01/footer_small_logo.png" style="width:220px">
</a>

## References

[1] Plebani, E., Ehlmann, B. L., Leask, E. K., Fox, V. K., & Dundar, M. M. (2022). A machine learning toolkit for CRISM image analysis. Icarus, 376, 114849. https://doi.org/10.1016/j.icarus.2021.114849

[2] Dhoundiyal, S., Dey, M. S., Singh, S., Arun, P. V., Thangjam, G., & Porwal, A. (2025). Explainable Machine Learning for Mapping Minerals From CRISM Hyperspectral Data. Journal of Geophysical Research: Machine Learning and Computation, 2(2), e2024JH000391. https://doi.org/10.1029/2024JH000391


### Licence
This package is released under the MIT licence.


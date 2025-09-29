# Take the ratioed data and apply the additional preprocessing steps from [1]. 
# This is only for benchmarking the model from [1] against HM4, and is not necessary for 
# training or running inference using HM4 itself. 
# One caution - you need a large amount of memory (>48GB) to compute this.
# [1] Dhoundiyal, S., Dey, M. S., Singh, S., Arun, P. V., Thangjam, G., & Porwal, A. (2025).  
# Explainable Machine Learning for Mapping Minerals From CRISM Hyperspectral Data.  
# Journal of Geophysical Research: Machine Learning and Computation, 2(2), e2024JH000391. https://doi.org/10.1029/2024JH000391

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pysptools.spectro import convex_hull_removal
import os
from n2n4m.wavelengths import PLEBANI_WAVELENGTHS
from sklearn_extra.cluster import KMedoids

holdout_data_path = # Insert path to the holdout data JSON file
dhoundiyal_wavelength_path = # Wavelengths used by [1], found here: https://doi.org/10.5281/zenodo.15036700
output_dir = # Choose your output directory

xy_data = pd.read_json(holdout_data_path)
dhoundiyal_wavelengths = np.load(dhoundiyal_wavelength_path)

# split the dataset into classes, ready to subsample
x_data_per_class = []
xy_data = xy_data.sort_values("Pixel_Class", axis=0, ascending=True) # sorted to iterate over classes in order

for pixel_class in xy_data["Pixel_Class"].unique():
    mineral_data = xy_data[xy_data["Pixel_Class"] == pixel_class]
    spectra = mineral_data.iloc[:, :-3].values
    x_data_per_class.append(spectra)

dhoundiyal_idx_plebani = [] # the indices of the dhoundiyal wavelengths in the plebani band list
i, j = 0, 0

for band in PLEBANI_WAVELENGTHS:
    if band == dhoundiyal_wavelengths[j]:
        dhoundiyal_idx_plebani.append(i)
        j += 1
    i += 1
    if j == 236:
        break

x_data_continuum_removed = []

for mineral_class in x_data_per_class:
    # clip unused bands from Plebani
    clipped_mineral_class = mineral_class[:, dhoundiyal_idx_plebani]
    
    cont_removed_class_arr = np.zeros_like(clipped_mineral_class)
    # iterate over each pixel for continuum removal
    for idx, pixel in enumerate(mineral_class):
        cont_removed_class_arr[idx] = convex_hull_removal(pixel, dhoundiyal_wavelengths)[0]
    # l2 normalization
    l2_norm = np.linalg.norm(cont_removed_class_arr, axis=-1)
    normalized_spectra = cont_removed_class_arr.T / l2_norm
    x_data_continuum_removed.append(normalized_spectra.T)

for idx, mineral_class in zip(xy_data["Pixel_Class"].unique(), x_data_continuum_removed):
    x_data_mineral = pd.DataFrame(mineral_class)
    x_data_ancillary = xy_data[xy_data["Pixel_Class"] == idx].reset_index().iloc[:, -3:]

    holdout_mineral_data = pd.concat([x_data_mineral, x_data_ancillary], axis=1)
    holdout_mineral_data.to_json(os.path.join(output_dir, f"{idx}_mineral_holdout_data.json"))

# Take the ratioed data and apply the additional preprocessing steps from [1]. 
# This is only for benchmarking the model from [1] against HM4, and is not necessary for 
# training or running inference using HM4 itself. 
# One caution - you need a large amount of memory (>48GB) to compute this.
# [1] Dhoundiyal, S., Dey, M. S., Singh, S., Arun, P. V., Thangjam, G., & Porwal, A. (2025).  
# Explainable Machine Learning for Mapping Minerals From CRISM Hyperspectral Data.  
# Journal of Geophysical Research: Machine Learning and Computation, 2(2), e2024JH000391. https://doi.org/10.1029/2024JH000391

import pandas as pd
import numpy as np
from pysptools.spectro import convex_hull_removal
import os
from sklearn_extra.cluster import KMedoids
from n2n4m.wavelengths import PLEBANI_WAVELENGTHS


# Insert the paths to ratioed training, validation, test data JSON files
train_data_path = #
val_data_path = #
test_data_path = #
# Wavelengths used by [1], found here: https://doi.org/10.5281/zenodo.15036700
dhoundiyal_wavelength_path = #
# Choose your output directory
output_dir_path = #

xy_train = pd.read_json(train_data_path)
xy_val = pd.read_json(val_data_path)
xy_test = pd.read_json(test_data_path)
xy_data = pd.concat([xy_train, xy_val, xy_test])

dhoundiyal_wavelengths = np.load(dhoundiyal_wavelength_path)

# split the dataset into classes, ready to subsample
x_data_per_class = []
xy_data = xy_data.sort_values("Pixel_Class", axis=0, ascending=True) # sorted to iterate over classes in order

for pixel_class in xy_data["Pixel_Class"].unique():
    mineral_data = xy_data[xy_data["Pixel_Class"] == pixel_class]
    spectra = mineral_data.iloc[:, :-3].values
    x_data_per_class.append(spectra)

# the indices of the dhoundiyal wavelengths in the plebani band list
dhoundiyal_idx_plebani = [] 
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


# zip this way rather than enumerate to avoid class 21 which has no samples
for idx, mineral_class in zip(xy_data["Pixel_Class"].unique(), x_data_continuum_removed):
    num_clusters = 1000
    if len(mineral_class) == 0:
        print(f"Skip class {idx} for having no samples")
        continue
    if len(mineral_class) < 1000:
        num_clusters = int(len(mineral_class) / 2)
    xy_mineral_data = xy_data[xy_data["Pixel_Class"] == idx]
    k_medoids = KMedoids(n_clusters=num_clusters, random_state=42).fit(mineral_class)
    # get the medoid centroid indices
    medoid_idx = k_medoids.medoid_indices_
    # mask the centroids
    spectra_idx_arr = np.arange(mineral_class.shape[0])
    spectra_medoid_mask = np.isin(spectra_idx_arr, medoid_idx)
    # slice spectra of centroids
    train_mineral_spectra = mineral_class[spectra_medoid_mask]
    # get labels + ancillary data from df for train set
    train_mineral_ancillary = xy_mineral_data.iloc[spectra_medoid_mask, -3:].reset_index()
    # get inverse of centroids (i.e. all other spectra) for test set
    test_mineral_spectra = mineral_class[~spectra_medoid_mask]
    test_mineral_ancillary = xy_mineral_data.iloc[~spectra_medoid_mask, -3:].reset_index()

    # concat to new df and write out
    train_mineral_spectra = pd.DataFrame(train_mineral_spectra)
    train_mineral_df = pd.concat([train_mineral_spectra, train_mineral_ancillary], axis=1)
    train_mineral_df.to_json(os.path.join(output_dir_path, f"{idx}_mineral_train_data.json"))
    
    test_mineral_spectra = pd.DataFrame(test_mineral_spectra)
    test_mineral_df = pd.concat([test_mineral_spectra, test_mineral_ancillary], axis=1)
    test_mineral_df.to_json(os.path.join(output_dir_path, f"{idx}_mineral_test_data.json"))
    print(f"Finished class {idx}", flush=True)
# Script to find suitable bland pixels for pixel ratioing in preprocessing
# for the classifier. This uses the model from [1] to identifiy the bland
# pixels.
# [1] Plebani, E., Ehlmann, B. L., Leask, E. K., Fox, V. K., & Dundar, M. M.
# (2022). A machine learning toolkit for CRISM image analysis.
# Icarus, 376, 114849. https://doi.org/10.1016/j.icarus.2021.114849

import argparse

import n2n4m.preprocessing as n2n4m_preprocessing
import numpy as np
import pandas as pd
import glob
import os
from crism_ml.preprocessing import filter_bad_pixels
from crism_ml.preprocessing import remove_spikes_column
from crism_ml.preprocessing import replace
from crism_ml.train import compute_bland_scores
from crism_ml.train import feat_masks
from crism_ml.train import train_model_bland
from n2n4m.io import load_image_from_shortcode
from n2n4m.wavelengths import ALL_WAVELENGTHS
from n2n4m.wavelengths import PLEBANI_WAVELENGTHS


parser = argparse.ArgumentParser(
    description="""Find suitable bland pixels for
    pixel ratioing in preprocessing for the classifier."""
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
    help="Output directory for the bland pixel dataframes.",
)
parser.add_argument(
    "-d",
    "--data_file",
    type=str,
    required=True,
    help="Path to JSON file of the mineral pixel data.",
)
parser.add_argument(
    "-i",
    "--image_dir",
    type=str,
    help="""Directory containing the raw CRISM images to
            find bland pixels for.""",
)
parser.add_argument(
    "-m",
    "--model_data_dir",
    type=str,
    help="""Directory containing the data to train the bland pixel model.
            This is the CRISM_ML Bland Unratioed Dataset and can be found
            here: https://zenodo.org/records/13338091""",
)
parser.add_argument(
    "-n",
    "--n_pixels",
    type=int,
    default=3,
    help="""Number of bland pixels to find for each mineral pixel.
            Default is 3.""",
)
parser.add_argument(
    "-w",
    "--window",
    type=int,
    default=50,
    help="""Window size to search for bland pixels around each mineral pixel.
            Default is 50.""",
)

arguments = parser.parse_args()
args = vars(arguments)

CRISM_ML_DATA_DIR = args["model_data_dir"]
RAW_IMAGE_DIR = args["image_dir"]
OUTPUT_DIR = args["output_dir"]

window = args["window"]
n_pixels = args["n_pixels"]

fin0, fin = feat_masks()  # fin0 for bland pixels, fin for non-bland pixels
bland_model = train_model_bland(
    CRISM_ML_DATA_DIR, fin0
)  # Train bland model using the unratioed bland pixel dataset

mineral_dataset = n2n4m_preprocessing.load_dataset(args["data_file"])

# Number of mineral pixels that no suitable bland pixels could be found for.
bad_pixel_count = 0

for image_id in mineral_dataset["Image_Name"].unique():
    print(f"Processing image {image_id}")
    # Load the image
    full_image = load_image_from_shortcode(
        mineral_dataset[mineral_dataset["Image_Name"] == image_id],
        RAW_IMAGE_DIR,
    )
    im_shape = full_image.shape[:2]  # spatial dims only, not including bands
    num_bands = full_image.shape[-1]

    full_image = full_image.reshape(-1, num_bands)  # Flatten the image
    # Boolean mask of whether each band in the L sensor
    # is included in this project
    # fmt: off
    wavelength_mask = [
        True if band in PLEBANI_WAVELENGTHS
        else False for band in ALL_WAVELENGTHS
    ]
    plebani_bands_spectra = full_image[:, wavelength_mask].reshape(
        *im_shape, len(PLEBANI_WAVELENGTHS)
    )
    # fmt: on
    spectra, bad_pixel_mask = filter_bad_pixels(plebani_bands_spectra)
    # Remove spikes using a median filter with window size 3,
    # removing spikes larger than 5 std dev. calculated per column
    despiked_spectra = remove_spikes_column(
        spectra.reshape(*im_shape, -1), size=3, sigma=5
    ).reshape(spectra.shape)
    # Compute the blandness score for each pixel
    bland_scores = compute_bland_scores(
        despiked_spectra.reshape(-1, len(PLEBANI_WAVELENGTHS)),
        (bland_model, fin0),
    )
    # replace the blandness score of any bad pixel by -inf,
    # so that when it comes to bland pixel selection, it will not be selected.
    screened_bland_scores = replace(
        bland_scores.reshape(im_shape), bad_pixel_mask, -np.inf
    ).reshape(im_shape)

    identified_useful_bland_pixels = []
    # fmt: off
    for _, row in mineral_dataset[
        mineral_dataset["Image_Name"] == image_id
    ].iterrows():
        # fmt: on
        sample_x_coord = row["Coordinates"][0]
        sample_y_coord = row["Coordinates"][1]

        sample_row = sample_y_coord - 1
        sample_col = sample_x_coord - 1

        # Adjust window to ensure it doesn't go outside the image
        neg_window = min(sample_row, window)
        pos_window = min(im_shape[0] - sample_row, window)

        idx_best_bland_pixels = []
        # Get the index positions of the bland pixels in the window
        # sorted by blandness score
        sorted_idx = np.argsort(
            screened_bland_scores[
                sample_row - neg_window: sample_row + pos_window, sample_col
            ]
        )
        sorted_idx = sorted_idx[
            ::-1
        ]  # Reverse order so that the most bland pixels are first

        # Load the full image to get the spectra of the bland pixels

        suitable_pixels = 0
        for idx in sorted_idx:
            if suitable_pixels == n_pixels:
                break
            # Check that the pixel has not been imputed,
            # and that it is not the sample pixel
            # fmt: off
            if (
                screened_bland_scores[
                    sample_row - neg_window + idx,
                    sample_col
                ] != -np.inf
                and idx - neg_window != 0
            ):
                # Ensure that no bad values are in the spectra selected.
                # Most are caught by filter_bad_pixels,
                # but only for the 350 bands used by Plebani et al.,
                # not the extras used in this project
                if not np.any(
                    plebani_bands_spectra[
                        sample_row - neg_window + idx,
                        sample_col, :]
                    > 1000
                ):
                    # 1000 catches any spurious values,
                    # as well as known invalid values (65535.0)
                    idx_best_bland_pixels.append(idx)
                    suitable_pixels += 1
        # fmt: on
        if len(idx_best_bland_pixels) == 0:
            bad_pixel_count += 1
            continue

        coords_best_bland_pixels = [
            [sample_x_coord, sample_y_coord - neg_window + y]
            for y in idx_best_bland_pixels
        ]

        bland_pixel_spectra = []
        for coords in coords_best_bland_pixels:
            bland_pixel_spectra.append(
                plebani_bands_spectra[coords[1] - 1, coords[0] - 1, :]
            )  # coords[1]-1 is the row, coords[0]-1 is the column

        average_bland_spectra = np.mean(bland_pixel_spectra, axis=0)
        average_bland_series = pd.Series(
            [
                average_bland_spectra,
                coords_best_bland_pixels,
                [sample_x_coord, sample_y_coord],
                image_id,
            ],
            index=[
                "Average_Spectra",
                "Bland_Pixel_Coordinates",
                "Mineral_Pixel_Coordinates",
                "Image_Name",
            ],
        )
        identified_useful_bland_pixels.append(average_bland_series)

    if len(identified_useful_bland_pixels) == 0:
        continue

    bland_pixel_dataframe = pd.DataFrame(identified_useful_bland_pixels)
    bland_pixel_dataframe["Image_Name"] = \
        bland_pixel_dataframe["Image_Name"].astype("string")
    bland_pixel_dataframe.to_json(f"{OUTPUT_DIR}/{image_id}_bland_pixels.json")

print(
    f"Number of mineral pixels that no suitable\n"
    f"bland pixels could be found for: {bad_pixel_count}"
)

bland_pixel_files = glob.glob(os.path.join(OUTPUT_DIR, "*_bland_pixels.json"))
bland_pixel_dfs = [pd.read_json(f) for f in bland_pixel_files]
all_bland_pixels_df = pd.concat(bland_pixel_dfs, ignore_index=True)
all_bland_pixels_df.reset_index().to_json(
    os.path.join(OUTPUT_DIR, "bland_data.json")
)

for f in bland_pixel_files:
    os.remove(f)

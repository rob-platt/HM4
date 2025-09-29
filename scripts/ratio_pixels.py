# Script to ratio mineral pixels by bland pixels as a preprocessing step for
# the classifier. Also ensures any pixels which can't be ratioed are dropped
# from the dataset.
import argparse

import n2n4m.preprocessing as n2n4m_preprocessing
import n2n4m.utils as n2n4m_utils
import pandas as pd
from n2n4m.wavelengths import PLEBANI_WAVELENGTHS, ALL_WAVELENGTHS

from hm4.preprocessing import align_datasets
from hm4.preprocessing import drop_pix_missing_blands

parser = argparse.ArgumentParser(
    description="""Ratio mineral pixels by bland pixels as a
    preprocessing step for training HM4."""
)

parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    required=True,
    help="Output filepath for the ratioed pixel JSON.",
)
parser.add_argument(
    "-m",
    "--mineral_data",
    type=str,
    required=True,
    help="Path to JSON file of the mineral pixel data.",
)
parser.add_argument(
    "-b",
    "--bland_data",
    type=str,
    required=True,
    help="Path to JSON file of the bland pixel data.",
)
arguments = parser.parse_args()
args = vars(arguments)

output_file = args["output_file"]
mineral_data_path = args["mineral_data"]
bland_data_path = args["bland_data"]

mineral_df = n2n4m_preprocessing.load_dataset(mineral_data_path)
bland_df = n2n4m_preprocessing.load_dataset(bland_data_path)

bland_df["Coordinates"] = bland_df["Mineral_Pixel_Coordinates"]
bland_df["Spectrum"] = bland_df["Average_Spectra"]
bland_df = bland_df.drop(
    columns=["Average_Spectra", "Mineral_Pixel_Coordinates"]
)
bland_df = n2n4m_utils.convert_coordinates_to_xy(bland_df)
mineral_df = n2n4m_utils.convert_coordinates_to_xy(mineral_df)

bland_df = n2n4m_preprocessing.expand_dataset(bland_df, PLEBANI_WAVELENGTHS)
mineral_df = n2n4m_preprocessing.expand_dataset(mineral_df)

clip_mineral_df, clip_bland_df = drop_pix_missing_blands(mineral_df, bland_df)
aligned_mineral_df, aligned_bland_df = align_datasets(
    clip_mineral_df, clip_bland_df
)

# if "index" is a column in either dataframe, drop it
if "index" in aligned_mineral_df.columns:
    aligned_mineral_df = aligned_mineral_df.drop(columns=["index"])
if "index" in aligned_bland_df.columns:
    aligned_bland_df = aligned_bland_df.drop(columns=["index"])

bland_spectra = aligned_bland_df.drop(
    columns=["Bland_Pixel_Coordinates", "x", "y", "Image_Name"]
).to_numpy()
mineral_spectra = aligned_mineral_df.drop(
    columns=["Pixel_Class", "x", "y", "Image_Name"]
).to_numpy()

# check to see if the mineral spectra have been cropped to
# the reduced number of bands
if mineral_spectra.shape[-1] == 438:
    wavelength_mask = [
        True if band in PLEBANI_WAVELENGTHS
        else False for band in ALL_WAVELENGTHS
    ]
    mineral_spectra = mineral_spectra[:, wavelength_mask]

ratioed_spectra = mineral_spectra / bland_spectra

ratioed_df = pd.DataFrame(ratioed_spectra, columns=PLEBANI_WAVELENGTHS)
ratioed_df["Pixel_Class"] = aligned_mineral_df["Pixel_Class"]
ratioed_df["x"] = aligned_mineral_df["x"]
ratioed_df["y"] = aligned_mineral_df["y"]
ratioed_df["Image_Name"] = aligned_mineral_df["Image_Name"]
ratioed_df = n2n4m_utils.convert_xy_to_coordinates(ratioed_df)

print(f"Length of the ratioed pixel dataset is {len(ratioed_df)}")
ratioed_df.to_json(output_file)

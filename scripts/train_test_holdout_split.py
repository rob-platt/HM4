import argparse
import os
from n2n4m.preprocessing import load_dataset
from hm4.preprocessing import imagewise_train_test_split

HOLDOUT_IMAGE_IDS = ["08F68", "20BF9", "19538"]

parser = argparse.ArgumentParser(
    description="""Split the mineral/spectral dataset into
                   training, validation, testing, and holdout
                   sets."""
)

parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
    help="Output directory path for the split JSON files.",
)

parser.add_argument(
    "-m",
    "--mineral_data",
    type=str,
    required=True,
    help="Path to JSON file of the ratioed mineral pixel data.",
)

arguments = parser.parse_args()
args = vars(arguments)

data = load_dataset(args["mineral_data"])
holdout_data = data.loc[data["Image_Name"].isin(HOLDOUT_IMAGE_IDS), :]

holdout_data.to_json(
    os.path.join(args["output_dir"], "holdout_set.json")
)
# remove holdout images from original data
data = data[~data["Image_Name"].isin(HOLDOUT_IMAGE_IDS)]
train_set, test_set = imagewise_train_test_split(data, test_size=0.2)
train_set, val_set = imagewise_train_test_split(train_set, test_size=0.2)

train_set.to_json(os.path.join(args["output_dir"], "train_set.json"))
val_set.to_json(os.path.join(args["output_dir"], "val_set.json"))
test_set.to_json(os.path.join(args["output_dir"], "test_set.json"))

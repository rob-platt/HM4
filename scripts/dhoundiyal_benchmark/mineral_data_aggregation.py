# Small script to aggregate the preprocessed data which is saved as a JSON per mineral per train/test/holdout set,
# and aggregate it to one JSON per train/test/holdout set.
import os
import pandas as pd

# Insert directory paths to the training, testing, and holdout data per class here.
train_dir = # 
test_dir = #
holdout_dir = #
output_dir = # 

data_dir_list = [train_dir, test_dir, holdout_dir]
output_name_list = ["train", "test", "holdout"]

for data_dir, name in zip(data_dir_list, output_name_list):
    data_files = os.listdir(data_dir)
    mineral_data_list = []
    for data_file in data_files:
        mineral_data_list.append(pd.read_json(os.path.join(data_dir, data_file)))
    aggregated_preproc_data = pd.concat(mineral_data_list, axis=0)
    aggregated_preproc_data.reset_index().to_json(os.path.join(output_dir, f"{name}_data.json"))
import n2n4m.preprocessing as n2n4m_prep
import n2n4m.utils as n2n4m_utils
import pandas as pd
import torch
from torch.utils.data import Dataset


def split_pixels(
    group: pd.DataFrame, threshold: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the pixels (rows) in the group based on the index location
    given by the threshold DataFrame. Assumes they are already sorted.

    Parameters
    ----------
    group : pd.DataFrame
        The group of pixels to be split. Must contain "x" column.
    threshold : pd.DataFrame
        The DataFrame containing the quantile index for each group.

    Returns
    -------
    test_set : pd.DataFrame
        The rows above the quantile index.
    train_set : pd.DataFrame
        The rows below the quantile index.
    """

    quantile_index = threshold.loc[group.name]["x"]
    # can just use one coord to find the quantile index,
    # as they are already sorted by both x and y
    test_set = group[group["x"] >= quantile_index]
    train_set = group[group["x"] < quantile_index]
    return test_set, train_set


def imagewise_train_test_split(
    data: pd.DataFrame, test_size: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into a training and testing set.
    Each image is split into a non-overlapping training and testing set
    regionally, with the right hand side of the image as the test set.
    Split is also based on pixel classes, so test_size proportion of
    each class is taken from each image.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the pixel data.
        Must have columns 'Image_Name', 'Pixel_Class',
        and either 'Coordinates' or 'x' and 'y'.
    test_size : float, optional
        The proportion of the data to be used for testing. Default is 0.25.

    Returns
    -------
    train_set : pd.DataFrame
        The training set.
    test_set : pd.DataFrame
        The testing set.
    """
    coords_flag = False
    class_in_list_flag = False
    if not data.index.is_unique:
        raise ValueError("Dataframe index must be unique")
    if "Image_Name" not in data.columns or "Pixel_Class" not in data.columns:
        raise ValueError(
            """Dataframe must have columns 'Image_Name'
                          and 'Pixel_Class'"""
        )
    if "Coordinates" in data.columns:
        coords_flag = True
        data = n2n4m_utils.convert_coordinates_to_xy(data)
    if "x" not in data.columns or "y" not in data.columns:
        raise ValueError("Dataframe must have columns 'x' and 'y'")
    if isinstance(data["Pixel_Class"].iloc[0], list):
        class_in_list_flag = True
        data["Pixel_Class"] = data["Pixel_Class"].apply(lambda x: x[0])

    quantile_threshold = data.groupby(["Image_Name", "Pixel_Class"]).quantile(
        1 - test_size
    )
    test_train_splits = data.groupby(["Image_Name", "Pixel_Class"]).apply(
        split_pixels, include_groups=False, threshold=quantile_threshold
    )

    test_set = pd.concat([split[0] for split in test_train_splits])
    train_set = pd.concat([split[1] for split in test_train_splits])

    # add the groupby columns to the sets based on the indexes
    test_set["Image_Name"] = data.loc[test_set.index, "Image_Name"]
    train_set["Image_Name"] = data.loc[train_set.index, "Image_Name"]
    test_set["Pixel_Class"] = data.loc[test_set.index, "Pixel_Class"]
    train_set["Pixel_Class"] = data.loc[train_set.index, "Pixel_Class"]

    if coords_flag:
        test_set = n2n4m_utils.convert_xy_to_coordinates(test_set)
        train_set = n2n4m_utils.convert_xy_to_coordinates(train_set)
        data = n2n4m_utils.convert_xy_to_coordinates(data)
    if class_in_list_flag:
        test_set["Pixel_Class"] = test_set["Pixel_Class"].apply(lambda x: [x])
        train_set["Pixel_Class"] = train_set["Pixel_Class"].apply(
            lambda x: [x]
        )
    return train_set, test_set


def drop_pix_missing_blands(
    mineral_df: pd.DataFrame,
    bland_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop any pixels in the mineral dataset which don't have corresponding
    pixels in the bland dataset.

    Parameters
    ----------
    mineral_df : pd.DataFrame
        The mineral dataset.
        Must have columns "Image_Name",
        and either "x" and "y" or "Coordinates".
    bland_df : pd.DataFrame
        The bland dataset.
        Must have columns "Image_Name",
        and either "x" and "y" or "Coordinates".

    Returns
    -------
    restricted_mineral_df : pd.DataFrame
        The mineral dataset with only the pixels which have corresponding
        pixels in the bland dataset.
    restricted_bland_df : pd.DataFrame
        The bland dataset with only the pixels which have corresponding
        pixels in the mineral dataset.
    """
    mineral_coords_flag = False
    bland_coords_flag = False
    if not {"x", "y"}.issubset(mineral_df.columns):
        if "Coordinates" in mineral_df.columns:
            mineral_coords_flag = True
            mineral_df = n2n4m_utils.convert_coordinates_to_xy(mineral_df)
        else:
            raise ValueError(
                """mineral_df must have either 'x' and 'y'
                or 'Coordinates' columns."""
            )

    if not {"x", "y"}.issubset(bland_df.columns):
        if "Coordinates" in bland_df.columns:
            bland_coords_flag = True
            bland_df = n2n4m_utils.convert_coordinates_to_xy(bland_df)
        else:
            raise ValueError(
                """bland_df must have either 'x' and 'y'
                or 'Coordinates' columns."""
            )

    lookup1 = mineral_df.set_index(["x", "y", "Image_Name"]).index
    lookup2 = bland_df.set_index(["x", "y", "Image_Name"]).index

    restricted_mineral_df = mineral_df[lookup1.isin(lookup2)]
    restricted_bland_df = bland_df[lookup2.isin(lookup1)]

    if mineral_coords_flag:
        restricted_mineral_df = n2n4m_utils.convert_xy_to_coordinates(
            restricted_mineral_df
        )
        mineral_df = n2n4m_utils.convert_xy_to_coordinates(mineral_df)
    if bland_coords_flag:
        restricted_bland_df = n2n4m_utils.convert_xy_to_coordinates(
            restricted_bland_df
        )
        bland_df = n2n4m_utils.convert_xy_to_coordinates(bland_df)
    return restricted_mineral_df, restricted_bland_df


def align_datasets(
    mineral_df: pd.DataFrame, bland_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align the mineral and bland datasets by image and pixel coordinates.
    Requires both datasets to by restricted to the same pixels.

    Parameters
    ----------
    mineral_df : pd.DataFrame
        The mineral dataset.
        Must have columns "Image_Name", and either "x" and "y"
        or "Coordinates".
    bland_df : pd.DataFrame
        The bland dataset.
        Must have columns "Image_Name", and either "x" and "y"
        or "Coordinates".

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The aligned mineral and bland datasets.
    """
    mineral_coords_flag = False
    bland_coords_flag = False
    if len(mineral_df) != len(bland_df):
        raise ValueError(
            "mineral_df and bland_df must have the same number of pixels."
        )
    if "Image_Name" not in mineral_df.columns:
        raise ValueError("mineral_df must have an 'Image_Name' column.")
    if "Image_Name" not in bland_df.columns:
        raise ValueError("bland_df must have an 'Image_Name' column.")

    if not {"x", "y"}.issubset(mineral_df.columns):
        if "Coordinates" in mineral_df.columns:
            mineral_coords_flag = True
            mineral_df = n2n4m_utils.convert_coordinates_to_xy(mineral_df)
        else:
            raise ValueError(
                """mineral_df must have either 'x' and 'y' or
                'Coordinates' columns."""
            )
    if not {"x", "y"}.issubset(bland_df.columns):
        if "Coordinates" in bland_df.columns:
            bland_coords_flag = True
            bland_df = n2n4m_utils.convert_coordinates_to_xy(bland_df)
        else:
            raise ValueError(
                """bland_df must have either 'x' and 'y' or
                'Coordinates' columns."""
            )

    mineral_df = mineral_df.sort_values(by=["Image_Name", "x", "y"])
    bland_df = bland_df.sort_values(by=["Image_Name", "x", "y"])
    mineral_df = mineral_df.reset_index(drop=True)
    bland_df = bland_df.reset_index(drop=True)
    if mineral_coords_flag:
        mineral_df = n2n4m_utils.convert_xy_to_coordinates(mineral_df)
    if bland_coords_flag:
        bland_df = n2n4m_utils.convert_xy_to_coordinates(bland_df)
    return mineral_df, bland_df


class CRISMData(Dataset):
    def __init__(
        self,
        path: str,
        transform: bool = False,
        bands_to_use: tuple[int, int] = (0, 350),
    ):
        """Custom dataset class for ratioed CRISM hyperspectral data.

        Parameters
        ----------
        path : str
            Path to the dataset JSON file.
        transform : bool, optional
            Whether to apply transformations to the dataset, by default False.
            Current transformation is solely normalization.
        bands_to_use : tuple, optional
            The index range of bands to use from the dataset,
            by default (0, 350).
        """
        self.path = path
        self.transform = transform

        self.data, self.labels, self.ancilliary = self.read_data()
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.data = self.data[:, bands_to_use[0] : bands_to_use[1]]
        self.labels = self.labels.unsqueeze(1)

        if self.transform:
            self.data = self.scale_data(self.data)

    def read_data(self) -> tuple[pd.DataFrame, torch.Tensor, pd.DataFrame]:
        """Reads the dataset JSON file."""
        dataset = n2n4m_prep.load_dataset(self.path)
        labels = torch.from_numpy(dataset["Pixel_Class"].values)
        ancilliary = dataset[["Image_Name", "Coordinates"]]
        data = dataset.drop(
            columns=["Pixel_Class", "Image_Name", "Coordinates"]
        )
        return data, labels, ancilliary

    def scale_data(self, data: torch.Tensor) -> torch.Tensor:
        """Scale each spectra to the range [0, 1]."""
        max = torch.max(data, dim=1, keepdim=True).values
        min = torch.min(data, dim=1, keepdim=True).values
        data = (self.data - min) / (max - min)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_ancilliary(self, idx):
        return self.ancilliary.iloc[idx]

import numpy as np
import pandas as pd
import pytest

import hm4.preprocessing as classifier_preproc
from n2n4m.utils import convert_xy_to_coordinates


@pytest.fixture
def single_image_class_data():
    "Simple dataset of 6 pixels, single image, single pixel class"
    data = pd.DataFrame(
        {
            "x": np.arange(1, 7),
            "y": np.arange(1, 7),
            "Pixel_Class": np.full(6, 1),
            "Image_Name": ["test_image" for x in range(6)],
        }
    )
    return data


@pytest.fixture
def multi_image_class_data(single_image_class_data):
    "12 pixels, 6 from each of 2 images, single pixel class"
    data_2 = single_image_class_data.copy(deep=True)
    data_2["Image_Name"] = "test_image_2"
    data = pd.concat([single_image_class_data, data_2], ignore_index=True)
    return data


@pytest.fixture
def multi_image_multi_class_data(multi_image_class_data):
    "12 pixels, 6 from each of 2 images, 2 pixel classes per image"
    data = multi_image_class_data.copy(deep=True)
    data.loc[0:3, "Pixel_Class"] = 2
    data.loc[6:9, "Pixel_Class"] = 2
    return data


@pytest.fixture
def single_image_class_bland_data(single_image_class_data):
    "Bland pixels for a single image. 6 pixels"
    data = single_image_class_data.copy(deep=True)
    data["Pixel_Class"] = 21
    return data


# Get a user warning for defining the name in this way
# but is the only way of mocking the name attribute
# of the groupby object
@pytest.mark.filterwarnings("ignore: Pandas")
def test_split_pixels_even_num(single_image_class_data):
    "Test the split_pixels function with an even number of samples"
    # With threshold = 0.5, should return 2 test and 2 train pixels
    thresh = single_image_class_data.groupby(
        ["Image_Name", "Pixel_Class"]
    ).quantile(0.5)

    data_groups = single_image_class_data.groupby(
        ["Image_Name", "Pixel_Class"]
    )
    single_group = data_groups.get_group(("test_image", 1))
    single_group.name = ("test_image", 1)
    test_set, train_set = classifier_preproc.split_pixels(single_group, thresh)
    assert len(test_set) == len(train_set)
    assert test_set["Image_Name"].nunique() == 1
    assert train_set["Image_Name"].nunique() == 1
    assert test_set["Pixel_Class"].nunique() == 1
    assert train_set["Pixel_Class"].nunique() == 1
    assert test_set["x"].nunique() == len(single_image_class_data) / 2
    assert train_set["x"].nunique() == len(single_image_class_data) / 2
    # check for duplicates between sets
    assert len(test_set.merge(train_set)) == 0


# Get a user warning for defining the name in this way
# but is the only way of mocking the name attribute
# of the groupby object
@pytest.mark.filterwarnings("ignore: Pandas")
def test_split_pixels_odd_num(single_image_class_data):
    """Test the split_pixels function with an odd number of samples.
    Should return 3 test and 2 train pixels."""
    single_image_class_data = single_image_class_data.drop(index=5)

    thresh = single_image_class_data.groupby(
        ["Image_Name", "Pixel_Class"]
    ).quantile(0.5)
    data_groups = single_image_class_data.groupby(
        ["Image_Name", "Pixel_Class"]
    )

    single_group = data_groups.get_group(("test_image", 1))
    single_group.name = ("test_image", 1)
    test_set, train_set = classifier_preproc.split_pixels(single_group, thresh)

    # with odd number of samples, and even split
    # want the extra sample in test set to prevent issues
    # with single sample.
    assert len(test_set) == len(train_set) + 1
    assert len(test_set.merge(train_set)) == 0


def test_imagewise_train_test_split_invalid_index(single_image_class_data):
    """Test that an error is raised if the index is not a unique range of
    integers."""
    single_image_class_data.index = np.array([1, 2, 1, 4, 5, 6])
    with pytest.raises(ValueError):
        classifier_preproc.imagewise_train_test_split(
            single_image_class_data, 0.5
        )


def test_imagewise_train_test_split_missing_coords(single_image_class_data):
    "Test that an error is raised if the x or y columns are missing."
    single_image_class_data_x_miss = single_image_class_data.drop(
        columns=["x"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.imagewise_train_test_split(
            single_image_class_data_x_miss, 0.5
        )

    single_image_class_data_y_miss = single_image_class_data.drop(
        columns=["y"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.imagewise_train_test_split(
            single_image_class_data_y_miss
        )


def test_imagewise_train_test_split_coord_as_list(single_image_class_data):
    """Test that the function works if provided with a coordinates column as a
    list."""
    single_image_class_data_coords = convert_xy_to_coordinates(
        single_image_class_data
    )
    test_coords, train_coords = classifier_preproc.imagewise_train_test_split(
        single_image_class_data_coords, 0.5
    )
    single_image_class_data_xy = single_image_class_data.drop(
        columns=["Coordinates"]
    )
    test_xy, train_xy = classifier_preproc.imagewise_train_test_split(
        single_image_class_data_xy, 0.5
    )

    test_xy = convert_xy_to_coordinates(test_xy)
    train_xy = convert_xy_to_coordinates(train_xy)

    assert test_coords.equals(test_xy)
    assert train_coords.equals(train_xy)


def test_imagewise_train_test_split_missing_class(single_image_class_data):
    """Test that an error is raised if the class column is missing."""
    single_image_class_data_class_miss = single_image_class_data.drop(
        columns=["Pixel_Class"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.imagewise_train_test_split(
            single_image_class_data_class_miss, 0.5
        )


def test_imagewise_train_test_split_missing_imname(single_image_class_data):
    """Test that an error is raised if the Image_Name column is missing."""
    single_image_class_data_imname_miss = single_image_class_data.drop(
        columns=["Image_Name"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.imagewise_train_test_split(
            single_image_class_data_imname_miss, 0.5
        )


def test_imagewise_train_test_split_multi_image(multi_image_class_data):
    """Test that the imagewise_train_test_split function works with multiple
    images and a single pixel class per image."""
    train_set, test_set = classifier_preproc.imagewise_train_test_split(
        multi_image_class_data, 0.5
    )
    # Check split is 0.5
    assert len(train_set) == len(multi_image_class_data) / 2
    assert len(test_set) == len(multi_image_class_data) / 2
    # Check all images are in both sets
    assert (
        train_set["Image_Name"].nunique()
        == multi_image_class_data["Image_Name"].nunique()
    )
    assert (
        test_set["Image_Name"].nunique()
        == multi_image_class_data["Image_Name"].nunique()
    )
    # Check all classes are in both sets
    assert (
        test_set["Pixel_Class"].nunique()
        == multi_image_class_data["Pixel_Class"].nunique()
    )
    assert (
        train_set["Pixel_Class"].nunique()
        == multi_image_class_data["Pixel_Class"].nunique()
    )
    # Check that the number of unique x values is halved
    assert test_set["x"].nunique() == multi_image_class_data["x"].nunique() / 2
    assert (
        train_set["x"].nunique() == multi_image_class_data["x"].nunique() / 2
    )
    # check for duplicates between sets
    assert len(test_set.merge(train_set)) == 0


def test_imagewise_train_test_split_multi_image_multi_class(
    multi_image_multi_class_data,
):
    """Test that the imagewise_train_test_split function works with multiple
    images and multiple pixel classes per image."""
    train_set, test_set = classifier_preproc.imagewise_train_test_split(
        multi_image_multi_class_data, 0.5
    )
    # Check split is 0.5
    assert len(train_set) == len(multi_image_multi_class_data) / 2
    assert len(test_set) == len(multi_image_multi_class_data) / 2
    # Check all images are in both sets
    assert (
        train_set["Image_Name"].nunique()
        == multi_image_multi_class_data["Image_Name"].nunique()
    )
    assert (
        test_set["Image_Name"].nunique()
        == multi_image_multi_class_data["Image_Name"].nunique()
    )
    # Check all classes are in both sets
    assert (
        test_set["Pixel_Class"].nunique()
        == multi_image_multi_class_data["Pixel_Class"].nunique()
    )
    assert (
        train_set["Pixel_Class"].nunique()
        == multi_image_multi_class_data["Pixel_Class"].nunique()
    )
    # Check for duplicates between sets
    assert len(test_set.merge(train_set)) == 0


def test_drop_pix_missing_blands(
    single_image_class_data,
    single_image_class_bland_data,
):
    """Test that the drop_pix_missing_blands function removes mineral pixels
    that don't have corresponding bland pixel data."""
    missing_pixel = single_image_class_bland_data.iloc[0].to_frame().T
    bland_data = single_image_class_bland_data.drop(index=0)
    restricted_mineral_df, restricted_bland_df = (
        classifier_preproc.drop_pix_missing_blands(
            single_image_class_data, bland_data
        )
    )
    # check that the mineral data has been reduced by 1
    assert len(restricted_mineral_df) == len(single_image_class_data) - 1
    # check that the bland data has also been reduced by 1
    assert len(restricted_bland_df) == len(single_image_class_bland_data) - 1
    # check that the missing pixel is not in the restricted data
    assert not restricted_mineral_df.isin(missing_pixel).all().all()
    # check x, y coords of missing pixel not in restricted mineral data
    assert (
        missing_pixel[["x", "y"]].values
        not in restricted_mineral_df[["x", "y"]].values
    )


def test_drop_pix_missing_blands_no_blands(
    single_image_class_data, single_image_class_bland_data
):
    """Test that the drop_pix_missing_blands function if there are no relevant
    bland pixels."""
    # Change the bland x values to be from columns not in the mineral data
    single_image_class_bland_data["x"] = 12

    restricted_mineral_df, restricted_bland_df = (
        classifier_preproc.drop_pix_missing_blands(
            single_image_class_data, single_image_class_bland_data
        )
    )
    # check that the restricted mineral data is empty
    assert len(restricted_mineral_df) == 0
    # check that the restricted bland data is empty
    assert len(restricted_bland_df) == 0


def test_drop_pix_missing_blands_missing_coords(single_image_class_bland_data):
    """Test that an error is raised if the x or y columns are missing."""
    bland_data_x_miss = single_image_class_bland_data.drop(columns=["x"])
    with pytest.raises(ValueError):
        classifier_preproc.drop_pix_missing_blands(
            single_image_class_bland_data, bland_data_x_miss
        )

    bland_data_y_miss = single_image_class_bland_data.drop(columns=["y"])
    with pytest.raises(ValueError):
        classifier_preproc.drop_pix_missing_blands(
            single_image_class_bland_data, bland_data_y_miss
        )


def test_drop_pix_missing_blands_coord_input(
    single_image_class_data, single_image_class_bland_data
):
    """Test that the function works if provided with a coordinates column as a
    list."""
    missing_pixel = single_image_class_bland_data.iloc[0].to_frame().T
    missing_pixel = convert_xy_to_coordinates(missing_pixel)
    bland_data = single_image_class_bland_data.drop(index=0)
    single_image_class_data_coords = convert_xy_to_coordinates(
        single_image_class_data
    )
    single_image_class_bland_data_coords = convert_xy_to_coordinates(
        bland_data
    )
    restricted_mineral_df, restricted_bland_df = (
        classifier_preproc.drop_pix_missing_blands(
            single_image_class_data_coords,
            single_image_class_bland_data_coords,
        )
    )
    assert len(restricted_mineral_df) == len(single_image_class_data) - 1
    # check that the bland data has also been reduced by 1
    assert len(restricted_bland_df) == len(single_image_class_bland_data) - 1
    # check that the missing pixel is not in the restricted data
    assert not restricted_mineral_df.isin(missing_pixel).all().all()
    # check x, y coords of missing pixel not in restricted mineral data
    assert (
        missing_pixel["Coordinates"].values
        not in restricted_mineral_df["Coordinates"].values
    )


def test_align_datasets(
    single_image_class_data, single_image_class_bland_data
):
    """Test that the align_datasets function correctly aligns the bland and
    mineral datasets."""
    # Randomize the order of the mineral data
    single_image_class_data = single_image_class_data.sample(
        frac=1
    ).reset_index(drop=True)

    aligned_mineral_df, aligned_bland_df = classifier_preproc.align_datasets(
        single_image_class_data, single_image_class_bland_data
    )
    # Check that the number of samples is the same
    assert len(aligned_mineral_df) == len(single_image_class_data)
    assert len(aligned_bland_df) == len(single_image_class_bland_data)
    # Check that the x and y values are the same and in the same order
    assert aligned_mineral_df["x"].equals(aligned_bland_df["x"])
    assert aligned_mineral_df["y"].equals(aligned_bland_df["y"])
    # Check that the index of the aligned datasets is the same
    assert aligned_mineral_df.index.equals(aligned_bland_df.index)
    # Check the index is in order
    assert aligned_mineral_df.index.is_monotonic_increasing
    assert aligned_bland_df.index.is_monotonic_increasing


def test_align_datasets_unequal_rows(
    single_image_class_data, single_image_class_bland_data
):
    """Test that an error is raised if the datasets have different numbers of
    rows."""
    single_image_class_data = single_image_class_data.drop(index=0)
    with pytest.raises(ValueError):
        classifier_preproc.align_datasets(
            single_image_class_data, single_image_class_bland_data
        )


def test_align_datasets_missing_coords(
    single_image_class_data, single_image_class_bland_data
):
    """Test that an error is raised if the x or y columns are missing."""
    single_image_class_data_x_miss = single_image_class_data.drop(
        columns=["x"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.align_datasets(
            single_image_class_data_x_miss, single_image_class_bland_data
        )

    single_image_class_data_y_miss = single_image_class_data.drop(
        columns=["y"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.align_datasets(
            single_image_class_data_y_miss, single_image_class_bland_data
        )

    single_image_class_bland_data_x_miss = single_image_class_bland_data.drop(
        columns=["x"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.align_datasets(
            single_image_class_data, single_image_class_bland_data_x_miss
        )

    single_image_class_bland_data_y_miss = single_image_class_bland_data.drop(
        columns=["y"]
    )
    with pytest.raises(ValueError):
        classifier_preproc.align_datasets(
            single_image_class_data, single_image_class_bland_data_y_miss
        )

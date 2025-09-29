import numpy as np
import torch

N_CLASSES = 38

CLASS_NAMES = {
    0: "CO2 Ice",  # co2_ice
    1: "H2O Ice",  # h20_ice
    2: "Gypsum",  # gypsum
    3: "Ferric Hydroxysulfate",  # hydrox_fe_sulf
    4: "Hematite",  # hematite
    5: "Nontronite",  # fe_smectite
    6: "Saponite",  # mg_smectite
    7: "Prehnite",  # Prehnite Zeolite # prehnite
    8: "Jarosite",  # jarosite
    9: "Serpentine",  # serpentine
    10: "Alunite",  # alunite
    11: "Akaganeite",  # Fe Oxyhydroxysulfate # hydrox_fe_sulf
    12: "Ca/Fe CO3",  # Calcite, Ca/Fe carbonate  # fe_ca_carbonate
    13: "Beidellite",  # Al-smectite # al_smectite
    14: "Kaolinite",  # kaolinite
    15: "Bassanite",  # bassanite
    16: "Epidote",  # epidote
    17: "Montmorillonite",  # Al-smectite # al_smectite
    18: "Rosenite",  # Polyhydrated sulfate # poly_hyd_sulf
    19: "Mg Cl salt",  # Mg(ClO3)2.6H2O # Polyhydrated sulfate # poly_hyd_sulf
    20: "Halloysite",  # Kaolinite # kaolinite
    21: "Bland",  # Neutral/no spectral features
    22: "Illite/Muscovite",  # illite_muscovite
    23: "Margarite",  # Illite/Muscovite # illite_muscovite
    24: "Analcime",  # Zeolite # analcime
    25: "Monohydrated sulfate",  # Szomolnokite # mono_hyd_sulf
    26: "Opal 1",  # Opal # Hydrated silica # hydrated_silica
    27: "Opal 2",  # Opal-A # Hydrated silica # hydrated_silica
    28: "Iron Oxide Silicate Sulfate",  # Polyhydrated sulfate # poly_hyd_sulf
    29: "MgCO3",  # Magnesite # mg_carbonate
    30: "Chlorite",  # chlorite
    31: "Clinochlore",  # chlorite
    32: "Low Ca Pyroxene",  # lcp
    33: "Olivine Forsterite",  # mg_olivine
    34: "High Ca Pyroxene",  # hcp
    35: "Olivine Fayalite",  # fe_olivine
    36: "Chloride",  # chloride
    37: "Artefact",  # Camera artefact described in Leask et al. 2018
}

PLATT_TO_PLEBANI = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    # Mapping the bland class to the neutral class in Plebani et al. 2022
    21: 39,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    27: 28,
    28: 29,
    29: 30,
    30: 31,
    31: 32,
    32: 33,
    33: 34,
    34: 35,
    35: 36,
    36: 37,
    37: 38,
}

BROAD_MINERAL_FAMILIES = {
    "CO2 Ice": "Ice",
    "H2O Ice": "Ice",
    "Gypsum": "Polyhydrated Sulfate",
    "Ferric Hydroxysulfate": "Hydroxylated Fe Sulfate",
    "Hematite": "Hematite",
    "Nontronite": "Smectite",
    "Saponite": "Smectite",
    "Prehnite": "Zeolite",
    "Jarosite": "Jarosite",
    "Serpentine": "Serpentine",
    "Alunite": "Alunite",
    "Akaganeite": "Hydroxylated Fe Sulfate",
    "Ca/Fe CO3": "Carbonate",
    "Beidellite": "Smectite",
    "Kaolinite": "Kaolinite",
    "Bassanite": "Zeolite",
    "Epidote": "Epidote",
    "Montmorillonite": "Smectite",
    "Rosenite": "Polyhydrated Sulfate",
    "Mg Cl salt": "Halide",
    "Halloysite": "Kaolinite",
    "Bland": "Bland",
    "Illite/Muscovite": "Illite/Muscovite",
    "Margarite": "Illite/Muscovite",
    "Analcime": "Zeolite",
    "Monohydrated sulfate": "Monohydrated Sulfate",
    "Opal 1": "Hydrated Silica",
    "Opal 2": "Hydrated Silica",
    "Iron Oxide Silicate Sulfate": "Polyhydrated Sulfate",
    "MgCO3": "Carbonate",
    "Chlorite": "Chlorite",
    "Clinochlore": "Chlorite",
    "Low Ca Pyroxene": "Pyroxene",
    "Olivine Forsterite": "Olivine",
    "High Ca Pyroxene": "Pyroxene",
    "Olivine Fayalite": "Olivine",
    "Chloride": "Halide",
    "Artefact": "Artefact",
}

MINERAL_CATEGORIES = {
    "CO2 Ice": "Ice",
    "H2O Ice": "Ice",
    "Gypsum": "Sulfate",
    "Ferric Hydroxysulfate": "Hydroxysulfate",
    "Hematite": "Oxide",
    "Nontronite": "Clay",
    "Saponite": "Clay",
    "Prehnite": "Zeolite",
    "Jarosite": "Hydroxysulfate",
    "Serpentine": "Hydrated Silicate",
    "Alunite": "Sulfate",
    "Akaganeite": "Hydroxylated Fe Sulfate",
    "Ca/Fe CO3": "Carbonate",
    "Beidellite": "Clay",
    "Kaolinite": "Clay",
    "Bassanite": "Zeolite",
    "Epidote": "Hydrated Silicate",
    "Montmorillonite": "Clay",
    "Rosenite": "Sulfate",
    "Mg Cl salt": "Halide",
    "Halloysite": "Clay",
    "Bland": "Bland",
    "Illite/Muscovite": "Clay",
    "Margarite": "Clay",
    "Analcime": "Zeolite",
    "Monohydrated sulfate": "Sulfate",
    "Opal 1": "Hydrated Silicate",
    "Opal 2": "Hydrated Silicate",
    "Iron Oxide Silicate Sulfate": "Sulfate",
    "MgCO3": "Carbonate",
    "Chlorite": "Hydrated Silicate",
    "Clinochlore": "Hydrated Silicate",
    "Low Ca Pyroxene": "Mafic",
    "Olivine Forsterite": "Mafic",
    "High Ca Pyroxene": "Mafic",
    "Olivine Fayalite": "Mafic",
    "Chloride": "Halide",
    "Artefact": "Artefact",
}


def is_array_zero_indexed(array: np.ndarray | torch.Tensor) -> bool:
    """
    Checks if array of class labels is 0-indexed.
    Only guaranteed to work if the array contains all classes.
    Strict on 0 indexed, loose on 1 indexed.

    Parameters
    ----------
    array : np.ndarray|torch.Tensor
        Array of class labels to be checked.
        Shape (n_samples).

    Returns
    -------
    bool
        True if array is 0-indexed, False otherwise.
    """
    if array.min() == 0 and array.max() == N_CLASSES - 1:
        return True
    return False


def make_array_zero_indexed(
    array: np.ndarray | torch.Tensor, forwards=True, force=False
) -> np.ndarray | torch.Tensor:
    """
    Converts array of class labels from 1-indexed to 0-indexed or vice versa.
    Will raise an error if the min or max
    values of the array suggest array is already in the desired format.
    Error handling not foolproof if the array does not contain all classes.

    Parameters
    ----------
    array : np.ndarray|torch.Tensor
        Array of class labels to be converted.
        Shape (n_samples).
    forwards : bool, default True
        If True, converts 1-indexed to 0-indexed.
        If False, converts 0-indexed to 1-indexed.
    force : bool, default False
        If True, forces conversion even if array is suspected
        to already be 0-indexed or 1-indexed.

    Returns
    -------
    output_arr : np.ndarray|torch.Tensor
        Array of class labels converted to 0-indexed or 1-indexed.
    """
    if isinstance(array, torch.Tensor):
        output_arr = array.clone()
    else:
        output_arr = array.copy()
    if forwards:
        if not force:
            if is_array_zero_indexed(array):
                raise ValueError("Suspect array is already 0-indexed.")
        output_arr = output_arr - 1
        return array - 1
    else:
        if not force:
            if not is_array_zero_indexed(array):
                raise ValueError("Suspect array is already 1-indexed.")
        return array + 1


def change_pixel_label(
    array: np.ndarray | torch.Tensor, in_val=38, out_val=21
) -> np.ndarray | torch.Tensor:
    """
    Change the value of label in the array. Used for changing the bland pixel
    label from 38 (the 0-indexed version of the Plebani dataset label) to 21
    (0-indexed version of the labels used in this study).

    Change is done so that there isn't an unused label value in the middle
    of the range (21), so can use the index position of output probabilities
    from the Classifier to make output label predictions.

    Parameters
    ----------
    array : np.ndarray|torch.Tensor
        Array of class labels to be converted.
        Shape (n_samples).
    in_val : int, default 38
        Value of label to be changed.
    out_val : int, default 21
        Value to change the label to.
    """
    output_arr = array
    output_arr[array == in_val] = out_val
    return output_arr

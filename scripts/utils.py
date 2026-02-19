import os
import random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from numpy.typing import NDArray


# -------------------- File System Utilities --------------------
def ensure_directory(path: str) -> None:
    """
    Create directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def get_patient_id(patient_index: int, prefix: str = "PAT") -> str:
    """
    Generate a standardized patient ID.

    Args:
        patient_index: Integer index of the patient (0-based).
        prefix: String prefix for the ID.

    Returns:
        Formatted patient ID string, e.g., "PAT_0001".
    """
    return f"{prefix}_{patient_index:04d}"


def get_image_filename(patient_id: str, modality: str, extension: str = ".npy") -> str:
    """
    Generate a filename for a patient's image of a given modality.

    Args:
        patient_id: Patient identifier (e.g., "PAT_0001").
        modality: Imaging modality (e.g., "MRI", "CT", "Ultrasound").
        extension: File extension including the dot.

    Returns:
        Filename string.
    """
    return f"{patient_id}_{modality}{extension}"


def get_image_path(patient_id: str, modality: str, base_dir: str, images_subdir: str = "images",
                   extension: str = ".npy") -> str:
    """
    Construct the full path to an image file.

    Args:
        patient_id: Patient identifier.
        modality: Imaging modality.
        base_dir: Root dataset directory.
        images_subdir: Subdirectory where images are stored.
        extension: File extension.

    Returns:
        Full path to the image file.
    """
    return os.path.join(base_dir, images_subdir, get_image_filename(patient_id, modality, extension))


# -------------------- Random Seed Utilities --------------------
def set_global_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python's random, NumPy, and (optionally) Python's hash.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Optionally set PYTHONHASHSEED via os.environ if needed, but not strictly required.
    os.environ["PYTHONHASHSEED"] = str(seed)


# -------------------- Image I/O Utilities --------------------
def save_image(array: NDArray[Any], filepath: str) -> None:
    """
    Save a NumPy array as a .npy file.

    Args:
        array: Image data as a NumPy array.
        filepath: Destination file path.
    """
    ensure_directory(os.path.dirname(filepath))
    np.save(filepath, array)


def load_image(filepath: str) -> NDArray[Any]:
    """
    Load a NumPy array from a .npy file.

    Args:
        filepath: Path to the .npy file.

    Returns:
        Loaded NumPy array.
    """
    return np.load(filepath)


# -------------------- Metadata I/O Utilities --------------------
def save_labels(dataframe: pd.DataFrame, filepath: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        dataframe: DataFrame containing labels/metadata.
        filepath: Destination CSV file path.
    """
    ensure_directory(os.path.dirname(filepath))
    dataframe.to_csv(filepath, index=False)


def load_labels(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with loaded data.
    """
    return pd.read_csv(filepath)


# -------------------- Data Splitting Utilities --------------------
def assign_split(patient_index: int, num_patients: int, split_ratios: Dict[str, float]) -> str:
    """
    Deterministically assign a patient to a split based on index and ratios.

    Args:
        patient_index: Integer index of the patient (0-based).
        num_patients: Total number of patients.
        split_ratios: Dictionary mapping split names to fractions (sum=1).

    Returns:
        Name of the split (e.g., "train", "val", "test").

    Raises:
        ValueError: If split ratios do not sum to 1.
    """
    if not np.isclose(sum(split_ratios.values()), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios.values())}")

    # Compute cumulative thresholds
    cumulative = 0.0
    boundaries = []
    for split, ratio in split_ratios.items():
        cumulative += ratio
        boundaries.append((split, cumulative))

    # Normalized position of this patient (ensure last patient falls into last split)
    position = (patient_index + 1) / num_patients  # +1 to avoid 0 at start

    for split, threshold in boundaries:
        if position <= threshold:
            return split

    # Fallback (should not happen due to floating point)
    return list(split_ratios.keys())[-1]
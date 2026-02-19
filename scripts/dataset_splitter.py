import os
import pandas as pd
from typing import Dict, List, Optional

from config import DATASET_ROOT, METADATA_DIR, LABELS_FILE, SPLIT_RATIOS
from utils import ensure_directory


def load_labels(labels_path: str) -> pd.DataFrame:
    """
    Load the labels CSV file.

    Args:
        labels_path: Path to the labels CSV file.

    Returns:
        DataFrame containing labels.
    """
    return pd.read_csv(labels_path)


def get_split_indices(labels_df: pd.DataFrame, split_name: str) -> List[str]:
    """
    Get all patient IDs belonging to a specific split.

    Args:
        labels_df: DataFrame with a 'split' column.
        split_name: One of 'train', 'val', 'test'.

    Returns:
        List of patient_id strings for that split.
    """
    if split_name not in SPLIT_RATIOS.keys():
        raise ValueError(f"Split name must be one of {list(SPLIT_RATIOS.keys())}, got '{split_name}'")
    return labels_df[labels_df["split"] == split_name]["patient_id"].tolist()


def get_split_dataframe(labels_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Get a DataFrame subset for a specific split.

    Args:
        labels_df: Full labels DataFrame.
        split_name: Split name.

    Returns:
        DataFrame containing only rows for that split.
    """
    return labels_df[labels_df["split"] == split_name].copy()


def write_split_csvs(labels_df: pd.DataFrame, output_dir: str, prefix: str = "") -> None:
    """
    Write separate CSV files for each split.

    Args:
        labels_df: DataFrame with a 'split' column.
        output_dir: Directory where CSV files will be saved.
        prefix: Optional prefix for filenames (e.g., "labels_").
    """
    ensure_directory(output_dir)
    for split in SPLIT_RATIOS.keys():
        split_df = get_split_dataframe(labels_df, split)
        output_path = os.path.join(output_dir, f"{prefix}{split}.csv")
        split_df.to_csv(output_path, index=False)
        print(f"Saved {len(split_df)} records to {output_path}")


def verify_split_distribution(labels_df: pd.DataFrame) -> Dict[str, float]:
    """
    Verify the actual split distribution against expected ratios.

    Args:
        labels_df: DataFrame with a 'split' column.

    Returns:
        Dictionary with split names as keys and actual fractions as values.
    """
    total = len(labels_df)
    actual = labels_df["split"].value_counts(normalize=True).to_dict()
    print("Expected split ratios:", SPLIT_RATIOS)
    print("Actual split ratios:  ", actual)
    return actual


# -------------------- Main entry point (if run as script) --------------------
if __name__ == "__main__":
    # Default paths
    labels_path = os.path.join(DATASET_ROOT, METADATA_DIR, LABELS_FILE)
    output_dir = os.path.join(DATASET_ROOT, METADATA_DIR)

    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        print("Please generate labels first by running labels_generator.py")
    else:
        df = load_labels(labels_path)
        print(f"Loaded labels for {len(df)} patients.")

        # Verify split distribution
        verify_split_distribution(df)

        # Write separate CSV files for each split
        write_split_csvs(df, output_dir, prefix="split_")
        print("Split CSV files created.")
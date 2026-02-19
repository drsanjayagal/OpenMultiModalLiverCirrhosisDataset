import os
from typing import Dict, List, Tuple

# -------------------- Random Seed --------------------
RANDOM_SEED: int = 42
"""Global random seed for reproducibility."""

# -------------------- Dataset Paths --------------------
DATASET_ROOT: str = "OpenMultiModalLiverCirrhosisDataset"
"""Root directory of the generated dataset."""
IMAGES_DIR: str = "images"
"""Subdirectory containing all image files."""
METADATA_DIR: str = "metadata"
"""Subdirectory containing metadata files (e.g., labels.csv)."""
LABELS_FILE: str = "labels.csv"
"""Filename for the main label CSV."""

# -------------------- Image Specifications --------------------
IMAGE_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "MRI": (256, 256),
    "CT": (512, 512),
    "Ultrasound": (224, 224)
}
"""Spatial resolution (height, width) for each modality."""

# -------------------- Dataset Size --------------------
NUM_PATIENTS: int = 1000
"""Total number of unique patients in the dataset."""

# -------------------- Train/Validation/Test Split --------------------
SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}
"""Fraction of patients assigned to each split. Must sum to 1.0."""

# -------------------- Label Definitions --------------------
FIBROSIS_STAGES: List[str] = ["F0", "F1", "F2", "F3", "F4"]
"""Ordered list of liver fibrosis stages according to METAVIR scoring."""

BINARY_LABEL_MAPPING: Dict[str, List[str]] = {
    "negative": ["F0", "F1", "F2"],   # No or mild fibrosis
    "positive": ["F3", "F4"]           # Advanced fibrosis / cirrhosis
}
"""Mapping from binary classes to the corresponding fibrosis stages."""

# -------------------- File Format Settings --------------------
IMAGE_FILE_EXTENSION: str = ".npy"
"""File extension for saved images (numpy binary format)."""
METADATA_FILE_FORMAT: str = "csv"
"""Format of the metadata file (CSV)."""
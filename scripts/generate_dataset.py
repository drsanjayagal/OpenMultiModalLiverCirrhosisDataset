import os
import sys
from typing import Dict, List

# Add project root to path to allow imports from scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.config import (
    RANDOM_SEED,
    DATASET_ROOT,
    IMAGES_DIR,
    METADATA_DIR,
    LABELS_FILE,
    IMAGE_RESOLUTIONS,
    NUM_PATIENTS,
    IMAGE_FILE_EXTENSION
)
from scripts.utils import (
    set_global_seed,
    ensure_directory,
    get_image_filename,
    get_image_path,
    save_image
)
from scripts.labels_generator import generate_all_labels
from scripts.image_generators import (
    generate_mri_image,
    generate_ct_image,
    generate_ultrasound_image
)
from scripts.dataset_splitter import write_split_csvs, verify_split_distribution


def generate_dataset(
    num_patients: int = NUM_PATIENTS,
    seed: int = RANDOM_SEED,
    base_dir: str = DATASET_ROOT
) -> None:
    """
    Main function to generate the entire synthetic dataset.

    Steps:
        1. Set random seed for reproducibility.
        2. Create directory structure.
        3. Generate labels and metadata for all patients.
        4. Save labels to CSV.
        5. Generate images for each patient and each modality.
        6. Write split-specific CSV files.
        7. Print summary.
    """
    # Step 1: Set seed
    set_global_seed(seed)
    print(f"Global random seed set to {seed}")

    # Step 2: Create directories
    images_path = os.path.join(base_dir, IMAGES_DIR)
    metadata_path = os.path.join(base_dir, METADATA_DIR)
    ensure_directory(images_path)
    ensure_directory(metadata_path)
    print(f"Directories created under {base_dir}")

    # Step 3: Generate labels
    print(f"Generating labels for {num_patients} patients...")
    labels_df = generate_all_labels(num_patients=num_patients, seed=seed)

    # Step 4: Save labels CSV
    labels_csv_path = os.path.join(metadata_path, LABELS_FILE)
    labels_df.to_csv(labels_csv_path, index=False)
    print(f"Labels saved to {labels_csv_path}")

    # Step 5: Generate images
    print("Generating images for all patients and modalities...")
    modalities = list(IMAGE_RESOLUTIONS.keys())
    total_images = num_patients * len(modalities)
    count = 0

    for idx, row in labels_df.iterrows():
        patient_id = row["patient_id"]
        fibrosis_stage = row["fibrosis_stage"]

        for modality in modalities:
            resolution = IMAGE_RESOLUTIONS[modality]

            # Choose the correct generator function
            if modality == "MRI":
                img = generate_mri_image(patient_id, fibrosis_stage, resolution)
            elif modality == "CT":
                img = generate_ct_image(patient_id, fibrosis_stage, resolution)
            elif modality == "Ultrasound":
                img = generate_ultrasound_image(patient_id, fibrosis_stage, resolution)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            # Construct full path and save
            img_filename = get_image_filename(patient_id, modality, IMAGE_FILE_EXTENSION)
            img_path = os.path.join(images_path, img_filename)
            save_image(img, img_path)

            count += 1
            if count % 100 == 0:
                print(f"  Progress: {count}/{total_images} images saved")

    print(f"All {total_images} images saved successfully.")

    # Step 6: Write split-specific CSV files
    print("Writing split CSV files...")
    write_split_csvs(labels_df, metadata_path, prefix="split_")

    # Step 7: Verify split distribution
    actual_dist = verify_split_distribution(labels_df)

    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total patients: {len(labels_df)}")
    print(f"Images per patient: {len(modalities)}")
    print(f"Total images: {total_images}")
    print(f"Split distribution: {actual_dist}")
    print(f"Data saved under: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    generate_dataset()
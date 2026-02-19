import numpy as np
import pandas as pd
from typing import Dict, List

from config import (
    RANDOM_SEED,
    NUM_PATIENTS,
    FIBROSIS_STAGES,
    BINARY_LABEL_MAPPING,
    SPLIT_RATIOS
)
from utils import get_patient_id, assign_split

# -------------------- Constants for Metadata --------------------
AGE_MEAN: float = 60.0
AGE_STD: float = 10.0
AGE_MIN: int = 30
AGE_MAX: int = 85

SEX_OPTIONS: List[str] = ["M", "F"]
SEX_PROBS: List[float] = [0.5, 0.5]  # equal distribution

# Prevalence of fibrosis stages (approximate real-world distribution)
FIBROSIS_PROBS: List[float] = [0.30, 0.25, 0.20, 0.15, 0.10]  # F0 to F4


# -------------------- Patient-Level Label Generation --------------------
def generate_patient_labels(patient_index: int, rng: np.random.RandomState) -> Dict[str, object]:
    """
    Generate labels and metadata for a single patient.

    Args:
        patient_index: Integer index of the patient (0-based).
        rng: NumPy random number generator for deterministic sampling.

    Returns:
        Dictionary containing:
            - patient_id: str
            - fibrosis_stage: str (F0-F4)
            - binary_label: str ("positive" or "negative")
            - split: str ("train", "val", "test")
            - age: int
            - sex: str ("M" or "F")
    """
    patient_id = get_patient_id(patient_index)

    # Sample fibrosis stage according to predefined probabilities
    fibrosis_stage = rng.choice(FIBROSIS_STAGES, p=FIBROSIS_PROBS)

    # Determine binary label based on mapping
    if fibrosis_stage in BINARY_LABEL_MAPPING["positive"]:
        binary_label = "positive"
    else:
        binary_label = "negative"

    # Assign data split based on index (using global SPLIT_RATIOS)
    split = assign_split(patient_index, NUM_PATIENTS, SPLIT_RATIOS)

    # Generate age (normal distribution, clipped to realistic range)
    age = int(np.clip(round(rng.normal(AGE_MEAN, AGE_STD)), AGE_MIN, AGE_MAX))

    # Generate sex
    sex = rng.choice(SEX_OPTIONS, p=SEX_PROBS)

    return {
        "patient_id": patient_id,
        "fibrosis_stage": fibrosis_stage,
        "binary_label": binary_label,
        "split": split,
        "age": age,
        "sex": sex,
    }


# -------------------- Full Dataset Label Generation --------------------
def generate_all_labels(num_patients: int = NUM_PATIENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate labels and metadata for all patients.

    Args:
        num_patients: Total number of patients.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: patient_id, fibrosis_stage, binary_label, split, age, sex.
    """
    # Create a global random number generator
    rng = np.random.RandomState(seed)

    records = []
    for idx in range(num_patients):
        # Use a per-patient RNG seeded from the global one to ensure deterministic but independent sampling
        # This avoids sequence dependence if we later shuffle or process in parallel.
        # Ensure the seed is within the valid range for RandomState (0 to 2**31-1 for signed 32-bit).
        patient_seed = rng.randint(0, 2**31 - 1)
        patient_rng = np.random.RandomState(patient_seed)
        record = generate_patient_labels(idx, patient_rng)
        records.append(record)

    df = pd.DataFrame(records)
    return df


# -------------------- Entry Point (if run as script) --------------------
if __name__ == "__main__":
    # Quick test: generate labels and print summary
    df = generate_all_labels()
    print(f"Generated labels for {len(df)} patients.")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nStage distribution:")
    print(df["fibrosis_stage"].value_counts(normalize=True).sort_index())
    print("\nSplit distribution:")
    print(df["split"].value_counts(normalize=True))
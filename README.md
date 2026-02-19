# Open Multi-Modal Liver Cirrhosis Dataset

A fully synthetic, open-access dataset of multi-modal medical images (MRI, CT, Ultrasound) for liver cirrhosis research, designed for machine learning and Vision Transformer applications.

## Overview

This dataset contains synthetic images from 1000 unique patients, each with three modalities: MRI, CT, and Ultrasound. All images are generated using parametric models that simulate realistic anatomy and pathology variations. The dataset includes ground truth labels for liver fibrosis stage (METAVIR F0–F4) and a binary classification (cirrhosis vs. non-cirrhosis). Data is split into training (70%), validation (15%), and test (15%) sets.

**Key Features:**
- **Multi-Modal:** MRI (T2-weighted), CT, and Ultrasound for each patient.
- **Realistic Synthetic Generation:** Patient-specific anatomy (liver shape, position, texture) and pathology-dependent features (nodules, echogenicity, density).
- **Clinical Labels:** Fibrosis stage (F0–F4) and binary cirrhosis indicator.
- **Structured Metadata:** Age, sex, and split assignment per patient.
- **Reproducible:** Fixed random seed ensures exact replication.

## Dataset Structure


Each image is saved as a 2D NumPy array (`.npy`) with float32 values in the range [0, 1].

## Metadata

The main `labels.csv` contains the following columns:

| Column           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `patient_id`     | Unique patient identifier (e.g., `PAT_0001`)                                |
| `fibrosis_stage` | METAVIR score: F0, F1, F2, F3, F4                                           |
| `binary_label`   | `positive` (cirrhosis: F3–F4) or `negative` (no/mild fibrosis: F0–F2)      |
| `split`          | Dataset split: `train`, `val`, or `test`                                    |
| `age`            | Patient age (integer, 30–85)                                                |
| `sex`            | Patient sex (`M` or `F`)                                                    |

## Generation Process

1. **Anatomy Model:** For each patient, a unique elliptical liver region is defined with random position, size, and rotation. These parameters are derived from a deterministic seed based on `patient_id` to ensure cross-modality consistency.
2. **Modality-Specific Synthesis:**
   - **MRI:** Background noise with elliptical liver region. Intensity increases with fibrosis stage; bright nodules added for F2–F4.
   - **CT:** Simulated Hounsfield units normalized to [0,1]. Liver density increases with stage; high-density spots appear in F3–F4.
   - **Ultrasound:** Speckle noise modelled by Gamma distribution. Echogenicity and texture coarseness increase with stage; bright reflections for advanced fibrosis.
3. **Label Assignment:** Fibrosis stages are sampled from a realistic prevalence distribution. Binary label derived from stage. Age sampled from N(60,10), sex balanced.
4. **Data Splitting:** Patients are deterministically assigned to train/val/test based on index to ensure reproducibility without shuffling.
``python
import numpy as np
image = np.load("OpenMultiModalLiverCirrhosisDataset/images/PAT_0001_MRI.npy")

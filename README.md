# OpenMultiModalLiverCirrhosisDataset

🧠🩺 **Synthetic Multi-Modal Liver Cirrhosis Dataset**

*For Machine Learning Research & Medical Imaging Applications*

**Designed & Developed by Dr. Sanjay Agal**

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)
![Images](https://img.shields.io/badge/Modalities-MRI%20%7C%20CT%20%7C%20Ultrasound-orange)
![Patients](https://img.shields.io/badge/Patients-1000%2B-brightgreen)
![Synthetic](https://img.shields.io/badge/Data-100%25%20Synthetic-blueviolet)
![MadeWith](https://img.shields.io/badge/Made%20with-Python-yellow)
![Developer](https://img.shields.io/badge/Developed%20By-Dr.%20Sanjay%20Agal-purple)

---

## ✨ Overview

This repository provides a **fully synthetic, open-access, multi-modal medical imaging dataset** designed to support **machine learning, deep learning, and Vision Transformer (ViT)** research for **early and advanced liver cirrhosis detection**.

The dataset simulates **realistic clinical imaging scenarios** using parametric and probabilistic models, enabling safe, reproducible, and large-scale experimentation **without any real patient data**.

It is ideal for:

- 🧠 Vision Transformer (ViT) benchmarking  
- 🔗 Multi-modal fusion research  
- 📊 Medical image classification  
- 🔍 Explainable AI (XAI)  
- 🎓 Academic teaching and demonstrations  

---

## 🧬 Dataset Summary

- **Patients:** 1,000+ synthetic subjects  
- **Modalities:** MRI, CT, Ultrasound  
- **Labels:**
  - METAVIR Fibrosis Stage (F0–F4)
  - Binary Cirrhosis Label (Positive / Negative)
- **Splits:** Train (70%) · Validation (15%) · Test (15%)
- **Format:** NumPy arrays (`.npy`, float32, [0–1])
- **Reproducibility:** Fixed random seed  

---

## 📁 Dataset Structure


---

## 📋 Metadata Description

The `labels.csv` file contains:

| Column | Description |
|------|-------------|
| `patient_id` | Unique synthetic patient ID |
| `fibrosis_stage` | METAVIR stage (F0–F4) |
| `binary_label` | positive (F3–F4) / negative (F0–F2) |
| `split` | train / val / test |
| `age` | Patient age (30–85) |
| `sex` | M / F |

---

## ⚙️ Synthetic Generation Highlights

- Patient-specific liver anatomy (shape, size, rotation)
- Modality-consistent geometry across MRI, CT, and Ultrasound
- Progressive texture and intensity changes with fibrosis stage
- Realistic noise models:
  - Gaussian (MRI)
  - Density-based (CT)
  - Speckle (Ultrasound)

---

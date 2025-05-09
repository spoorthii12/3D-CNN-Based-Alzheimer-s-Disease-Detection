# 🧠 Alzheimer's Disease Classification with 3D CNN and Tabular Data

This project presents a deep learning pipeline to classify Alzheimer's disease (nondemented vs. demented) using a combination of 3D brain MRI scans and demographic tabular data. It is built using the [OASIS-2 longitudinal dataset](https://www.oasis-brains.org/), incorporating both image-based and clinical information for better accuracy.

---

## 🗂 Dataset

- **Source**: OASIS-2 longitudinal dataset
- **Subjects**: 150 individuals aged 60–96, with 373 total MRI sessions
- **Classes**:
  - 206 nondemented
  - 167 demented (including MCI/AD)
- **Features**:
  - 3D T1-weighted MRI scans, preprocessed to shape `(128×128×128)`
  - Tabular: `Age`, `EDUC`, `SES`, `MMSE`
- **Split**:
  - 297 training scans
  - 76 validation scans (using `GroupShuffleSplit` to avoid subject overlap)

---

## 🧬 Project Structure

```bash
├── preprocessing.py           # Converts raw MRI (.nii/.img) to normalized .npy volumes
├── train.py                   # Data loading, training, validation, evaluation, plotting
├── model.py                   # CNN3DTabular model definition
├── data/
│   └── processed/             # Output .npy volumes used for training
├── plots/                     # Loss, accuracy, and confusion matrix plots
└── 3DACNN-Based Alzheimer's Disease Detection.pdf  # Presentation slides

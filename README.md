_**Alzheimer’s Disease Classification with 3D CNN and Tabular Data**__


Overview
This project develops a deep learning model to classify Alzheimer’s disease (nondemented vs. demented) using 3D MRI scans and tabular demographic data. We use the OASIS-2 dataset, which includes 373 T1-weighted MRI scans from 150 subjects, along with features like Age, Education (EDUC), Socioeconomic Status (SES), and Mini-Mental State Examination (MMSE) scores. The hybrid CNN3DTabular model combines a 3D CNN for MRI processing with an MLP for tabular data, achieving a validation balanced accuracy of 0.7820.
Dataset

Source: OASIS-2 longitudinal dataset
Size: 373 MRI sessions from 150 subjects (206 nondemented, 167 demented)
Features:
    3D MRI scans (T1-weighted, preprocessed to 128x128x128)
    Tabular: Age, EDUC, SES, MMSE


Split: 297 training, 76 validation (80:20 split using GroupShuffleSplit)

Project Structure

preprocessing.py: 
  Converts raw MRI scans (Analyze/NIfTI format) into 128x128x128 .npy files with normalization and anti-aliasing.
train.py: 
  Handles data loading, augmentation, training, and evaluation of the CNN3DTabular model. Generates plots for loss, accuracy, and confusion matrix.
model.py: 
  Defines the CNN3D model architecture (used in early iterations).
3DACNN-Based Alzheimer's Disease Detection.pdf: 
  Presentation slides detailing the project, including objectives, preprocessing, model architecture, training, and results.

Model Architecture
The CNN3DTabular model (from train.py) integrates:

3D CNN: Processes MRI scans (128x128x128x1) with three Conv3D layers (8, 16, 32 filters), max pooling, and dropout (0.3).
MLP: Processes tabular features (4 units → 64 units) with dropout (0.6).
Fusion: Concatenates CNN and MLP outputs, followed by a classifier (131,136 → 128 → 1) with dropout (0.6).

An earlier CNN3D model (in model.py) used only MRI data with a simpler architecture.
Key Preprocessing Steps

MRI:
Loaded raw scans from OAS2_RAW_PART1/2.
Removed singleton dimensions, normalized to [0,1], resized to 128x128x128 with anti-aliasing, saved as .npy (float32).


Tabular:
Normalized Age, EDUC, SES, MMSE using StandardScaler.


Data Cleaning:
Matched MRI files with tabular data; handled missing files.


Splitting & Balancing:
80:20 split with GroupShuffleSplit (no subject overlap).
WeightedRandomSampler for class imbalance (206:167).



Training

Setup:
15 epochs, batch size 4, Adam optimizer (lr=1e-3, weight_decay=1e-4).
Loss: BCEWithLogitsLoss with pos_weight=1.2336.


Augmentation:
Training: Random rotation (±15°), horizontal flip (p=0.3).
Validation: No augmentation.


Results:
Best Model (Epoch 13): Validation balanced accuracy 0.7820, train-val gap 0.0193.
Confusion Matrix: Recall (Control: 0.80, AD: 0.76), Precision (Control: 0.87, AD: 0.66).



Setup Instructions

Clone the Repository:git clone https://github.com/your-username/alzheimers-classification.git
cd alzheimers-classification


Install Dependencies:pip install torch numpy pandas scikit-learn matplotlib seaborn nibabel skimage tqdm


Download Data:
Obtain the OASIS-2 dataset (OAS2_RAW_PART1, OAS2_RAW_PART2, and oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx).
Place raw MRI data in a directory and update INPUT_DIRS in preprocessing.py.
Place the demographics file in the root directory.


Preprocess Data:python preprocessing.py

This generates .npy files in data/processed/.
Train the Model:python train.py

This trains the model, saves the best model as best_model.pt, and generates plots in the plots/ directory.

Results

Performance:
Validation Balanced Accuracy: 0.7820
Train-Val Gap: 0.0193 (indicating minimal overfitting)


Plots:
Training loss and accuracy trends (loss_plot.png, accuracy_plot.png).
Confusion matrix (confusion_matrix.png): 41/51 nondemented and 19/25 demented correctly classified.



Usage
To use the trained model for inference:

Load the saved model (best_model.pt) and the CNN3DTabular class from train.py.
Preprocess new MRI scans using preprocessing.py and normalize tabular data with StandardScaler.
Pass the data through the model for predictions.

Acknowledgments

Dataset: OASIS-2 (Open Access Series of Imaging Studies)
Tools: PyTorch, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Nibabel


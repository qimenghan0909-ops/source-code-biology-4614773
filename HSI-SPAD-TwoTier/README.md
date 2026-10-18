# HSI-SPAD-TwoTier
## Two-Tier Grid Search Optimization Framework for Hyperspectral-Based SPAD Estimation

## Overview

This repository contains the code for the two-tier grid search optimization framework for hyperspectral-based SPAD estimation in maize leaves. The framework systematically evaluates 672 method combinations across four modeling stages (preprocessing, wavelength selection, sample partitioning, and regression modeling), followed by hyperparameter optimization of the selected pipeline.

## Features

- **672 Method Combinations**: Systematic evaluation of preprocessing methods, wavelength selection algorithms, sample partitioning strategies, and regression models
- **Two-Tier Optimization**: First-tier grid search for method selection, second-tier for hyperparameter tuning
- **Pixel-Level SPAD Mapping**: Generate spatial distribution maps of SPAD values across maize leaves
- **GLCM Texture Analysis**: Extract texture features and leaf coverage from hyperspectral images
- **Multiple Algorithms**: Support for 8 preprocessing methods, 6 wavelength selection methods, 3 partitioning strategies, and 4 regression models


## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd SPAD_2
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Data Preparation

Prepare your data in the following format:
- **Spectral data**: Excel file with shape (n_samples, n_wavelengths)
- **SPAD labels**: Excel file with shape (n_samples, 2) containing [SampleID, SPAD_Mean]

### Step 2: First-Tier Grid Search

Evaluate 672 method combinations to identify the optimal pipeline:

```bash
python "tier1_algorithm combinations.py"
```

This will evaluate all combinations of:
- 8 preprocessing methods (None, MA, SG, MSC, D1, D2, DT, WAVE)
- 7 wavelength selection methods (None, CARS, LARS, UVE, SPA, GA, PCA)
- 3 sample partitioning methods (random, SPXY, KS)
- 4 regression models (PLSR, SVR, ELM, MLR)


### Step 3: Second-Tier Hyperparameter Optimization

Optimize hyperparameters for the selected pipeline (MSC + CARS + SPXY + SVR):

```bash
python "tier2_hyperparameter optimization.py"
```

This optimizes:
- CARS iterations (2-50)
- SVR penalty parameter C (1-100)
- SVR kernel type (linear, rbf)

### Step 4: Pixel-Level SPAD Prediction

Generate spatial distribution maps of SPAD values:

```bash
# 1. Train and save the model
python SPAD_predicted/save_model.py

# 2. Smooth pixel spectra
python SPAD_predicted/Smoothing_spectrum.py

# 3. Predict SPAD for each pixel
python SPAD_predicted/Predicting_spectrapy.py

# 4. Generate SPAD pseudo-color maps
python SPAD_predicted/SPAD.py
```

### Step 5: GLCM Texture Analysis

Extract texture features and calculate leaf coverage:

```bash
python texture_and_coverage_analysis.py
```

This script:
- Extracts 14 Haralick texture features from hyperspectral images
- Calculates leaf coverage (ratio of leaf pixels to total pixels)
- Saves results to Excel files

## Data Format

### Input Data

- **Spectral data**: Excel file (.xlsx) with samples as rows and wavelengths as columns
- **SPAD labels**: Excel file (.xlsx) with columns [SampleID, SPAD_Mean]

### Output Data

- **Grid search results**: Excel files containing R², RMSE, MAE for each combination
- **Texture features**: Excel file with 14 GLCM features per sample
- **Leaf coverage**: Excel file with leaf pixel count and coverage ratio
- **SPAD maps**: PNG images showing spatial distribution of SPAD values

## Data Availability

The code and anonymized dataset will be made publicly available upon acceptance of the manuscript to support reproducibility and facilitate future research.

## Citation

If you use this code in your research, please cite our paper:

```
[Paper citation will be added after publication]
```

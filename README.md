# TopDIA Evaluation Scripts

This repository contains scripts for evaluating the performance of TopDIA for top-down data-independent acquisition mass spectrometry (TD-DIA-MS) data analysis.

### Data Download
You can download the necessary data from [TopDIA Published Data](https://wavetulane-my.sharepoint.com/:f:/g/personal/xwliu_tulane_edu/EnwzHddNSWJLminnB6IdY_gBrXf_WJ8JJ1kLTH02aEPPbg?e=XKjcrp).

## Setup Instructions

1. **Create a Directory**: 
   - Name the directory `TopDIA`.
   
2. **Copy Files**: 
   - Clone or download the `TopDIA_Evaluation_Scripts` repository and place it inside the `TopDIA` folder.
   - Download the `TopDIA_Published_Data` and place it inside the same `TopDIA` folder.

   The scripts are designed to automatically load data from the `TopDIA_Published_Data` folder.

## Script Usage

### 1. Train the Logistic Regression Model
Use this script to train a logistic regression model for evaluating proteoform and fragment feature pairs:
   ```bash
   python3 00_train_model.py
   ```

### 2. Compare DDA and DIA Performance
This script compares the performance of Data-Dependent Acquisition (DDA) and Data-Independent Acquisition (DIA) data:
   ```bash
   python3 01_compare_dda_dia.py
   ```

### 3. Compare TopPIC Pipeline and Pseudo Pipeline
Use this script to compare the performance of the TopPIC pipeline with a pseudo pipeline using DIA data:
   ```bash
   python3 02_compare_toppic_pseudo_pipeline.py
   ```

### 4. Compare Reproducibility of DIA and DDA Data
This script assesses the reproducibility of DIA and DDA data:
   ```bash
   python3 03_compare_reproducibility.py
   ```

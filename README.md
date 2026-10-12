# Financial-Assessment-Classifier-Neural-Network

A PyTorch neural network that classifies survey respondents into three federal poverty-level categories using household and financial characteristics.

## Overview

This project explores how a feedforward neural network can learn patterns in household financial data to predict a respondent's federal poverty-level category. I built the model as part of an AI program at Stanford University Pre-Collegiate Studies.

The project covers data cleaning, feature selection, class balancing, neural network design, model training, and evaluation on held-out data.

**Test accuracy: 91.3%**

## Dataset

- **Source:** [National Financial Well-Being Survey, 2016 Public Use File](https://www.kaggle.com/anthonyku1031/nfwbs-puf-2016-data/data)
- **Original publisher:** Consumer Financial Protection Bureau (CFPB)
- **Target:** Three federal poverty-level categories, represented by the `fpl` variable.
- **Features:** 10 household and financial characteristics.

The selected features include household income, employment status, education, household size, presence of children, housing, savings range, ability to make ends meet, SNAP participation, and age category.

## Methodology

### 1. Data preprocessing

- Selected 10 input features and the `fpl` target variable.
- Replaced designated nonresponse codes with column medians or modes.
- Balanced the dataset to 600 observations per class, resulting in 1,800 observations.
- Converted the target labels to zero-based class indices for PyTorch.

### 2. Training and evaluation

- Split the data into 1,350 training observations and 450 test observations using a stratified 75/25 split.
- Standardized features using `StandardScaler`, fitted on the training set.
- Set random seeds for reproducibility.

### 3. Neural network architecture

Implemented a feedforward neural network in PyTorch with:

- 10 input features
- Two hidden layers containing 64 and 32 neurons
- ReLU activation functions
- Three output classes
- Cross-entropy loss
- Adam optimizer

Trained the network for 400 epochs and compared learning rates before selecting 0.001.

## Results

The model achieved **91.3% accuracy** on the held-out test set.

The prediction function accepts a respondent's feature values and returns a predicted class, its softmax confidence score, and a bar chart of the predicted probabilities across all three classes.

## Limitations

- The dataset was balanced by selecting 600 observations from each class, so the resulting class distribution does not reflect the original survey distribution.
- Household income is closely related to federal poverty-level classification and may account for a substantial portion of the model's predictive performance.
- The test accuracy reflects this particular dataset and preprocessing approach. It does not establish how well the model would generalize to other populations or future data.
- The model is an educational project, not a validated tool for determining an individual's financial circumstances or eligibility for assistance.

## Technologies

Python · PyTorch · pandas · NumPy · scikit-learn · Matplotlib · Google Colab

## Run the notebook

1. Open `FinalProject.ipynb` in Google Colab.
2. Download the dataset from the Kaggle link above.
3. Run the notebook cells in order and upload `NFWBS_PUF_2016_data.csv` when prompted.
4. Follow the preprocessing, training, evaluation, and prediction steps.


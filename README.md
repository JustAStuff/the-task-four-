# the-task-four-
# Breast Cancer Classification - Logistic Regression

This project uses logistic regression to classify breast cancer tumors as malignant or benign using the UCI Breast Cancer Wisconsin dataset.

## Objective

To build a binary classifier that predicts whether a tumor is malignant or benign.

## Tools Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

- Source: [Kaggle - Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- File: data.csv

## Steps Performed

1. Loaded and cleaned the dataset
2. Encoded the target variable (`M` = 1, `B` = 0)
3. Split the data into training and test sets
4. Standardized the features
5. Trained a Logistic Regression model
6. Evaluated the model using:
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC Curve and AUC Score
7. Explained the output and visualized the ROC curve

## How to Run

1. Install required libraries:
   ```bash
   pip install pandas matplotlib scikit-learn

# Breast Cancer Prediction using Machine Learning

A machine learning mini project that predicts whether a breast tumour is **malignant** or **benign** from measurements of its cell nuclei. Five classification algorithms are trained and compared, and the best one reaches **98.25% accuracy** on unseen test data.

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset), which ships with scikit-learn, so no download is needed.

- 569 samples (212 malignant, 357 benign)
- 30 numeric features such as radius, texture, perimeter, area and smoothness
- No missing values

## Results

| Model | Test accuracy |
|---|---|
| Logistic Regression | 98.25% |
| Support Vector Machine | 98.25% |
| K-Nearest Neighbours | 95.61% |
| Random Forest | 95.61% |
| Decision Tree | 91.23% |

The best model, Logistic Regression, scores **98.07%** on 5-fold cross-validation. On the 114 test samples it misclassifies only two: one malignant tumour predicted as benign and one benign tumour predicted as malignant.

## How it works

1. Load the dataset into a pandas DataFrame
2. Explore the data: class distribution, statistics and a correlation heatmap
3. Split into training (80%) and test (20%) sets with stratification
4. Scale the features with `StandardScaler`
5. Train five classifiers and compare accuracy, precision, recall and F1 score
6. Evaluate the best model with a confusion matrix, classification report and 5-fold cross-validation
7. Rank the most important features using Random Forest
8. Save the model with `joblib` and predict for a new patient

## Files

| File | Description |
|---|---|
| `Breast_Cancer_Prediction_ML_Mini_Project.ipynb` | Jupyter notebook with code, outputs and charts |
| `breast_cancer_prediction.py` | The same code as a plain Python script |

## How to run

**Google Colab:** open the notebook in [Colab](https://colab.research.google.com) and choose **Runtime → Run all**.

**Locally:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
jupyter notebook Breast_Cancer_Prediction_ML_Mini_Project.ipynb
```

or run the script directly:

```bash
python breast_cancer_prediction.py
```

## Tech stack

Python, NumPy, pandas, Matplotlib, Seaborn, scikit-learn

## Future scope

- Hyperparameter tuning with `GridSearchCV`
- Feature selection to remove highly correlated features
- A simple web app for predictions using Streamlit or Flask

## Disclaimer

This project is for learning purposes only and is not a medical diagnostic tool.

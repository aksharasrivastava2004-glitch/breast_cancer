#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction using Machine Learning
# 
# **ML Lab Mini Project**
# 
# **Aim:** To build a machine learning model that predicts whether a breast tumour is **malignant** (cancerous) or **benign** (not cancerous), and to compare five classification algorithms to find the best one.
# 
# **Dataset:** Breast Cancer Wisconsin (Diagnostic) dataset, built into scikit-learn (no download needed).
# - 569 samples, 30 numeric features (radius, texture, perimeter, area, smoothness, etc. of the cell nuclei)
# - Target: `0 = malignant`, `1 = benign`
# 
# **Algorithms compared:** Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Support Vector Machine
# 
# **Steps:**
# 1. Import libraries
# 2. Load the dataset
# 3. Explore the data (EDA)
# 4. Split into train and test sets
# 5. Scale the features
# 6. Train five models and compare them
# 7. Evaluate the best model (confusion matrix, classification report, cross-validation)
# 8. Find the most important features
# 9. Save the model and predict for a new patient

# ## Step 1: Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import warnings
warnings.filterwarnings("ignore")


# ## Step 2: Load the dataset

# In[2]:


data = load_breast_cancer()

# Put the data in a pandas DataFrame so it is easy to look at
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target      # 0 = malignant, 1 = benign

print("Shape of dataset:", df.shape)
df.head()


# ## Step 3: Explore the data (EDA)

# In[3]:


# Check for missing values and data types
print("Missing values in the whole dataset:", df.isnull().sum().sum())
print("Data types:", df.dtypes.unique())


# In[4]:


# Basic statistics of the first few features
df.describe().T.head(10)


# In[5]:


# How many samples are in each class?
counts = df["target"].value_counts().sort_index()
print("Malignant (0):", counts[0])
print("Benign    (1):", counts[1])

plt.figure(figsize=(5, 4))
plt.bar(["Malignant (0)", "Benign (1)"], counts.values, color=["#d9534f", "#5cb85c"])
plt.title("Class Distribution")
plt.ylabel("Number of samples")
plt.show()


# **Observation:** The dataset has no missing values, and all features are numeric, so no encoding or imputation is needed. The classes are slightly imbalanced (more benign than malignant), so we use a *stratified* split in the next step.

# In[6]:


# Correlation heatmap of the first 10 features (the "mean" features)
plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, :10].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (mean features)")
plt.show()


# **Observation:** `mean radius`, `mean perimeter` and `mean area` are almost perfectly correlated (above 0.98), which makes sense because all three measure the size of the tumour.

# ## Step 4: Split into train and test sets

# In[7]:


X = df.drop("target", axis=1)   # features (30 columns)
y = df["target"]                # label

# 80% for training, 20% for testing
# stratify=y keeps the same malignant/benign ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ## Step 5: Scale the features

# In[8]:


# Features have very different ranges (area is in hundreds, smoothness is below 1).
# StandardScaler converts every feature to mean = 0 and standard deviation = 1.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # learn mean/std from training data only
X_test_scaled = scaler.transform(X_test)        # apply the same mean/std to test data


# **Note:** We call `fit_transform` on the training data and only `transform` on the test data. If we fitted the scaler on the test data too, information from the test set would leak into training (*data leakage*).

# ## Step 6: Train five models and compare them

# In[9]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf"),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)        # train
    y_pred = model.predict(X_test_scaled)     # predict on unseen test data

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df = results_df.reset_index(drop=True)
results_df.round(4)


# In[10]:


# Bar chart of test accuracy for every model
plt.figure(figsize=(8, 4))
bars = plt.bar(results_df["Model"], results_df["Accuracy"] * 100, color="#4a90d9")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f"{bar.get_height():.2f}%", ha="center")
plt.ylim(85, 101)
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison on Test Data")
plt.xticks(rotation=15)
plt.show()


# ## Step 7: Evaluate the best model

# In[11]:


best_name = results_df.loc[0, "Model"]
best_model = models[best_name]
y_pred_best = best_model.predict(X_test_scaled)

print("Best model:", best_name)
print("Test accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_best) * 100))


# In[12]:


# Confusion matrix: rows = actual class, columns = predicted class
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - " + best_name)
plt.show()

print("Malignant tumours correctly detected :", cm[0, 0])
print("Malignant tumours missed (said benign):", cm[0, 1])
print("Benign wrongly flagged as malignant   :", cm[1, 0])
print("Benign correctly identified           :", cm[1, 1])


# In[13]:


print(classification_report(y_test, y_pred_best, target_names=["Malignant", "Benign"]))


# In[14]:


# 5-fold cross-validation: a more reliable estimate than a single train/test split.
# The pipeline scales inside each fold, so there is no data leakage.
cv_pipeline = make_pipeline(StandardScaler(), clone(best_model))
cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring="accuracy")

print("Accuracy in each fold:", np.round(cv_scores, 4))
print("Mean CV accuracy: {:.2f}%  (std {:.2f}%)".format(cv_scores.mean() * 100,
                                                        cv_scores.std() * 100))


# ## Step 8: Most important features

# In[15]:


# Random Forest tells us how much each feature helped in making decisions
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 4))
top10.sort_values().plot(kind="barh", color="#5cb85c")
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance")
plt.show()


# ## Step 9: Save the model and predict for a new patient

# In[16]:


# Save the scaler and the best model together in one file
joblib.dump({"scaler": scaler, "model": best_model}, "breast_cancer_model.pkl")
print("Model saved as breast_cancer_model.pkl")


# In[17]:


def predict_tumour(features):
    """Takes the 30 feature values of one patient and returns the prediction."""
    saved = joblib.load("breast_cancer_model.pkl")
    features = np.array(features).reshape(1, -1)      # one row, 30 columns
    features_scaled = saved["scaler"].transform(features)
    result = saved["model"].predict(features_scaled)[0]
    return "Benign" if result == 1 else "Malignant"


# Try it on two patients from the test set that the model has never trained on
for i in [0, 1]:
    patient = X_test.iloc[i].values
    actual = "Benign" if y_test.iloc[i] == 1 else "Malignant"
    print("Patient", i + 1, "-> Predicted:", predict_tumour(patient), "| Actual:", actual)


# ## Conclusion

# In[18]:


best = results_df.loc[0]
print("Five classifiers were trained on the Breast Cancer Wisconsin dataset.")
print("Best model : {}".format(best["Model"]))
print("Accuracy   : {:.2f}%".format(best["Accuracy"] * 100))
print("Precision  : {:.2f}%".format(best["Precision"] * 100))
print("Recall     : {:.2f}%".format(best["Recall"] * 100))
print("F1 Score   : {:.2f}%".format(best["F1 Score"] * 100))
print("5-fold CV  : {:.2f}%".format(cv_scores.mean() * 100))


# All five models scored above 90% on the test data, which shows that the 30 features separate the two classes well. Features related to the **size and shape** of the tumour (area, perimeter, radius, concave points) were the most important.
# 
# **Future scope:** hyperparameter tuning with `GridSearchCV`, feature selection to remove the highly correlated features, and deploying the saved model as a small web app using Streamlit or Flask.

# ---
# ## Viva Questions (quick revision)
# 
# **1. What type of ML problem is this?**
# Supervised learning, binary classification. Supervised because the data has labels; binary because there are two classes.
# 
# **2. Why did you scale the features?**
# The features are on very different ranges. KNN and SVM use distances, and Logistic Regression uses gradient-based optimisation, so large-valued features would dominate. Decision Tree and Random Forest do not need scaling, but it does not harm them.
# 
# **3. Why `fit_transform` on train but only `transform` on test?**
# To avoid data leakage. The test set must behave like unseen data, so the mean and standard deviation are learnt from the training set only.
# 
# **4. What is `stratify=y`?**
# It keeps the same class ratio in the train and test sets as in the full dataset.
# 
# **5. What is `random_state=42`?**
# A seed that makes the split (and the model) give the same result every time the code runs. 42 has no special meaning.
# 
# **6. Explain the confusion matrix.**
# A table of actual vs predicted classes: true positives, true negatives, false positives, false negatives.
# 
# **7. Accuracy, precision, recall, F1?**
# - Accuracy = correct predictions / total predictions
# - Precision = TP / (TP + FP): of everything predicted positive, how much was right
# - Recall = TP / (TP + FN): of all actual positives, how many were found
# - F1 = harmonic mean of precision and recall
# 
# **8. Which metric matters most here?**
# Recall for the malignant class. Missing a cancer (saying benign when it is malignant) is far more dangerous than a false alarm. Note that in this dataset benign is class 1, so the Precision/Recall columns in the comparison table are for the benign class; the classification report shows both classes.
# 
# **9. How does each algorithm work (one line each)?**
# - Logistic Regression: fits a linear boundary and uses the sigmoid function to output a probability.
# - KNN: looks at the k nearest training points and takes a majority vote.
# - Decision Tree: splits the data with if-else questions on features, chosen by Gini impurity.
# - Random Forest: many decision trees trained on random samples and random features; majority vote (bagging).
# - SVM: finds the boundary with the maximum margin between the classes; the RBF kernel handles non-linear data.
# 
# **10. What is cross-validation and why use it?**
# The data is split into 5 folds; the model trains on 4 and tests on 1, repeated 5 times. The average is more reliable than one train/test split.
# 
# **11. What is overfitting?**
# When the model memorises the training data and performs badly on new data. A single Decision Tree overfits easily; Random Forest reduces it by averaging many trees.
# 
# **12. Why was a pipeline used in cross-validation?**
# So that scaling is done separately inside each fold, which avoids data leakage.
# 
# **13. Why save the scaler along with the model?**
# A new patient's values must be scaled with the same mean and standard deviation that were used during training, otherwise the prediction is wrong.

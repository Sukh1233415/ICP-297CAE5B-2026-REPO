# HEART DISEASE PREDICTION
# Week 2
# Binary Classification Project

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Load Dataset

df = pd.read_csv(r"C:Week2\heart.csv")
print("Dataset Loaded Successfully\n")

# Display Dataset

print(df.head())
print("\nDataset Shape")
print(df.shape)
print("\nColumn Names")
print(df.columns)
print("\nDataset Information")
print(df.info())
# Missing Values

print("\nMissing Values")
print(df.isnull().sum())
# Duplicate Values

duplicates = df.duplicated().sum()
print("\nDuplicate Records :", duplicates)
df = df.drop_duplicates()

# Statistical Summary

print("\nSummary Statistics")
print(df.describe())
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
X = df.drop("target", axis=1)
y = df["target"]
# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Data :", X_train.shape)
print("Testing Data :", X_test.shape)
# Logistic Regression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_prediction)
print("\nLogistic Regression Accuracy")
print(lr_accuracy)
# Random Forest

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)
print("\nRandom Forest Accuracy")
print(rf_accuracy)
# Compare Accuracy

print("\nModel Comparison")
print("Logistic Regression :", lr_accuracy)
print("Random Forest :", rf_accuracy)
# Best Model

if rf_accuracy > lr_accuracy:
    print("\nRandom Forest Performs Better")
else:
    print("\nLogistic Regression Performs Better")
# Classification Report

print("\nClassification Report")
print(classification_report(
    y_test,
    rf_prediction
))
# Confusion Matrix

cm = confusion_matrix(
    y_test,
    rf_prediction
)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

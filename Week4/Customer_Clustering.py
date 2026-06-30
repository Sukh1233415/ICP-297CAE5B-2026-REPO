# Customer Clustering using K-Means
# Week 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Load Dataset
df = pd.read_csv(r"Week4\Customer_Data.csv")
print("\nDataset Loaded Successfully")

# Display Dataset
print("\nFirst 5 Records")
print(df.head())
print("\nDataset Shape")
print(df.shape)
print("\nColumn Names")
print(df.columns)
print("\nDataset Information")
print(df.info())
print("\nSummary Statistics")
print(df.describe())

# Check Missing Values
print("\nMissing Values Before Filling")
print(df.isnull().sum())

# Remove Customer ID
df = df.drop("CUST_ID", axis=1)

# Remove Duplicate Records
duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)
df = df.drop_duplicates()

# Replace Missing Values with Mean
numeric_columns = df.select_dtypes(include=["number"]).columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())
print("\nMissing Values After Filling")
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(2,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
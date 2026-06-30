# Customer Clustering Final Report
# Project Title
Customer Clustering using K-Means

# Objective
To segment customers into different groups based on their purchasing behaviour using the K-Means clustering algorithm.

# Dataset
- Dataset Name: Customer_Data.csv
- Total Records: 8950
- Total Features: 18

# Data Preprocessing
The following preprocessing steps were performed:
- Loaded the dataset
- Checked missing values
- Replaced missing values with the mean
- Removed duplicate records
- Removed the CUST_ID column
- Standardized numerical features using StandardScaler

# Algorithm Used
K-Means Clustering

# Feature Scaling
StandardScaler was used to normalize all numerical features before clustering.

# Cluster Evaluation
The clustering performance was evaluated using:
- Elbow Method
- Silhouette Score

# Results
The K-Means model successfully divided customers into three clusters.
Each cluster represents customers with similar purchasing behaviour and financial characteristics.

# Files Generated
- customer_cluster_model.pkl
- clustering_report.txt
- customer_clusters.png 

# Applications
- Customer Segmentation
- Personalized Marketing
- Business Analytics
- Product Recommendation
- Customer Retention

# Conclusion
The Customer Clustering project successfully grouped customers into meaningful clusters using the K-Means algorithm. The project demonstrates data preprocessing, feature scaling, clustering, and evaluation techniques commonly used in machine learning.

# Future Scope
- Hierarchical Clustering
- DBSCAN
- PCA Visualization
- Streamlit Dashboard
- Real-time Customer Segmentation
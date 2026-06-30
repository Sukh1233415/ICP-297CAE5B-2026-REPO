# Technical Notes
# Week 4
# Project Title
Customer Clustering using K-Means

# Objective
The goal of this project is to divide customers into groups. We want to do this by looking at how they buy things and their money situation. We are going to use the K-Means clustering algorithm to make these customer groups. The K-Means clustering algorithm will help us make sense of the customer information. We will use the K-Means clustering algorithm to segment customers into groups that're similar, to each other.

# Project Description
People who buy things from companies can be grouped together because they are similar. This is done by using a kind of computer program that looks at the customers and finds the things they have in common. When we know what these groups are we can understand what the customers like to do and make plans to sell them things.

In this project we used information from credit card customers to find kinds of customers who are similar, to each other. We used the Credit Card Customer Dataset to do this. The Credit Card Customer Dataset helped us identify customer segments.

# Dataset Information
- Dataset Name: Customer_Data.csv
- Learning Type: Unsupervised Learning
- Algorithm Used: K-Means Clustering

The dataset contains customer information such as:
- Balance
- Purchases
- Cash Advance
- Credit Limit
- Payments
- Minimum Payments
- Purchase Frequency
- Cash Advance Frequency
- Tenure
The CUST_ID column was removed because it is only an identifier and does not contribute to clustering.

# Data Preprocessing
The following preprocessing steps were performed:
- Loaded the dataset using Pandas.
- Checked dataset dimensions.
- Displayed column names.
- Examined data types.
- Checked for missing values.
- Replaced missing values with the mean of each numerical column.
- Removed duplicate records.
- Removed the Customer ID column.
- Standardized numerical features using StandardScaler.

# Feature Scaling
Since different features have different ranges, StandardScaler was applied to normalize the data before clustering.
Benefits of feature scaling:
- Improves clustering accuracy.
- Prevents features with larger values from dominating.
- Ensures all variables contribute equally.

# K-Means Clustering
K-Means is an unsupervised learning algorithm that partitions data into K clusters.
Steps followed:
1. Initialize cluster centroids.
2. Assign each customer to the nearest centroid.
3. Update centroids.
4. Repeat until convergence.

# Elbow Method
The Elbow Method was used to determine the optimal number of clusters.
The Within Cluster Sum of Squares (WCSS) was calculated for different values of K.
The point where the curve starts to flatten indicates the optimal number of clusters.

# Model Evaluation
The clustering model was evaluated using:
- Elbow Method
- Silhouette Score
The Silhouette Score measures how well customers are grouped within their respective clusters.

# Data Visualization
The following visualizations were generated:
- Correlation Heatmap
- Elbow Method Graph
- Customer Cluster Scatter Plot
These visualizations help in understanding relationships among variables and cluster distribution.

# Python Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

# Applications
Customer clustering can be used in:
- Customer Segmentation
- Personalized Marketing
- Product Recommendation
- Customer Retention
- Sales Strategy
- Business Analytics

# Learning Outcomes
After completing this project, the following concepts were learned:
- Data preprocessing
- Handling missing values
- Feature scaling
- K-Means clustering
- Elbow Method
- Silhouette Score
- Data visualization
- Customer segmentation

# Conclusion
The K-Means clustering algorithm did a job of grouping customers based on how they spend their money and what they buy. When we got the data ready and scaled the features right the K-Means clustering algorithm worked even better. The customer groups that the K-Means clustering algorithm came up with can really help companies make choices and create marketing plans that are just right for each customer segment. The K-Means clustering algorithm is very useful, for this kind of thing.
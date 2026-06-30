# Customer_Clustering_Final.py
import warnings 
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv(r"Week4\Customer_Data.csv")

# Display dataset information
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Replace missing values using mean
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].mean())

# Remove CUST_ID column
if "CUST_ID" in df.columns:
    df = df.drop(columns=["CUST_ID"])

# Remove duplicate records
df = df.drop_duplicates()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Elbow Method
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(scaled_data)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Train KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
df["Cluster"] = clusters

# Silhouette Score
score = silhouette_score(scaled_data, clusters)
print("Silhouette Score:", score)

# Cluster distribution
print("\nCluster Distribution:")
print(df["Cluster"].value_counts())

# Plot customer clusters (first two scaled features)
plt.figure(figsize=(7,5))
plt.scatter(scaled_data[:,0], scaled_data[:,1], c=clusters)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Customer Clusters")
plt.savefig("customer_clusters.png", dpi=300)
plt.show()

# Save model
joblib.dump(kmeans, "customer_cluster_model.pkl")

# Report
with open("clustering_report.txt","w") as f:
    f.write("Customer Clustering Report\n")
    f.write(f"Records: {len(df)}\n")
    f.write(f"Features: {len(df.columns)-1}\n")
    f.write("Clusters: 3\n")
    f.write(f"Silhouette Score: {score:.4f}\n")
    f.write("\nCluster Distribution:\n")
    f.write(df["Cluster"].value_counts().to_string())

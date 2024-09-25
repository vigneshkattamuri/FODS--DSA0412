import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset (replace 'customer_data.csv' with your actual dataset)
data = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 2: Data Preprocessing
# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Drop missing values (or you can handle them with imputation)
data = data.dropna()

# Select relevant features for clustering (e.g., TotalSpent, Frequency)
# Assuming the dataset has columns like 'CustomerID', 'TotalSpent', 'Frequency'
features = ['TotalSpent', 'Frequency']  # Adjust as per your dataset
X = data[features]

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply the K-Means Clustering Algorithm
# Choose the optimal number of clusters using the elbow method
wcss = []  # Within-cluster sum of squares
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph to determine the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow plot, choose an optimal number of clusters (e.g., 3 or 4)
optimal_clusters = 3  # Adjust based on the elbow method plot

# Step 5: Train the K-Means model with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# Assign the cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Step 6: Evaluate the Clustering using Silhouette Score
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f"\nSilhouette Score: {sil_score:.2f}")

# Step 7: Visualize the Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['TotalSpent'], y=data['Frequency'], hue=data['Cluster'], palette='viridis')
plt.title('Customer Segments based on Total Spending and Frequency')
plt.xlabel('Total Amount Spent')
plt.ylabel('Frequency of Visits')
plt.show()

# Step 8: Analyze the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Inverse scale to get original values
cluster_df = pd.DataFrame(cluster_centers, columns=features)
print("\nCluster Centers (average values for each segment):")
print(cluster_df)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'customer_data.csv' with your dataset file)
# Assume it contains columns like 'Age', 'AnnualIncome', 'SpendingScore', etc.
data = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset
print("Dataset:")
print(data.head())

# Step 1: Preprocess the data
# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Fill missing values or drop rows/columns as necessary
# For simplicity, we drop rows with missing values here
data = data.dropna()

# Select relevant features for clustering
# For example: 'Age', 'Annual Income', 'Spending Score'
features = ['Age', 'AnnualIncome', 'SpendingScore']  # Adjust based on your dataset
X = data[features]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Use the Elbow Method to find the optimal number of clusters
inertia = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Step 3: Apply K-Means Clustering with the optimal number of clusters
# From the elbow plot, you can choose an appropriate number of clusters, e.g., 4
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['AnnualIncome'], y=data['SpendingScore'], hue=data['Cluster'], palette='viridis', s=100)
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()

# Step 5: Interpret the clusters
# Calculate average values for each feature within each cluster
cluster_summary = data.groupby('Cluster')[features].mean()
print("\nCluster Summary:")
print(cluster_summary)

# Step 6: Save the dataset with the cluster labels for further analysis
data.to_csv('customer_segments.csv', index=False)
print("\nCustomer segmentation complete. Results saved to 'customer_segments.csv'.")

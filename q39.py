import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset (replace 'transaction_data.csv' with your actual file)
data = pd.read_csv('transaction_data.csv')

# Display the first few rows of the dataset
print("Transaction Data Preview:")
print(data.head())

# Step 2: Preprocess the data
# Assuming the dataset has columns 'CustomerID', 'TotalAmount', and 'ItemsPurchased'
# Group by 'CustomerID' and aggregate the data
customer_data = data.groupby('CustomerID').agg({
    'TotalAmount': 'sum',
    'ItemsPurchased': 'sum'
}).reset_index()

# Display the aggregated customer data
print("\nAggregated Customer Data:")
print(customer_data.head())

# Step 3: Standardize the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['TotalAmount', 'ItemsPurchased']])

# Step 4: Apply K-Means clustering
# Determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Choose the optimal K (e.g., based on the elbow plot)
optimal_k = 3  # Adjust this based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 5: Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalAmount', y='ItemsPurchased', hue='Cluster', data=customer_data, palette='viridis', s=100)
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Total Amount Spent')
plt.ylabel('Total Items Purchased')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Step 6: Analyze the clusters
cluster_summary = customer_data.groupby('Cluster').agg({
    'TotalAmount': ['mean', 'count'],
    'ItemsPurchased': 'mean'
}).reset_index()
cluster_summary.columns = ['Cluster', 'AverageTotalAmount', 'CustomerCount', 'AverageItemsPurchased']

print("\nCluster Summary:")
print(cluster_summary)

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming you already have a DataFrame named 'transaction_data' with columns: 'CustomerID', 'TotalAmountSpent', 'FrequencyOfVisits'

# Data preprocessing - No missing values handling or outlier removal is shown here, you might need to perform these steps
# if your data requires it.
import pandas as pd

# Sample transaction data
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'TotalAmountSpent': [100, 250, 80, 120, 300, 150, 200, 180, 90, 210],
    'FrequencyOfVisits': [5, 10, 3, 6, 12, 8, 9, 7, 4, 11]
}

# Create DataFrame
transaction_data = pd.DataFrame(data)

# Display the DataFrame
print(transaction_data)

# Feature selection/engineering
X = transaction_data[['TotalAmountSpent', 'FrequencyOfVisits']]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choosing the number of clusters (K)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# From the plot, choose an appropriate value for K

# Model training
k = 3  # Example: choosing K based on the elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Interpretation and evaluation
transaction_data['Cluster'] = kmeans.labels_
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Scale back cluster centers
transaction_data['ClusterCenterDistance'] = ((X - cluster_centers[transaction_data['Cluster']])**2).sum(axis=1) ** 0.5

# Visualization
plt.scatter(transaction_data['TotalAmountSpent'], transaction_data['FrequencyOfVisits'], c=transaction_data['Cluster'], cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=100, label='Cluster Centers')
plt.xlabel('Total Amount Spent')
plt.ylabel('Frequency of Visits')
plt.title('Customer Segmentation')
plt.legend()
plt.show()

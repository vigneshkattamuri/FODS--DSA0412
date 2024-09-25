import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample dataset: Replace this with your actual dataset
# For example, let's assume the dataset contains features like 'TotalSpent', 'PurchaseFrequency', and 'AverageCartValue'
data = pd.read_csv('customer_shopping_data.csv')

# Define the features (replace with actual column names from your dataset)
X = data[['TotalSpent', 'PurchaseFrequency', 'AverageCartValue']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the K-Means clustering model (Assume we want 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Function to assign a new customer to one of the existing segments
def assign_customer_segment():
    print("Enter new customer data:")
    total_spent = float(input("Total Spent: "))
    purchase_frequency = float(input("Purchase Frequency: "))
    average_cart_value = float(input("Average Cart Value: "))

    # Create feature array for the new customer
    new_customer = np.array([[total_spent, purchase_frequency, average_cart_value]])

    # Scale the features
    new_customer_scaled = scaler.transform(new_customer)

    # Predict the cluster (segment) for the new customer
    segment = kmeans.predict(new_customer_scaled)
    
    # Display the segment result
    print(f"The new customer belongs to segment: {segment[0]}")

# Call the function to assign the new customer to a segment
assign_customer_segment()

import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame (replace this with your actual DataFrame)
data = {
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Sales': [100, 120, 90, 110, 130]
}

sales_data = pd.DataFrame(data)
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

# Step 1: Explore dataset structure
print("Dataset Structure:")
print(sales_data.head())
print("\nDataset Info:")
print(sales_data.info())

# Step 2: Summarize data using descriptive statistics
print("\nDescriptive Statistics:")
print(sales_data.describe())

# Step 3: Visualize distribution of individual variables (Histogram)
plt.figure(figsize=(10, 6))
plt.hist(sales_data['Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 4: Identify outliers (Boxplot)
plt.figure(figsize=(10, 6))
plt.boxplot(sales_data['Sales'], vert=False)
plt.title('Boxplot of Sales')
plt.xlabel('Sales')
plt.grid(True)
plt.show()

# Step 5: Compute covariance and correlation
covariance_matrix = sales_data.cov()
correlation_matrix = sales_data.corr()

print("\nCovariance Matrix:")
print(covariance_matrix)

print("\nCorrelation Matrix:")
print(correlation_matrix)


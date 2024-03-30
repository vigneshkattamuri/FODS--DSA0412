import pandas as pd

# Load datasets
customer_demographics = pd.read_csv("customer_demographics.csv")
user_activity_logs = pd.read_csv("user_activity_logs.csv")
customer_support = pd.read_csv("customer_support.csv")

# Merge customer demographics with user activity logs
merged_data = pd.merge(customer_demographics, user_activity_logs, on="customer_id", how="left")

# Merge with customer support interactions
merged_data = pd.merge(merged_data, customer_support, on="customer_id", how="left")

# Drop duplicates if any
merged_data.drop_duplicates(inplace=True)

# Display the merged dataset
print(merged_data.head())

# Save the merged dataset to a new CSV file
merged_data.to_csv("merged_dataset.csv", index=False)

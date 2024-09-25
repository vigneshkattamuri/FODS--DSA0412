import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample dataset: You can load your own dataset
# In this example, let's assume a simple dataset is in CSV format
# where 'Churn' is the target variable (0: not churned, 1: churned)
data = pd.read_csv('customer_churn_data.csv')

# Assuming the dataset has features like 'UsageMinutes', 'ContractDuration', 'Age', etc.
X = data[['UsageMinutes', 'ContractDuration', 'Age']]  # Feature columns
y = data['Churn']  # Target column

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Function to predict churn for a new customer
def predict_churn():
    print("Enter new customer data:")
    usage_minutes = float(input("Usage Minutes: "))
    contract_duration = float(input("Contract Duration (in months): "))
    age = float(input("Age: "))

    # Create feature array for the new customer
    new_customer = np.array([[usage_minutes, contract_duration, age]])

    # Scale the features
    new_customer_scaled = scaler.transform(new_customer)

    # Make a prediction
    prediction = model.predict(new_customer_scaled)
    
    # Display result
    if prediction == 1:
        print("The customer is likely to churn.")
    else:
        print("The customer is not likely to churn.")

# Call the function to predict churn for a new customer
predict_churn()

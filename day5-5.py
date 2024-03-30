import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample dataset (replace this with your actual dataset)
data = {
    'usage_minutes': [200, 300, 150, 400, 250],
    'contract_duration': [12, 24, 6, 12, 18],
    'age': [30, 45, 25, 50, 35],
    'churn_status': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Splitting features and target variable
X = df.drop('churn_status', axis=1)
y = df['churn_status']

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Allowing user to input features of a new customer
usage_minutes = float(input("Enter usage minutes: "))
contract_duration = float(input("Enter contract duration (in months): "))
age = float(input("Enter age: "))

# Scaling the input features
new_customer = scaler.transform([[usage_minutes, contract_duration, age]])

# Predicting churn status for the new customer
prediction = model.predict(new_customer)
if prediction[0] == 0:
    print("The new customer is predicted not to churn.")
else:
    print("The new customer is predicted to churn.")

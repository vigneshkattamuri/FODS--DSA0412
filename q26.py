import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Simulate a housing dataset (you can replace this with your real dataset)
# Features: area (in square feet), number of bedrooms, location encoded as numeric (e.g., 1, 2, 3 for different locations)
np.random.seed(42)
num_samples = 100

# Simulating some random data for illustration
area = np.random.randint(500, 3500, size=num_samples)  # House area in square feet
bedrooms = np.random.randint(1, 6, size=num_samples)   # Number of bedrooms
location = np.random.randint(1, 4, size=num_samples)   # 1, 2, 3 representing different locations
price = (area * 300) + (bedrooms * 50000) + (location * 100000) + np.random.randint(20000, 50000, size=num_samples)

# Create a DataFrame
df = pd.DataFrame({'area': area, 'bedrooms': bedrooms, 'location': location, 'price': price})

# Step 2: Split the dataset into training and testing sets
X = df[['area', 'bedrooms', 'location']]  # Features
y = df['price']  # Target: House Price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse:.2f}")

# Function to predict house price based on user input
def predict_price(area, bedrooms, location):
    features = np.array([[area, bedrooms, location]])
    predicted_price = regressor.predict(features)
    return predicted_price[0]

# Step 5: Get user input for house features
print("Enter the features of the new house:")
area = float(input("Area (in square feet): "))
bedrooms = int(input("Number of bedrooms: "))
location = int(input("Location (1, 2, or 3): "))  # In real scenario, map locations to numbers

# Step 6: Predict the price for the new house
predicted_price = predict_price(area, bedrooms, location)

# Output the predicted price
print(f"The predicted price of the house is: ${predicted_price:.2f}")

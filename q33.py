import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'car_data.csv' with your dataset)
data = pd.read_csv('car_data.csv')

# Display the first few rows of the dataset
print("Dataset:")
print(data.head())

# Step 1: Preprocessing the Data
# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Drop or fill missing values (Here, we'll drop any rows with missing values for simplicity)
data = data.dropna()

# Step 2: Select relevant features for linear regression
# Assume the dataset has columns like 'EngineSize', 'Horsepower', 'MPG', 'Weight', 'Price'
features = ['EngineSize', 'Horsepower', 'MPG', 'Weight']  # Adjust as per your dataset
target = 'Price'  # Target variable (price of the car)

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Standardize the features to bring them to a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Build a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model's Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Step 7: Provide Insights - Analyze Feature Coefficients
coefficients = pd.DataFrame(model.coef_, index=features, columns=['Coefficient'])
coefficients['Impact'] = np.where(coefficients['Coefficient'] > 0, 'Positive', 'Negative')
print("\nFeature Coefficients:")
print(coefficients)

# Step 8: Visualize Predictions vs Actual Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Actual vs Predicted Car Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

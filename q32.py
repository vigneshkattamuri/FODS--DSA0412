import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'house_data.csv' with your dataset)
data = pd.read_csv('house_data.csv')

# Display the first few rows of the dataset
print("Dataset:")
print(data.head())

# Step 1: Bivariate Analysis
# Let's assume 'HouseSize' and 'Price' are two relevant columns in the dataset
selected_feature = 'HouseSize'  # Feature selected for linear regression
target = 'Price'  # Target variable

# Scatter plot to visualize the relationship between house size and price
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data[selected_feature], y=data[target])
plt.title(f'Bivariate Analysis: {selected_feature} vs {target}')
plt.xlabel(f'{selected_feature} (Square Feet)')
plt.ylabel(f'{target} (USD)')
plt.show()

# Correlation between the selected feature and the target
correlation = data[[selected_feature, target]].corr().iloc[0, 1]
print(f"\nCorrelation between {selected_feature} and {target}: {correlation:.2f}")

# Step 2: Split the data into training and testing sets
X = data[[selected_feature]]  # Feature matrix
y = data[target]  # Target vector

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model's Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Step 6: Visualize the Regression Line with the Test Data
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[selected_feature], y=y_test, color='blue', label='Actual')
sns.lineplot(x=X_test[selected_feature], y=y_pred, color='red', label='Predicted')
plt.title(f'Regression Line: {selected_feature} vs {target}')
plt.xlabel(f'{selected_feature} (Square Feet)')
plt.ylabel(f'{target} (USD)')
plt.legend()
plt.show()

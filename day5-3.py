import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Sample house data
data = {
    'house_size': [1200, 1500, 1800, 1100, 1350, 1550, 1900, 1250, 1300, 1650],
    'house_price': [250000, 300000, 350000, 200000, 280000, 320000, 370000, 230000, 240000, 310000]
}

# Create DataFrame
house_data = pd.DataFrame(data)

# Explore the data
print(house_data.head())
print(house_data.info())

# Bivariate Analysis
selected_feature = 'house_size'
target_variable = 'house_price'

plt.scatter(house_data[selected_feature], house_data[target_variable])
plt.title('Bivariate Analysis')
plt.xlabel(selected_feature)
plt.ylabel(target_variable)
plt.show()

# Data Preprocessing
X = house_data[[selected_feature]]
y = house_data[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

print("Training R-squared:", r2_train)
print("Testing R-squared:", r2_test)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Visualization
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_test, color='red', label='Predicted')
plt.title('Linear Regression Model')
plt.xlabel(selected_feature)
plt.ylabel(target_variable)
plt.legend()
plt.show()

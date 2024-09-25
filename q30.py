import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler

# Load dataset (replace 'car_data.csv' with your actual dataset)
# Assume the dataset has columns like 'Mileage', 'Age', 'Brand', 'EngineType', 'Price'
data = pd.read_csv('car_data.csv')

# Function to predict the price of a car and show the decision path
def predict_car_price():
    # Get user input for feature columns
    feature_columns = input("Enter feature columns separated by commas (e.g., Mileage,Age,Brand,EngineType): ").split(',')
    target_column = input("Enter the target variable (Price): ")

    # Split the dataset into features (X) and target (y)
    X = data[feature_columns]
    y = data[target_column]

    # Convert categorical variables if any (e.g., 'Brand', 'EngineType') into numerical
    X = pd.get_dummies(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Decision Tree Regressor (CART algorithm)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Get new car features from the user
    print("\nEnter new car data:")
    new_car_data = []
    for feature in feature_columns:
        value = input(f"{feature}: ")
        new_car_data.append(value)

    # Convert categorical inputs to match the one-hot encoding of the training data
    new_car_df = pd.DataFrame([new_car_data], columns=feature_columns)
    new_car_df = pd.get_dummies(new_car_df)

    # Align the new car dataframe with the training feature set to ensure same dummy variables
    new_car_df = new_car_df.reindex(columns=X_train.columns, fill_value=0)

    # Predict the price for the new car
    predicted_price = model.predict(new_car_df)[0]

    # Display the predicted price
    print(f"\nPredicted Price for the new car: ${predicted_price:.2f}")

    # Display the decision path for the prediction
    decision_path = model.decision_path(new_car_df)
    tree_rules = export_text(model, feature_names=list(X.columns))
    print("\nDecision Path (Rules Applied):")
    print(tree_rules)

# Call the function to predict car price and show the decision path
predict_car_price()

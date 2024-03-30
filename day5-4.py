import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the car dataset
car_data = pd.read_csv(r"D:\FODS\DAY5_4.csv")

# Separate features (X) and target variable (y)
X = car_data.drop('price', axis=1)
y = car_data['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing categorical variables
categorical_cols = ['brand', 'engine_type']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)],
    remainder='passthrough')

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Initialize and train the DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train_processed, y_train)

# Allowing user to input features of a new car
mileage = float(input("Enter mileage of the car: "))
age = float(input("Enter age of the car (in years): "))
brand = input("Enter brand of the car: ")
engine_type = input("Enter engine type of the car: ")

# Convert user input to DataFrame
new_car_features = pd.DataFrame({'mileage': [mileage], 'age': [age], 'brand': [brand], 'engine_type': [engine_type]})

# Preprocess the new car features
new_car_features_processed = preprocessor.transform(new_car_features)

# Predicting the price of the new car
prediction = model.predict(new_car_features_processed)

print("Predicted price of the new car:", prediction[0])

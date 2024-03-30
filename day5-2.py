import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Sample patient data
data = {
    'age': [45, 50, 60, 35, 55, 65, 40, 70, 30, 48],
    'gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'blood_pressure': [120, 130, 140, 115, 125, 135, 118, 145, 110, 128],
    'cholesterol': [200, 220, 240, 180, 210, 230, 190, 250, 170, 215],
    'outcome': ['Good', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Good', 'Bad', 'Good', 'Bad']
}

# Create DataFrame
patients_data = pd.DataFrame(data)

# Split the data into features and target variable
X = patients_data[['age', 'gender', 'blood_pressure', 'cholesterol']]
y = patients_data['outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: scaling numerical features and encoding categorical features
numeric_features = ['age', 'blood_pressure', 'cholesterol']
categorical_features = ['gender']

# Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the training data
X_train_scaled = preprocessor.fit_transform(X_train)

# Transform the test data
X_test_scaled = preprocessor.transform(X_test)

# Train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_scaled)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Good')
recall = recall_score(y_test, y_pred, pos_label='Good')
f1 = f1_score(y_test, y_pred, pos_label='Good')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Display predictions on the test set
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset (replace 'patient_data.csv' with your dataset)
data = pd.read_csv('patient_data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Step 2: Data Preprocessing
# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Drop or fill missing values (Here, we drop rows with missing values for simplicity)
data = data.dropna()

# Encode categorical features like 'Gender' and 'Outcome' (assuming 'Outcome' is 'Good' or 'Bad')
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Encoding Gender (e.g., Male=1, Female=0)
data['Outcome'] = label_encoder.fit_transform(data['Outcome'])  # Encoding Outcome ('Good'=1, 'Bad'=0)

# Step 3: Select features and target
# Assume the dataset has columns 'Age', 'Gender', 'BloodPressure', 'Cholesterol', 'Outcome'
features = ['Age', 'Gender', 'BloodPressure', 'Cholesterol']  # Feature columns
target = 'Outcome'  # Target column

X = data[features]  # Feature matrix
y = data[target]  # Target vector

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Build the KNN Classifier model
knn_model = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors
knn_model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Step 8: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Step 9: Display Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Step 1: Load the dataset (replace 'student_data.csv' with your actual file)
data = pd.read_csv('student_data.csv')

# Display the first few rows of the dataset
print("Student Data Preview:")
print(data.head())

# Assuming the dataset has 'StudyTime' and 'ExamScore' columns
# Step 2: Calculate the correlation coefficient
correlation = data['StudyTime'].corr(data['ExamScore'])
print(f"\nCorrelation Coefficient between Study Time and Exam Scores: {correlation:.2f}")

# Step 3: Visualize the relationship using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='StudyTime', y='ExamScore', data=data, color='blue', s=100, alpha=0.6)
plt.title('Scatter Plot of Study Time vs Exam Scores')
plt.xlabel('Study Time (Hours)')
plt.ylabel('Exam Score')
plt.grid(True)

# Step 4: Fit a linear regression model to visualize the trend
X = data[['StudyTime']]
y = data['ExamScore']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Plot the regression line
plt.plot(data['StudyTime'], predictions, color='red', linewidth=2, label='Regression Line')
plt.legend()
plt.show()

# Step 5: Additional visualizations
# Histogram of Study Time
plt.figure(figsize=(10, 6))
sns.histplot(data['StudyTime'], bins=10, kde=True, color='purple', alpha=0.6)
plt.title('Distribution of Study Time')
plt.xlabel('Study Time (Hours)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Histogram of Exam Scores
plt.figure(figsize=(10, 6))
sns.histplot(data['ExamScore'], bins=10, kde=True, color='orange', alpha=0.6)
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

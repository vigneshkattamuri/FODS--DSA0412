import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

# Sample data
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110],
    'body_fat_percent': [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate mean, median, and standard deviation
mean_age = df['age'].mean()
median_age = df['age'].median()
std_dev_age = df['age'].std()

mean_body_fat = df['body_fat_percent'].mean()
median_body_fat = df['body_fat_percent'].median()
std_dev_body_fat = df['body_fat_percent'].std()

print("Age:")
print("Mean:", mean_age)
print("Median:", median_age)
print("Standard Deviation:", std_dev_age)
print("\nBody Fat %:")
print("Mean:", mean_body_fat)
print("Median:", median_body_fat)
print("Standard Deviation:", std_dev_body_fat)

# Draw boxplots
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y='age', data=df)
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y='body_fat_percent', data=df)
plt.title('Boxplot of Body Fat %')
plt.tight_layout()
plt.show()

# Draw scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['body_fat_percent'])
plt.title('Scatter Plot of Age vs. Body Fat %')
plt.xlabel('Age')
plt.ylabel('Body Fat %')
plt.grid(True)
plt.show()

# Draw q-q plot
sm.qqplot_2samples(df['age'], df['body_fat_percent'], line='45')
plt.title('Q-Q Plot of Age vs. Body Fat %')
plt.xlabel('Age')
plt.ylabel('Body Fat %')
plt.grid(True)
plt.show()

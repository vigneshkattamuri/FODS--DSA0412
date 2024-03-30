import numpy as np
import matplotlib.pyplot as plt

# Scores data
scores = [85, 78, 92, 89, 67, 76, 94, 82, 91, 88, 75, 81, 96, 90, 79, 83, 77, 85, 98, 72]

# Step 1: Calculate mean, median, and quartiles
mean_score = np.mean(scores)
median_score = np.median(scores)
quartiles = np.percentile(scores, [25, 50, 75])

# Step 2: Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(scores)
plt.title('Box Plot of Exam Scores')
plt.ylabel('Scores')
plt.show()

# Step 3: Identify potential outliers
q1, q3 = quartiles[0], quartiles[2]
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = [score for score in scores if score < lower_bound or score > upper_bound]

print("Mean Score:", mean_score)
print("Median Score:", median_score)
print("Quartiles (Q1, Q2, Q3):", quartiles)
print("Potential Outliers:", outliers)

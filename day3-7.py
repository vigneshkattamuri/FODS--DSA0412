import numpy as np
import matplotlib.pyplot as plt

# Time spent data
time_spent = [8, 12, 5, 15, 10, 20, 7, 18, 25, 8, 12, 22, 15, 10, 30, 7, 18, 15, 20, 12]

# Calculate the median
median_time_spent = np.median(time_spent)

# Calculate the interquartile range (IQR)
q1, q3 = np.percentile(time_spent, [25, 75])
iqr = q3 - q1

# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(time_spent)
plt.title('Box Plot of Time Spent on Website')
plt.xlabel('Time Spent (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print the median and interquartile range
print("Median time spent on the website:", median_time_spent)
print("Interquartile Range (IQR):", iqr)

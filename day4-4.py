import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Example data for the clinical trial
placebo_group = np.array([85, 88, 90, 92, 84, 87, 89, 91, 83, 86])
treatment_group = np.array([79, 82, 81, 85, 80, 84, 83, 87, 78, 81])

# Perform hypothesis test (assuming unequal variances)
t_stat, p_value = stats.ttest_ind(placebo_group, treatment_group, equal_var=False)

# Visualization
plt.figure(figsize=(12, 6))

# Boxplot to visualize the distributions of the two groups
plt.subplot(1, 2, 1)
plt.boxplot([placebo_group, treatment_group], labels=['Placebo', 'Treatment'], patch_artist=True, boxprops=dict(facecolor='none', color='black'))
plt.title('Boxplot of Placebo and Treatment Groups')
plt.ylabel('Score')

# Annotation for p-value
plt.text(0.5, 0.9, f'p-value = {p_value:.4f}', ha='center', va='center', transform=plt.gca().transAxes)

# Histograms to visualize the distributions of the two groups
plt.subplot(1, 2, 2)
plt.hist(placebo_group, alpha=0.5, label='Placebo', color='blue', edgecolor='black')
plt.hist(treatment_group, alpha=0.5, label='Treatment', color='green', edgecolor='black')
plt.title('Histogram of Placebo and Treatment Groups')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Print the p-value and conclusion
print("Hypothesis Test:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The new treatment has a statistically significant effect compared to the placebo.")
else:
    print("Fail to reject the null hypothesis: There is no statistically significant effect of the new treatment compared to the placebo.")


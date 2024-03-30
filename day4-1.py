import numpy as np
from scipy import stats

# Example data for two groups
group1_scores = np.array([85, 88, 90, 92, 84, 87, 89, 91, 83, 86])
group2_scores = np.array([79, 82, 81, 85, 80, 84, 83, 87, 78, 81])

# Calculate mean scores for each group
mean_group1 = np.mean(group1_scores)
mean_group2 = np.mean(group2_scores)

# Calculate standard error for each group
se_group1 = np.std(group1_scores, ddof=1) / np.sqrt(len(group1_scores))
se_group2 = np.std(group2_scores, ddof=1) / np.sqrt(len(group2_scores))

# Construct 95% confidence intervals for the mean scores
ci_group1 = stats.norm.interval(0.95, loc=mean_group1, scale=se_group1)
ci_group2 = stats.norm.interval(0.95, loc=mean_group2, scale=se_group2)

print("Group 1:")
print(f"Mean score: {mean_group1}")
print(f"95% Confidence Interval: {ci_group1}")

print("\nGroup 2:")
print(f"Mean score: {mean_group2}")
print(f"95% Confidence Interval: {ci_group2}")

# Perform hypothesis test (assuming unequal variances)
t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores, equal_var=False)

print("\nHypothesis Test:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")

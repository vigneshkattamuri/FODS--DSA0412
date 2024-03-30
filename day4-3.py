import numpy as np
from scipy import stats

# Example data for conversion rates of two website designs
design_A_conversions = np.array([25, 30, 28, 32, 27, 29, 31, 26, 33, 30])
design_B_conversions = np.array([22, 24, 26, 27, 25, 23, 28, 26, 29, 27])

# Perform two-sample independent t-test
t_stat, p_value = stats.ttest_ind(design_A_conversions, design_B_conversions)

print("Hypothesis Test:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a statistically significant difference in mean conversion rates between the two website designs.")
else:
    print("Fail to reject the null hypothesis: There is no statistically significant difference in mean conversion rates between the two website designs.")

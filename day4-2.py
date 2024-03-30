import numpy as np
from scipy import stats

# Example data for two product types (lifespans in hours)
product1_lifespans = np.array([1500, 1550, 1520, 1480, 1510, 1530, 1495, 1470, 1485, 1540])
product2_lifespans = np.array([1450, 1460, 1430, 1480, 1420, 1445, 1475, 1490, 1465, 1410])

# Calculate point estimates for the mean lifespan of each product type
mean_product1 = np.mean(product1_lifespans)
mean_product2 = np.mean(product2_lifespans)

# Calculate standard error for each product type
se_product1 = np.std(product1_lifespans, ddof=1) / np.sqrt(len(product1_lifespans))
se_product2 = np.std(product2_lifespans, ddof=1) / np.sqrt(len(product2_lifespans))

# Construct 90% confidence intervals for the mean lifespans
ci_product1 = stats.norm.interval(0.90, loc=mean_product1, scale=se_product1)
ci_product2 = stats.norm.interval(0.90, loc=mean_product2, scale=se_product2)

print("Product 1:")
print(f"Mean lifespan: {mean_product1}")
print(f"90% Confidence Interval: {ci_product1}")

print("\nProduct 2:")
print(f"Mean lifespan: {mean_product2}")
print(f"90% Confidence Interval: {ci_product2}")

# Perform hypothesis test (assuming unequal variances)
t_stat, p_value = stats.ttest_ind(product1_lifespans, product2_lifespans, equal_var=False)

print("\nHypothesis Test:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a statistically significant difference in lifespans between the two products.")
else:
    print("Fail to reject the null hypothesis: There is no statistically significant difference in lifespans between the two products.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
from sklearn.metrics import silhouette_score

# Sample gene expression data (replace this with your actual data)
data = {
    'Gene_1': [10, 20, 30, 40, 50],
    'Gene_2': [15, 25, 35, 45, 55]
}

gene_expression_data = pd.DataFrame(data)

# i. Develop a novel measure of asymmetry

skewness = np.abs(gene_expression_data.skew())
kurtosis = np.abs(gene_expression_data.kurtosis())
tail_behavior = np.sum(np.abs(gene_expression_data) > gene_expression_data.mean(), axis=0) / len(gene_expression_data)
asymmetry_index = skewness + kurtosis + tail_behavior
print("Asymmetry Index:")
print(asymmetry_index)

# ii. Assess the impact of data transformation

transformers = {
    'StandardScaler': StandardScaler(),
    'Logarithmic': PowerTransformer(method='box-cox', standardize=False)
}

for name, transformer in transformers.items():
    transformed_data = transformer.fit_transform(gene_expression_data)
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(transformed_data)
    control_group = transformed_data[labels == 0]
    treatment_group = transformed_data[labels == 1]
    _, p_value = ttest_ind(control_group, treatment_group)
    print(f"Transformation Method: {name}")
    print(f"Clustering Silhouette Score: {silhouette_score(transformed_data, labels)}")
    print(f"Differential Expression p-value: {p_value}")
    print("\n")

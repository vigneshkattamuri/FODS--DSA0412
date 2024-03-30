import pandas as pd

# Disease data
disease_data = {
    'DISEASE_NAME': ['Common Cold', 'Diabetes', 'Bronchitis', 'Influenza', 'Kidney Stones'],
    'DIAGNOSED_PATIENTS': [320, 120, 100, 150, 60]
}

# Create a DataFrame from the disease data
df = pd.DataFrame(disease_data)

# Sort DataFrame by number of diagnosed patients in descending order
df_sorted = df.sort_values(by='DIAGNOSED_PATIENTS', ascending=False)

# Print the sorted DataFrame
print("Frequency distribution of diseases:")
print(df_sorted)

# Get the most common disease
most_common_disease = df_sorted.iloc[0]['DISEASE_NAME']

# Print the most common disease
print("\nThe most common disease is:", most_common_disease)

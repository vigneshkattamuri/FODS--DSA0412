import pandas as pd

# Step 1: Load the dataset (replace 'temperature_data.csv' with your actual file)
data = pd.read_csv('temperature_data.csv')

# Display the first few rows of the dataset
print("Temperature Data Preview:")
print(data.head())

# Assuming the dataset has columns 'City', 'Date', and 'Temperature'
# Step 2: Calculate mean temperature for each city
mean_temperatures = data.groupby('City')['Temperature'].mean().reset_index()
mean_temperatures.columns = ['City', 'MeanTemperature']

# Step 3: Calculate standard deviation of temperature for each city
std_deviations = data.groupby('City')['Temperature'].std().reset_index()
std_deviations.columns = ['City', 'StdDeviation']

# Step 4: Calculate the temperature range for each city
temperature_ranges = data.groupby('City')['Temperature'].agg(['max', 'min']).reset_index()
temperature_ranges['TempRange'] = temperature_ranges['max'] - temperature_ranges['min']
temperature_ranges = temperature_ranges[['City', 'TempRange']]

# Step 5: Merge the results into a single DataFrame
results = pd.merge(mean_temperatures, std_deviations, on='City')
results = pd.merge(results, temperature_ranges, on='City')

# Display the results
print("\nTemperature Analysis Results:")
print(results)

# Step 6: Determine the city with the highest temperature range
highest_temp_range_city = results.loc[results['TempRange'].idxmax()]
print(f"\nCity with the Highest Temperature Range: {highest_temp_range_city['City']} "
      f"with a range of {highest_temp_range_city['TempRange']}°C")

# Step 7: Find the city with the most consistent temperature (lowest standard deviation)
most_consistent_city = results.loc[results['StdDeviation'].idxmin()]
print(f"City with the Most Consistent Temperature: {most_consistent_city['City']} "
      f"with a standard deviation of {most_consistent_city['StdDeviation']:.2f}°C")

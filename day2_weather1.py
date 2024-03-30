import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'date': ['2/22/2024', '2/13/2024', '2/12/2024', '4/11/2024', '4/12/2024', '4/14/2024', '6/18/2024', '6/24/2024', '8/20/2024', '8/4/2024'],
    'temp': [12, 15, 20, 22, 18, 30, 35, 19, 24, 40],
    'humidity': [30, 32, 36, 40, 28, 31, 36, 20, 42, 60],
    'pressure': [10, 20, 30, 40, 50, 60, 70, 50, 40, 25],
    'wind': [55, 66, 25, 43, 15, 54, 41, 57, 96, 45],
    'precipitation': ['rainy', 'drizzle', 'stormy', 'heavy rain', None, 'rainy', 'stormy', 'rainy', 'heavy rain', 'drizzle']
}

# Load data into DataFrame
df = pd.DataFrame(data)

# Preprocess the dataset
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Visualize the average temperature trends over different seasons
df['month'] = df.index.month
monthly_avg_temp = df.groupby('month')['temp'].mean()

# Plotting temperature trends
plt.figure(figsize=(10, 6))
monthly_avg_temp.plot(kind='line', color='red', marker='o')
plt.title('Average Temperature Trends Over Different Months')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# Identify and visualize any patterns or anomalies in precipitation
plt.figure(figsize=(10, 6))
df['precipitation'].value_counts().plot(kind='bar', color='blue')
plt.title('Precipitation Types')
plt.xlabel('Precipitation')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Plot temperature distributions and highlight extreme weather events
plt.figure(figsize=(10, 6))
plt.hist(df['temp'], bins=20, color='green', edgecolor='black')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.grid(True)

# Highlight extreme weather events
extreme_temps = df[(df['temp'] < -10) | (df['temp'] > 35)]  # Example extreme thresholds
plt.scatter(extreme_temps.index, extreme_temps['temp'], color='red', label='Extreme Events')
plt.legend()
plt.show()

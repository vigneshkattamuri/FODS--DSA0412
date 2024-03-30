import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Temperature': [10, 12, 15, 18, 20, 22],
    'Rainfall': [50, 40, 30, 20, 10, 5]
}

weather_data = pd.DataFrame(data)

# Create a line plot of the monthly temperature data
plt.figure(figsize=(10, 6))
plt.plot(weather_data['Month'], weather_data['Temperature'], marker='o', color='r', linestyle='-')
plt.title('Monthly Temperature Data')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a scatter plot of the monthly rainfall data
plt.figure(figsize=(10, 6))
plt.scatter(weather_data['Temperature'], weather_data['Rainfall'], color='b')
plt.title('Scatter Plot of Rainfall vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.tight_layout()
plt.show()

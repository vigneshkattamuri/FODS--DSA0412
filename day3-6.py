# Sample weather data (replace this with your actual data)
weather_data = {
    'Sunny': 150,
    'Cloudy': 100,
    'Rainy': 80,
    'Snowy': 50,
    'Foggy': 20
}

# Calculate the frequency distribution of weather conditions
weather_freq = {}

for weather, count in weather_data.items():
    weather_freq[weather] = count

# Find the most common weather type
most_common_weather = max(weather_freq, key=weather_freq.get)

# Print the frequency distribution and the most common weather type
print("Frequency Distribution of Weather Conditions:")
for weather, count in weather_freq.items():
    print(f"{weather}: {count}")

print("\nThe most common weather type is:", most_common_weather)

import numpy as np

# Weather data
weather = ['sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'sunny', 'cloudy', 'sunny', 'rainy', 'cloudy',
           'sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'sunny', 'cloudy', 'sunny', 'rainy', 'cloudy',
           'sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'cloudy', 'sunny']

# Count occurrences of each weather type
weather_counts = {}
for condition in weather:
    if condition in weather_counts:
        weather_counts[condition] += 1
    else:
        weather_counts[condition] = 1

# Find the most common weather type
most_common_weather = max(weather_counts, key=weather_counts.get)

# Print frequency distribution
print("Frequency distribution of weather conditions:")
for condition, frequency in weather_counts.items():
    print(f"{condition}: {frequency}")

print("\nThe most common weather type is:", most_common_weather)

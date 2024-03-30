import numpy as np

# Sample data (replace this with your actual dataset)
fuel_efficiency = np.array([25, 30, 28, 32, 27, 29, 31, 26, 33, 35])

# 1. Calculate the average fuel efficiency across all car models
average_fuel_efficiency = np.mean(fuel_efficiency)

# 2. Determine the fuel efficiency of the two specific car models
car_model_1_efficiency = fuel_efficiency[0]  # Example fuel efficiency of car model 1
car_model_2_efficiency = fuel_efficiency[3]  # Example fuel efficiency of car model 2

# 3. Calculate the percentage improvement in fuel efficiency between the two car models
percentage_improvement = ((car_model_2_efficiency - car_model_1_efficiency) / car_model_1_efficiency) * 100

# Print results
print("Average fuel efficiency across all car models:", average_fuel_efficiency, "miles per gallon")
print("Fuel efficiency of car model 1:", car_model_1_efficiency, "miles per gallon")
print("Fuel efficiency of car model 2:", car_model_2_efficiency, "miles per gallon")
print("Percentage improvement in fuel efficiency between car model 1 and car model 2:", percentage_improvement, "%")

import pandas as pd

# Sample DataFrame
data = {
    'Property_ID': [1, 2, 3, 4, 5],
    'Location': ['City A', 'City A', 'City B', 'City B', 'City C'],
    'Num_Bedrooms': [3, 4, 3, 5, 4],
    'Area_Sqft': [1500, 1800, 1600, 2200, 2000],
    'Listing_Price': [250000, 300000, 280000, 350000, 320000]
}

property_data = pd.DataFrame(data)

# Calculate the average listing price of properties in each location
average_price_by_location = property_data.groupby('Location')['Listing_Price'].mean()

# Count the number of properties with more than four bedrooms
num_properties_more_than_4_bedrooms = property_data[property_data['Num_Bedrooms'] > 4].shape[0]

# Find the property with the largest area
property_with_largest_area = property_data.loc[property_data['Area_Sqft'].idxmax()]

print("Average Listing Price by Location:")
print(average_price_by_location)
print("\nNumber of Properties with More than Four Bedrooms:", num_properties_more_than_4_bedrooms)
print("\nProperty with the Largest Area:")
print(property_with_largest_area)

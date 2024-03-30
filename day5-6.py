import pandas as pd

# Creating a sample property_data DataFrame
data = {
    'property_id': [1, 2, 3, 4, 5],
    'location': ['A', 'B', 'A', 'C', 'B'],
    'number_of_bedrooms': [3, 4, 5, 3, 6],
    'area_in_square_feet': [1500, 1800, 2200, 1600, 2500],
    'listing_price': [200000, 250000, 300000, 220000, 350000]
}

property_data = pd.DataFrame(data)

# i. The average listing price of properties in each location
average_price_per_location = property_data.groupby('location')['listing_price'].mean()
print("Average listing price of properties in each location:")
print(average_price_per_location)

# ii. The number of properties with more than four bedrooms
properties_more_than_four_bedrooms = property_data[property_data['number_of_bedrooms'] > 4]
number_of_properties_more_than_four_bedrooms = len(properties_more_than_four_bedrooms)
print("\nNumber of properties with more than four bedrooms:", number_of_properties_more_than_four_bedrooms)

# iii. The property with the largest area
property_with_largest_area = property_data.loc[property_data['area_in_square_feet'].idxmax()]
print("\nProperty with the largest area:")
print(property_with_largest_area)

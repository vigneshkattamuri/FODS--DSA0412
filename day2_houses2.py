import numpy as np

np.random.seed(0)
house_data = np.random.randint(1, 10, size=(100, 5))  
bedrooms_column = 0
sale_price_column = 4
houses_more_than_four_bedrooms = house_data[house_data[:, bedrooms_column] > 4]
sale_prices_more_than_four_bedrooms = houses_more_than_four_bedrooms[:, sale_price_column]
average_sale_price = np.mean(sale_prices_more_than_four_bedrooms)

print("Average sale price of houses with more than four bedrooms:", average_sale_price)
print("AVerage sale price of houses with less than four bedrooms

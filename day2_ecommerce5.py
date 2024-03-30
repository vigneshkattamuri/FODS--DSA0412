import pandas as pd

# Sample data
data = {
    'customer_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'order_date': ['2023-01-01', '2023-02-05', '2023-03-10', '2023-04-15', '2023-05-20', '2023-06-25', '2023-07-30', '2023-08-05', '2023-09-10'],
    'product_name': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'order_quantity': [2, 3, 1, 4, 2, 3, 1, 5, 2]
}

# Create DataFrame
order_data = pd.DataFrame(data)

# Convert 'order_date' column to datetime type
order_data['order_date'] = pd.to_datetime(order_data['order_date'])

# 1. Total number of orders made by each customer
total_orders_per_customer = order_data.groupby('customer_id').size()

# 2. Average order quantity for each product
average_order_quantity_per_product = order_data.groupby('product_name')['order_quantity'].mean()

# 3. Earliest and latest order dates in the dataset
earliest_order_date = order_data['order_date'].min()
latest_order_date = order_data['order_date'].max()

# Printing the results
print("Total number of orders made by each customer:")
print(total_orders_per_customer)

print("\nAverage order quantity for each product:")
print(average_order_quantity_per_product)

print("\nEarliest order date:", earliest_order_date)
print("Latest order date:", latest_order_date)

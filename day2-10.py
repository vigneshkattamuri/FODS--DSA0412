import pandas as pd
import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
dates = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
sales = [100, 120, 90, 110, 130]

# Convert dates to datetime objects
dates = pd.to_datetime(dates)

# Create a DataFrame
sales_data = pd.DataFrame({'Date': dates, 'Sales': sales})

# Sort DataFrame by date
sales_data = sales_data.sort_values(by='Date')

# Create a line plot to visualize sales over time
plt.figure(figsize=(10, 6))
plt.plot(sales_data['Date'], sales_data['Sales'], marker='o', color='b', linestyle='-')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a scatter plot to visualize sales over time
plt.figure(figsize=(10, 6))
plt.scatter(sales_data['Date'], sales_data['Sales'], color='r')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a bar plot to visualize monthly sales data
plt.figure(figsize=(10, 6))
plt.bar(sales_data['Date'], sales_data['Sales'], color='skyblue')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

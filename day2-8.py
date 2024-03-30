import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [10000, 12000, 15000, 11000, 13000, 14000]
}

sales_data = pd.DataFrame(data)

# Create a line plot of the monthly sales data
plt.figure(figsize=(10, 6))
plt.plot(sales_data['Month'], sales_data['Sales'], marker='o', color='b', linestyle='-')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a bar plot of the monthly sales data
plt.figure(figsize=(10, 6))
plt.bar(sales_data['Month'], sales_data['Sales'], color='skyblue')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the stock data (replace 'stock_data.csv' with your actual file)
data = pd.read_csv('stock_data.csv')

# Display the first few rows of the dataset
print("Stock Data Preview:")
print(data.head())

# Assuming the dataset has a 'Date' column and a 'Close' column for closing prices
# Convert 'Date' to datetime format for easier plotting and analysis
data['Date'] = pd.to_datetime(data['Date'])

# Step 2: Basic Statistics on Stock Prices
# Calculate the mean closing price
mean_price = data['Close'].mean()

# Calculate the standard deviation (volatility measure)
std_dev_price = data['Close'].std()

# Display the statistics
print(f"\nMean Closing Price: {mean_price:.2f}")
print(f"Standard Deviation (Volatility): {std_dev_price:.2f}")

# Step 3: Calculate Daily Percentage Change in Stock Prices
data['DailyChange'] = data['Close'].pct_change() * 100  # Percentage change

# Display the first few percentage changes
print("\nDaily Percentage Change in Stock Prices:")
print(data[['Date', 'DailyChange']].head())

# Step 4: Moving Averages to Identify Trends
data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average

# Step 5: Plot the Closing Prices and Moving Averages
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label='Closing Price', color='blue', alpha=0.6)
plt.plot(data['Date'], data['SMA_20'], label='20-Day SMA', color='green', linestyle='--')
plt.plot(data['Date'], data['SMA_50'], label='50-Day SMA', color='red', linestyle='--')
plt.title('Stock Closing Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 6: Analyze Volatility using Daily Percentage Changes
# Plot histogram of daily percentage changes
plt.figure(figsize=(8, 5))
plt.hist(data['DailyChange'].dropna(), bins=50, color='purple', alpha=0.7)
plt.title('Distribution of Daily Percentage Changes in Stock Prices')
plt.xlabel('Daily % Change')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Insights into Stock Price Variability
# Calculate the average daily change and standard deviation of daily changes
mean_daily_change = data['DailyChange'].mean()
std_daily_change = data['DailyChange'].std()

print(f"\nAverage Daily Percentage Change: {mean_daily_change:.2f}%")
print(f"Standard Deviation of Daily Percentage Change: {std_daily_change:.2f}%")

# Detect significant price movements (e.g., >2 standard deviations)
threshold = 2 * std_daily_change
significant_movements = data[np.abs(data['DailyChange']) > threshold]

print("\nSignificant Price Movements (more than 2 standard deviations from the mean):")
print(significant_movements[['Date', 'Close', 'DailyChange']])

# Example data
item_prices = [10, 20, 30]  # Prices of individual items
item_quantities = [2, 1, 3]  # Quantities of individual items
discount_rate = 10  # 10% discount
tax_rate = 8  # 8% tax

# Calculate subtotal
subtotal = sum(price * quantity for price, quantity in zip(item_prices, item_quantities))

# Apply discount
discount = subtotal * (discount_rate / 100)
subtotal -= discount

# Apply tax
tax = subtotal * (tax_rate / 100)

# Calculate total cost
total_cost = subtotal + tax

print("Total cost of the purchase including discounts and taxes:", total_cost)

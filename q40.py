import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset from the CSV file
data = pd.read_csv('soccer_players.csv')

# Display the first few rows of the dataset
print("Soccer Players Data:")
print(data.head())

# Step 2: Find the top 5 players with the highest number of goals scored
top_goals = data.nlargest(5, 'Goals')[['Name', 'Goals']]
print("\nTop 5 Players with Highest Goals Scored:")
print(top_goals)

# Step 3: Find the top 5 players with the highest salaries
top_salaries = data.nlargest(5, 'WeeklySalary')[['Name', 'WeeklySalary']]
print("\nTop 5 Players with Highest Salaries:")
print(top_salaries)

# Step 4: Calculate the average age of players
average_age = data['Age'].mean()
print(f"\nAverage Age of Players: {average_age:.2f} years")

# Step 5: Display the names of players who are above the average age
above_average_age = data[data['Age'] > average_age]['Name']
print("\nPlayers Above Average Age:")
print(above_average_age.to_string(index=False))

# Step 6: Visualize the distribution of players based on their positions
plt.figure(figsize=(10, 6))
position_counts = data['Position'].value_counts()
position_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Players by Position')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

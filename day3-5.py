import pandas as pd
import matplotlib.pyplot as plt

# Sample user interaction data (replace this with your actual dataset)
data = {
    'post_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'likes': [10, 15, 20, 10, 25, 15, 30, 25, 20, 15]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Calculate the frequency distribution of likes
likes_freq = df['likes'].value_counts().sort_index()

# Print the frequency distribution
print("Frequency Distribution of Likes:")
print(likes_freq)

# Plot the frequency distribution
plt.figure(figsize=(8, 6))
plt.bar(likes_freq.index, likes_freq.values, color='skyblue')
plt.title('Frequency Distribution of Likes')
plt.xlabel('Number of Likes')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

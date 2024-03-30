import numpy as np
import matplotlib.pyplot as plt

# Generate random exam scores for 50 students (replace this with your actual data)
np.random.seed(0)
exam_scores = np.random.randint(40, 101, size=50)

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(exam_scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Scores')
plt.ylabel('Frequency')
plt.grid(True)
plt.axvline(np.mean(exam_scores), color='red', linestyle='dashed', linewidth=1, label='Mean Score')
plt.legend()
plt.show()

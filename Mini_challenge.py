''''
Mini challenge — apply both topics together

Create a dataset: 50 students, each with scores in 3 subjects (make it up using numpy)
Store it as a matrix (50 × 3)
Compute: mean score per subject, std per subject
Find which student has the highest total score (sum across 3 subjects)
Plot a histogram of total scores
'''


import numpy as np
import matplotlib.pyplot as plt

# Create dataset
scores = np.random.randint(40, 100, size=(50, 3))

# Mean per subject
subject_mean = np.mean(scores, axis=0)

# Standard deviation per subject
subject_std = np.std(scores, axis=0)

# Total score of each student
total_scores = np.sum(scores, axis=1)

# Best student
top_student = np.argmax(total_scores)

# Print results
print("Scores Matrix:")
print(scores)

print("Mean per subject:", subject_mean)
print("Std per subject:", subject_std)

print("Top Student:", top_student + 1)
print("Highest Total:", total_scores[top_student])

# Plot histogram
plt.hist(total_scores, bins=10)
plt.title("Distribution of Total Scores")
plt.xlabel("Total Marks")
plt.ylabel("Number of Students")
plt.show()
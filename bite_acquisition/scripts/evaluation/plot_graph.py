import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values1 = [10, 15, 7, 12]  # Data for first group
values2 = [8, 12, 10, 14]  # Data for second group

x = np.arange(len(categories))  # X locations for the groups
width = 0.3  # Width of the bars

fig, ax = plt.subplots()

# Plot bars
bars1 = ax.bar(x - width/2, values1, width, label='Group 1', color='b')
bars2 = ax.bar(x + width/2, values2, width, label='Group 2', color='g')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Side-by-Side Bar Graph')
ax.legend()

plt.show()
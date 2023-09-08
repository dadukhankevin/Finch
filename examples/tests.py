import matplotlib.pyplot as plt
import numpy as np

# Your list of data points [(number, color_int), ...]
data = [(1, 0), (2, 1), (3, 2), (4, 3)]

# Extract the x and y values
x = [item[0] for item in data]
y = [item[1] for item in data]

# Create a colormap for changing line colors
colormap = plt.cm.viridis  # You can choose any colormap you like

# Create a solid line plot with changing colors
for i in range(len(x) - 1):
    plt.plot(x[i:i+2], y[i:i+2], color=colormap(y[i] / max(y)), marker='o', linestyle='-')

plt.xlabel('Number')
plt.ylabel('Color Intensity')
plt.title('Line Plot with Changing Colors')

plt.show()

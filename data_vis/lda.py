import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the CSV without headers
data = np.loadtxt('lda.csv', delimiter=",")

# Extract all data points from the fourth column (column index 3)
data2 = np.log(data)
data_points = data2[:, -1]
print(data_points.shape)
kde = gaussian_kde(data_points)

x_min = data_points.min() - 1  
x_max = data_points.max() + 1
x_grid = np.linspace(x_min, x_max, 1000) 

# 4. Evaluate the KDE on the grid
kde_values = kde(x_grid)

# 5. Visualization
plt.figure(figsize=(8, 6))

# Plot the KDE
plt.plot(x_grid, kde_values, color='blue', linewidth=2)
plt.title('Kernel Density Estimation of LDA (log transformation)')
plt.xlabel('Output Values')
plt.ylabel('Density')
plt.legend()

# Customize the plot (optional)
plt.grid(True, linestyle=':', alpha=0.6)  # Add a grid
plt.xlim([x_min, x_max])

# Show the plot
plt.show()
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# Load the CSV without headers
df = pd.read_csv('svm.csv', header=None)

# Extract all data points from the fourth column (column index 3)
data_points = df[3].values.reshape(-1, 1)

# Create and fit the Kernel Density Estimate
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_points)

# Generate points for evaluation
x_range = np.linspace(data_points.min(), data_points.max(), 5000).reshape(-1, 1)

# Calculate log density and convert to density
log_dens = kde.score_samples(x_range)
density = np.exp(log_dens)

# Plot the KDE
plt.plot(x_range, density)
plt.title("Kernel Density Estimate of SVM")
plt.xlabel("Values")
plt.ylabel("Density")
plt.show()
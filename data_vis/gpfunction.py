import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sns.set_theme()

def fxn(x, y):
    val1 = (1 + (x+y+1)**2 * (19-14*x+3*x**2-14*y+6*x*y+3*y**2))
    val2 = (30 + ((2*x-3*y)**2) * (18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    return val1 * val2

x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X1, X2 = np.meshgrid(x, y)
X_grid = np.vstack((X1.ravel(), X2.ravel())).T
Z = fxn(X1, X2)
Z_comp = [Z, np.log10(Z), np.sqrt(Z), np.square(Z)]
plt.figure(figsize=(10, 8))
plt.imshow(Z_comp[1], cmap='jet', aspect='auto')  # 'auto' aspect maintains the correct proportions
plt.colorbar()
plt.gca().invert_yaxis()


# Set ticks and labels (similar to seaborn's xticklabels/yticklabels)
num_ticks = 11
tick_values = np.round(np.linspace(0, 999, num_ticks, dtype=int), 1)
tick_labels = np.round(np.linspace(-2, 2, num_ticks, dtype=float), 1)

plt.xticks(tick_values, tick_labels)
plt.yticks(tick_values, tick_labels)

plt.xlabel("X")  # Add labels for clarity
plt.ylabel("Y")
plt.title("Heatmap of Goldstein-Price Function (log base 10 Transformation)") 
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
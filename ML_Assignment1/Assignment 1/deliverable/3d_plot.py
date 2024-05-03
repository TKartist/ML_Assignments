import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming X is a numpy array with long 2D arrays
# Extract x_1, x_2, and y from each 2D array within X
data = np.load("../Data/data.npz")
X = data["x"]
y = data["y"]
x_1 = X[:, 0].flatten()
x_2 = X[:, 1].flatten()


# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x_1, x_2, y, c=y, cmap="viridis", edgecolor="k")
ax.set_xlabel("x_1")
ax.set_ylabel("x_2")
ax.set_zlabel("Target Variable (y)")
ax.set_title("3D Scatter Plot of Dataset")
plt.show()

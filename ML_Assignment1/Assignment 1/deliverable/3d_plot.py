import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the model function
def model_function(theta, x):
    t0, t1, t2 = theta
    return t0 + t1 * x[:, 0] + t2 * x[:, 1]


# Generate values for x1 and x2
x1 = np.linspace(1, 10, 10)
x2 = np.linspace(1, 10, 10)
x1, x2 = np.meshgrid(x1, x2)
x = np.column_stack((x1.ravel(), x2.ravel()))

# Compute y using the model function and given theta
theta = (1, 2, 3)
y = model_function(theta, x)

# Reshape y to match the shape of x1 and x2
y = y.reshape(x1.shape)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x1, x2, y, cmap="viridis")

# Set labels and title
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
ax.set_title("3D Plot of Model Function")

plt.show()

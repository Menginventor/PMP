import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function and constraint
def f(x, y):
    return (x - 1)**2 + (y - 2)**2

# Constraint: x + 2y = 4
def g(x, y):
    return x + 2*y - 4

# Analytical solution via Lagrange multipliers
# L = (x-1)^2 + (y-2)^2 + λ(x + 2y - 4)
# ∂L/∂x: 2(x-1) + λ = 0  -> x = 1 - λ/2
# ∂L/∂y: 2(y-2) + 2λ = 0 -> y = 2 - λ
# Constraint: (1 - λ/2) + 2(2 - λ) = 4
lam = 0.4
x_star = 1 - lam/2
y_star = 2 - lam

# Grid for the surface
X, Y = np.meshgrid(np.linspace(-1, 5, 200), np.linspace(-1, 5, 200))
F = f(X, Y)

# Points along the constraint line in XY space
xs = np.linspace(-1, 5, 600)
ys = (4 - xs) / 2
fs = f(xs, ys)  # values along the curve

# Create the figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the objective surface
ax.plot_surface(X, Y, F, cmap='viridis', alpha=0.7, edgecolor='none')

# Create the constraint plane
y_line = np.linspace(-1, 5, 50)
x_line = 4 - 2*y_line
z_plane = np.linspace(0, 15, 20)  # z range for the plane
X_plane, Z_plane = np.meshgrid(x_line, z_plane)
Y_plane, _ = np.meshgrid(y_line, z_plane)
ax.plot_surface(X_plane, Y_plane, Z_plane, color='red', alpha=0.3)

# Plot the intersection curve
ax.plot(xs, ys, fs, color='r', linewidth=2, label='Intersection curve')

# Plot the optimal point
ax.scatter([x_star], [y_star], [f(x_star, y_star)],
           color='k', s=60, label='LM Solution')

# Labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Objective Surface and Constraint Plane')
ax.legend()

plt.tight_layout()
plt.show()

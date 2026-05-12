# This script solves the Omnitech problem by calling cam_creator().

import numpy as np
import matplotlib.pyplot as plt
from Cam_Creator import cam_creator 


# Define input x and y vectors
x = np.arange(0, 181, 10, dtype=float)  # 0 to 180 with 10-degree increments

y = np.array([0, 1.742222, 3.153444, 4.263166, 5.138888, 5.839111, 6.431333,
              6.960055, 7.467277, 8.013000, 8.586222, 9.261444, 10.038666,
              10.946388, 12.011611, 13.234833, 14.640055, 16.243277, 18.029000], dtype=float)

# Define the machine increment
delta = 1.0

# Define the degree of the desired polynomial
n = 6

# Call Cam Creator function
X, a, Vel, Vel_Error, Point_Error = cam_creator(x, y, delta, n)

# Print key outputs (similar to MATLAB no-semicolon behavior)
print("X (Cam matrix):")
print(X)
print("\nPolynomial coefficients a (highest power first):")
print(a)
print("\nVel [V_Start, V_End]:")
print(Vel)
print("\nVel_Error (%):")
print(Vel_Error)
print("\nPoint_Error (%):")
print(Point_Error)

# Plot the cam with the data
plt.figure(figsize=(12, 7))

# "Capstan" line from (0,0) to (180, y_end)
plt.plot([0, x[-1]], [0, y[-1]], 'b-', linewidth=4, label="Capstan")

# Cam fit curve
plt.plot(X[:, 0], X[:, 1], 'r-', linewidth=5, label="Cam")

# Data points
plt.plot(x, y, 'ko', linewidth=0, markersize=8, label="Data")

plt.title(f"Graph of Capstan and Cam-Fit with Degree {n} Polynomial", fontsize=18)
plt.xlabel(r"Degree ($x$)", fontsize=14)
plt.ylabel("Capstan and Cam Length", fontsize=14)

# Annotations 

ax = plt.gca()
ax.text(0.05, 0.95, rf"Velocity at $0^\circ$ = {Vel[0]:.6g}",
        transform=ax.transAxes, fontsize=12, va="top")
ax.text(0.05, 0.90, rf"Velocity at $180^\circ$ = {Vel[1]:.6g}",
        transform=ax.transAxes, fontsize=12, va="top")
ax.text(0.05, 0.85, rf"Rel Error of Velocities = {Vel_Error:.6g}%",
        transform=ax.transAxes, fontsize=12, va="top")
ax.text(0.05, 0.80, rf"Rel Error of Points = {Point_Error:.6g}%",
        transform=ax.transAxes, fontsize=12, va="top")

plt.legend(loc="lower right", fontsize=12)
plt.grid(True, which="major")
plt.minorticks_on()
plt.grid(True, which="minor", alpha=0.3)
plt.tight_layout()
plt.show()


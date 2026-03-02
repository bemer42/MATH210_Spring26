import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# This program solves the systems presented in Phase 2, Worksheet 1:
    
###########
# Problem 1
###########

# Create the 2x2 matrix A
A = np.array([[1, -2], 
              [1, 2]], dtype = float)

# Create the right hand side vector b
b = np.array([1, 5], dtype = float)

# Solve the system
x = la.solve(A,b)
print(x)


###########
# Problem 2
###########

# Create the 2x2 matrix A
A = np.array([[1,  1], 
              [-4, 1]], dtype = float)

# Create the right hand side vector b
b = np.array([-3, 5], dtype = float)

# Solve the system
x = np.dot(la.inv(A),b)
print(x)

# Define the slope and y-int
m = x[0];
b = x[1]; 

# Define the line equation
f = lambda x: m*x + b

# Plot the line with the two points
xplot = np.linspace(-5,5,int(1e3))
yplot = f(xplot)

fig1 = plt.figure(1)
plt.plot(xplot, yplot, 'k-', linewidth=3, label='line')
plt.plot((1,-4),(-3,5), 'ro', markersize=7, label='points')
plt.title('Interpolating Line', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('f(x) = mx + b', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, which='both')
plt.xlim([np.min(xplot)-1, np.max(xplot)+1])
plt.ylim([np.min(yplot)-1, np.max(yplot)+1])
plt.legend()

# Show the plot
plt.show()


###########
# Problem 3
###########

# Create the 3x3 matrix A
A = np.array([[1,  -3, 1], 
              [-4, 5, 1],
              [0, 3, 1]], dtype = float)
 
# Create the right hand side vector b
b = np.array([2, 1, 1], dtype = float)

# Solve the system
x = la.solve(A,b)
print(x)

# Define the coefficients for the plane
m1 = x[0];
m2 = x[1]; 
b =  x[2]; 

# Define the plane function
f = lambda x,y: m1*x + m2*y + b

# Define grid for x-y plane
xplot = np.linspace(-6,6,int(1e3))
yplot = np.linspace(-6,6,int(1e3))
X, Y = np.meshgrid(xplot,yplot)
Z = f(X,Y)

# Plot
fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax.scatter((1,-4,0),(-3,5,3), (2,1,1), 'ko', s = 30, alpha = 1.0)
ax.set_title('Interpolating Plane', fontsize=20)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('$f(x,y) = m_1x + m_2y + b$', fontsize=15)
ax.set_xlim([np.min(xplot)-1, np.max(xplot)+1])
ax.set_xlim([np.min(yplot)-1, np.max(yplot)+1])

# Show the plot:
plt.show()






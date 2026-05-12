
import numpy as np
import numpy.linalg as LA

# Objective: This script outputs the solution to Homework 2
# problem 1, which solves a linear system using the linalg module. 

# Define, manually, the matrix A: 
A = np.array([[-3, 1, 1],
             [2, -1, 4], 
             [1, 3, -11]])

# Define the right hand side vector, b: 
b = np.array([5, 1, -2])

# Solve the system: 
x = LA.solve(A, b)

# Output the solution with the error: 
print(f"Solution, x =  {x}")    

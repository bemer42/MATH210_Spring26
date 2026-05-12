import numpy as np

# This program solves Phase 3 Homework, Problem 1

# Define the matrix 
A = np.array([[5, 2], 
              [1, 2]])

# Use numpy to find eigenvalues and eigenvectors
evals, evecs = np.linalg.eig(A)
print(evals)
print(evecs)
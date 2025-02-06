import numpy as np

def gauss_seidel(A, b, max_iter=1000, tol=1e-10):
    # Number of equations (rows in matrix A)
    n = len(A)

    # Initialize the solution vector x with zeros
    x = np.zeros_like(b)

    # Iterate to solve the system
    for k in range(max_iter):
        x_old = x.copy()  # Save the previous solution

        # Update each variable in the solution
        for i in range(n):
            # Sum of A[i][j] * x[j] for all j ≠ i
            sum1 = np.dot(A[i, :i], x[:i])  # Part of the sum (before i)
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # Part of the sum (after i)
            
            # Update x[i] using the Gauss-Seidel formula
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Check if the solution has converged
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f'Converged in {k+1} iterations.')
            return x

    # If maximum iterations are reached without convergence
    print('Max iterations reached without convergence.')
    return x

# Example usage with a simple system of equations

# The system of equations:
# 4x1 - x2 + x3 = 3
# x1 + 3x2 + x3 = 5
# x1 + x2 + 4x3 = 6

A = np.array([[4, -1, 1],
              [1, 3, 1],
              [1, 1, 4]], dtype=float)

b = np.array([3, 5, 6], dtype=float)

# Initial guess for the solution (starting with zeros)
solution = gauss_seidel(A, b)

print("Solution:", solution)

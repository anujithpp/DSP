import numpy as np

def gauss_seidel(A, b, x_init=None, max_iter=1000, tol=1e-10):
    n = len(A)
    x = np.zeros_like(b) if x_init is None else x_init.copy()
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])  # Sum of A[i][j] * x[j] for j < i
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # Sum of A[i][j] * x[j] for j > i
            x[i] = (b[i] - sum1 - sum2) / A[i, i]  # Gauss-Seidel update

        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f'Converged in {k+1} iterations.')
            break
    else:
        print('Max iterations reached without convergence.')
    
    return x

# Example usage
A = np.array([[4, -1, 1],
              [2, 3, 1],
              [1, 1, 4]], dtype=float)

b = np.array([3, 5, 6], dtype=float)

x_init = np.zeros_like(b)

solution = gauss_seidel(A, b, x_init)

print("Solution:", solution)

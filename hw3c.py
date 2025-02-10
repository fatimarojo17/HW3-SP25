import numpy as np
from scipy.linalg import cholesky, lu


def is_symmetric(A):
    """Check if matrix A is symmetric (A == A^T)."""
    return np.allclose(A, A.T)


def is_positive_definite(A):
    """Check if matrix A is positive definite (all eigenvalues > 0)."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition (A = LL^T)."""
    L = cholesky(A, lower=True)  # Compute lower triangular matrix L
    y = np.linalg.solve(L, b)  # Solve Ly = b
    x = np.linalg.solve(L.T, y)  # Solve L^T x = y
    return x


def doolittle_solve(A, b):
    """Solve Ax = b using Doolittle's LU factorization."""
    P, L, U = lu(A)  # Compute LU decomposition with partial pivoting
    y = np.linalg.solve(L, np.dot(P, b))  # Solve Ly = Pb
    x = np.linalg.solve(U, y)  # Solve Ux = y
    return x


def solve_system(A, b):
    """Determine the method to use and solve the system Ax = b."""
    print("Checking if the matrix is symmetric and positive definite...")

    if is_symmetric(A) and is_positive_definite(A):
        print("\nThe matrix is symmetric and positive definite. Using Cholesky decomposition.")
        x = cholesky_solve(A, b)
    else:
        print("\nThe matrix is not symmetric and positive definite. Using Doolittle LU decomposition.")
        x = doolittle_solve(A, b)

    print("\nSolution vector (x):")
    print(x)


def main():
    """Main function to input and solve matrix equations."""

    # Define matrices and vectors for the two problems
    A1 = np.array([[1, -1, 3, 2],
                   [-1, 5, -5, -2],
                   [3, -5, 19, 3],
                   [2, -2, 3, 21]])

    b1 = np.array([15, -35, 94, 1])

    A2 = np.array([[4, 1, 4, 0],
                   [2, 2, 3, 2],
                   [4, 3, 6, 3],
                   [0, 2, 3, 9]])

    b2 = np.array([20, 36, 60, 122])

    print("\nSolving first system (A1x = b1)...")
    solve_system(A1, b1)

    print("\n" + "=" * 50 + "\n")

    print("Solving second system (A2x = b2)...")
    solve_system(A2, b2)


if __name__ == "__main__":
    main()

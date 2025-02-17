import math


def is_symmetric(A):
    """Check if matrix A is symmetric (A == A^T)."""
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            if A[i][j] != A[j][i]:
                return False
    return True


def is_positive_definite(A):
    """Check if matrix A is positive definite (all eigenvalues > 0)."""
    n = len(A)
    try:
        # Cholesky decomposition
        L = cholesky_decompose(A)
        return True
    except ValueError:
        return False


def cholesky_decompose(A):
    """Perform Cholesky decomposition (A = LL^T), returns lower triangular matrix L."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            sum_ = 0.0
            if j == i:
                for k in range(j):
                    sum_ += L[j][k] ** 2
                L[j][j] = math.sqrt(A[j][j] - sum_)
            else:
                for k in range(j):
                    sum_ += L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - sum_) / L[j][j]
    return L


def forward_substitution(L, b):
    """Solve Ly = b using forward substitution."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        sum_ = 0.0
        for j in range(i):
            sum_ += L[i][j] * y[j]
        y[i] = (b[i] - sum_) / L[i][i]
    return y


def backward_substitution(U, y):
    """Solve Ux = y using backward substitution."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ = 0.0
        for j in range(i + 1, n):
            sum_ += U[i][j] * x[j]
        x[i] = (y[i] - sum_) / U[i][i]
    return x


def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition (A = LL^T)."""
    L = cholesky_decompose(A)
    y = forward_substitution(L, b)
    x = backward_substitution([list(row) for row in zip(*L)], y)  # L^T is the transpose of L
    return x


def lu_decompose(A):
    """Perform LU decomposition using Doolittle's method (A = LU)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            sum_ = 0.0
            for k in range(i):
                sum_ += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - sum_

        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                sum_ = 0.0
                for k in range(i):
                    sum_ += L[j][k] * U[k][i]
                L[j][i] = (A[j][i] - sum_) / U[i][i]

    return L, U


def doolittle_solve(A, b):
    """Solve Ax = b using Doolittle's LU decomposition."""
    L, U = lu_decompose(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
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
    A1 = [
        [1, -1, 3, 2],
        [-1, 5, -5, -2],
        [3, -5, 19, 3],
        [2, -2, 3, 21]
    ]

    b1 = [15, -35, 94, 1]

    A2 = [
        [4, 1, 4, 0],
        [2, 2, 3, 2],
        [4, 3, 6, 3],
        [0, 2, 3, 9]
    ]

    b2 = [20, 36, 60, 122]

    print("\nSolving first system (A1x = b1)...")
    solve_system(A1, b1)

    print("\n" + "=" * 50 + "\n")

    print("Solving second system (A2x = b2)...")
    solve_system(A2, b2)


if __name__ == "__main__":
    main()

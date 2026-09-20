import numpy as np

def householder_tridiagonal(A):
    A = A.astype(float).copy()
    n = A.shape[0]

    Q = np.eye(n)

    for k in range(n - 2):
        x = A[k + 1:, k].copy()

        norm_x = np.linalg.norm(x)

        if norm_x == 0:
            continue

        sign = 1 if x[0] >= 0 else -1

        e1 = np.zeros_like(x)
        e1[0] = 1

        v = x + sign * norm_x * e1
        v = v / np.linalg.norm(v)

        H_small = np.eye(len(x)) - 2 * np.outer(v, v)

        H = np.eye(n)
        H[k + 1:, k + 1:] = H_small

        A = H @ A @ H.T
        Q = Q @ H

    A[np.abs(A) < 1e-12] = 0

    return A, Q

A = np.array([
    [4, 1, 2, 1],
    [1, 3, 0, 2],
    [2, 0, 5, 1],
    [1, 2, 1, 2]
], dtype=float)

T, Q = householder_tridiagonal(A)

print("Исходная матрица A:")
print(A)

print("\nТрёхдиагональная матрица T:")
print(T)

print("\nМатрица Q:")
print(Q)

print("\nПроверка Q.T @ A @ Q:")
print(Q.T @ A @ Q)

print("\nПроверка Q.T @ Q:")
print(Q.T @ Q)
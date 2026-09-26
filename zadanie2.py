import numpy as np
import time

def lu_decomposition(A):
    n = len(A)

    L = np.eye(n)
    U = A.astype(float).copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]

            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]

    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def lu_solve(L, U, b):
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x

n = 500

np.random.seed(1)

A = np.random.rand(n, n)
A += n * np.eye(n)

system_counts = [10, 50, 100, 200, 500]

print("Размер матрицы A:", n)
print()

start = time.perf_counter()

L, U = lu_decomposition(A)

lu_decomposition_time = time.perf_counter() - start

print("Время LU-разложения:", lu_decomposition_time)
print()

print("Количество систем | Обычное решение | Решение через LU | Выигрыш")

for m in system_counts:

    B = np.random.rand(m, n)

    start = time.perf_counter()

    X_lu = []

    for b in B:
        x = lu_solve(L, U, b)
        X_lu.append(x)

    lu_time = time.perf_counter() - start

    start = time.perf_counter()

    X_direct = []

    for b in B:
        x = np.linalg.solve(A, b)
        X_direct.append(x)

    direct_time = time.perf_counter() - start

    total_lu_time = lu_decomposition_time + lu_time

    gain = direct_time / total_lu_time

    print(
        f"{m:18d} | "
        f"{direct_time:16.6f} | "
        f"{total_lu_time:17.6f} | "
        f"{gain:8.2f} раз"
    )

b = B[0]

x = lu_solve(L, U, b)

error = np.linalg.norm(A @ x - b)

print()
print("Проверка решения:")
print("Норма ошибки ||Ax - b|| =", error)
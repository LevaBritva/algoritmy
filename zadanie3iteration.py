import numpy as np

def simple_iteration(A, b, x0, eps=1e-10, max_iter=1000):
    """
    Решение системы Ax = b
    методом простой итерации.

    A — матрица системы
    b — правая часть
    x0 — начальное приближение
    eps — требуемая точность
    max_iter — максимальное количество итераций
    """

    n = len(b)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    # Строим матрицу B и вектор c:
    #
    # x = Bx + c
    #
    B = np.zeros((n, n))
    c = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                B[i, j] = -A[i, j] / A[i, i]

        c[i] = b[i] / A[i, i]

    # Итерационный процесс
    for k in range(max_iter):

        x_new = B @ x + c

        # Проверяем условие остановки
        if np.linalg.norm(x_new - x) < eps:
            return x_new, k + 1

        x = x_new

    raise ValueError("Метод не сошёлся за заданное число итераций")

# Матрица системы

A = np.array([
    [5, 1, 1],
    [1, 6, 1],
    [1, 1, 7]
], dtype=float)

# Правая часть

b = np.array([7, 8, 9], dtype=float)

# Начальное приближение

x0 = np.zeros(len(b))

# Решение системы

x, iterations = simple_iteration(A, b, x0)

print("Решение системы:")
print(x)

print("\nКоличество итераций:")
print(iterations)

# Проверка решения

error = np.linalg.norm(A @ x - b)

print("\nНорма ошибки:")
print(error)
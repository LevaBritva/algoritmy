import numpy as np

def tridiagonal_solve(a, b, c, d):
    """
    Решение трёхдиагональной системы
    методом прогонки.

    a — нижняя диагональ
    b — главная диагональ
    c — верхняя диагональ
    d — правая часть
    """

    n = len(b)

    # Создаём копии, чтобы не изменять исходные данные
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    # Прямой ход
    for i in range(1, n):
        m = a[i] / b[i - 1]

        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Обратный ход
    x = np.zeros(n)

    x[n - 1] = d[n - 1] / b[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x

# Коэффициенты системы

a = [0, 1, 1, 1]
b = [4, 4, 4, 4]
c = [1, 1, 1, 0]
d = [6, 10, 10, 6]

# Решение системы

x = tridiagonal_solve(a, b, c, d)

print("Решение системы:")
print(x)

# Проверка решения

A = np.array([
    [4, 1, 0, 0],
    [1, 4, 1, 0],
    [0, 1, 4, 1],
    [0, 0, 1, 4]
], dtype=float)

d = np.array(d, dtype=float)

error = np.linalg.norm(A @ x - d)

print("\nНорма ошибки:")
print(error)
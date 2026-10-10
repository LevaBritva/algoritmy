def logistic_map(r, x):
    return r * x * (1 - x)

def find_superstable(n, left, right, eps=1e-14):
    """
    Находит параметр r, при котором
    после 2**n итераций x возвращается к 0.5.
    Используется метод бисекции.
    """

    def f(r):
        x = 0.5

        for _ in range(2**n):
            x = logistic_map(r, x)

        return x - 0.5

    a = left
    b = right

    for _ in range(200):
        mid = (a + b) / 2

        if abs(f(mid)) < eps or (b - a) < eps:
            return mid

        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid

    return (a + b) / 2

# Интервалы, содержащие нужные точки удвоения периода
intervals = [
    (3.2, 3.3),
    (3.49, 3.51),
    (3.55, 3.56),
    (3.566, 3.568),
    (3.569, 3.5695),
    (3.5697, 3.5699)
]

# Находим параметры для периодов 2, 4, 8, 16, 32 и 64
r_values = [2.0]

for n, (left, right) in enumerate(intervals, start=1):
    r = find_superstable(n, left, right)
    r_values.append(r)

print("Параметры удвоения периода:")

for n, r in enumerate(r_values):
    print(f"n = {n}: r = {r:.12f}")

print("\nПриближения константы Фейгенбаума:")

for n in range(2, len(r_values)):
    delta = (
        (r_values[n - 1] - r_values[n - 2])
        / (r_values[n] - r_values[n - 1])
    )

    print(f"delta_{n} = {delta:.10f}")

print("\nТабличное значение: 4.6692016091")
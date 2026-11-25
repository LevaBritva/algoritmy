
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. ИСХОДНЫЕ ДАННЫЕ ИНДИВИДУАЛЬНОГО ВАРИАНТА
# ============================================================

N = 26042004

x_points = np.array([0.0, 1.0, 2.0, 3.0])

y_points = np.array([
    1.0 + (N % 3) * 0.5,
    2.5 - (N % 4) * 0.3,
    0.5 + (N % 5) * 0.4,
    4.0 - (N % 2) * 1.0
])

print("=" * 65)
print("ИНДИВИДУАЛЬНОЕ РАСЧЕТНОЕ ЗАДАНИЕ")
print("Алгоритмы численных решений прикладных математических задач")
print("=" * 65)

print(f"\nЧисло N = {N}")
print("\nИсходные экспериментальные точки:")

for x, y in zip(x_points, y_points):
    print(f"x = {x:.0f}, f(x) = {y:.4f}")


# ============================================================
# 2. ЭТАП 1. ВОССТАНОВЛЕНИЕ ФУНКЦИИ ЧЕРЕЗ СЛАУ
# ============================================================

# P(x) = a3*x^3 + a2*x^2 + a1*x + a0

A = np.array([
    [x**3, x**2, x, 1]
    for x in x_points
], dtype=float)

b = y_points.copy()


# Метод Гаусса с выбором главного элемента

def gauss(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    n = len(b)

    for i in range(n):
        # Выбор главного элемента
        max_row = i + np.argmax(np.abs(A[i:, i]))

        if abs(A[max_row, i]) < 1e-12:
            raise ValueError("Матрица вырождена!")

        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Прямой ход
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Обратный ход
    solution = np.zeros(n)

    for i in range(n - 1, -1, -1):
        solution[i] = (
            b[i] - np.dot(A[i, i + 1:], solution[i + 1:])
        ) / A[i, i]

    return solution


det_A = np.linalg.det(A)

print("\n" + "=" * 65)
print("ЭТАП 1. РЕШЕНИЕ СИСТЕМЫ ЛИНЕЙНЫХ УРАВНЕНИЙ")
print("=" * 65)

print("\nМатрица A:")
print(A)

print("\nВектор b:")
print(b)

print(f"\nОпределитель det(A) = {det_A:.6f}")

if abs(det_A) < 1e-12:
    raise ValueError("Определитель равен нулю. Решение невозможно.")

print("Проверка: det(A) != 0 — система имеет единственное решение.")

coefficients = gauss(A, b)

a3, a2, a1, a0 = coefficients

print("\nКоэффициенты интерполяционного многочлена:")
print(f"a3 = {a3:.10f}")
print(f"a2 = {a2:.10f}")
print(f"a1 = {a1:.10f}")
print(f"a0 = {a0:.10f}")

print("\nВосстановленный интерполяционный многочлен:")
print(f"P(x) = {a3:.6f}*x^3 + ({a2:.6f})*x^2 "
      f"+ ({a1:.6f})*x + {a0:.6f}")


# Функция интерполяционного многочлена

def P(x):
    return a3*x**3 + a2*x**2 + a1*x + a0


# Проверка прохождения через исходные точки

print("\nПроверка интерполяции:")

for x, y in zip(x_points, y_points):
    print(
        f"P({x:.0f}) = {P(x):.10f}; "
        f"f({x:.0f}) = {y:.10f}"
    )


# ============================================================
# 3. ЭТАП 2. АНАЛИТИЧЕСКОЕ ИНТЕГРИРОВАНИЕ
# ============================================================

a, b_interval = 0.0, 3.0

# Первообразная:
# F(x) = a3*x^4/4 + a2*x^3/3 + a1*x^2/2 + a0*x

def F(x):
    return (
        a3*x**4 / 4
        + a2*x**3 / 3
        + a1*x**2 / 2
        + a0*x
    )


I_exact = F(b_interval) - F(a_interval := a)

print("\n" + "=" * 65)
print("ЭТАП 2. АНАЛИТИЧЕСКОЕ ИНТЕГРИРОВАНИЕ")
print("=" * 65)

print("\nПервообразная:")
print(
    f"F(x) = {a3/4:.6f}*x^4 "
    f"+ ({a2/3:.6f})*x^3 "
    f"+ ({a1/2:.6f})*x^2 "
    f"+ ({a0:.6f})*x"
)

print(f"\nТочный интеграл I_exact = {I_exact:.12f}")


# ============================================================
# 4. ЧИСЛЕННЫЕ МЕТОДЫ ИНТЕГРИРОВАНИЯ
# ============================================================

# Метод левых прямоугольников

def left_rectangles(f, a, b, n):
    h = (b - a) / n
    x = a + np.arange(n) * h
    return h * np.sum(f(x))


# Метод средних прямоугольников

def middle_rectangles(f, a, b, n):
    h = (b - a) / n
    x = a + (np.arange(n) + 0.5) * h
    return h * np.sum(f(x))


# Метод трапеций

def trapezoids(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    return h * (
        (y[0] + y[-1]) / 2
        + np.sum(y[1:-1])
    )


# Правило Симпсона 1/3

def simpson_13(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Для Симпсона 1/3 n должно быть четным.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    return h / 3 * (
        y[0] + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-1:2])
    )


# Правило Симпсона 3/8

def simpson_38(f, a, b, n):
    if n % 3 != 0:
        raise ValueError("Для Симпсона 3/8 n должно делиться на 3.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    total = y[0] + y[-1]

    for i in range(1, n):
        if i % 3 == 0:
            total += 2 * y[i]
        else:
            total += 3 * y[i]

    return 3 * h / 8 * total


methods = {
    "Левые прямоугольники": left_rectangles,
    "Средние прямоугольники": middle_rectangles,
    "Трапеции": trapezoids,
    "Симпсон 1/3": simpson_13,
    "Симпсон 3/8": simpson_38
}


# Расчеты для n = 6 и n = 12

print("\n" + "=" * 65)
print("ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ: n = 6 И n = 12")
print("=" * 65)

results = {}

for n in [6, 12]:
    print(f"\nКоличество разбиений n = {n}")
    print("-" * 65)
    print(f"{'Метод':<27} {'Интеграл':>17} {'Погрешность':>17}")

    for name, method in methods.items():
        value = method(P, a, b_interval, n)
        error = abs(I_exact - value)

        results[(name, n)] = value

        print(f"{name:<27} {value:>17.10f} {error:>17.3e}")


# ============================================================
# 5. ЭТАП 3. АНАЛИЗ СХОДИМОСТИ
# ============================================================

print("\n" + "=" * 65)
print("ЭТАП 3. АНАЛИЗ СХОДИМОСТИ")
print("=" * 65)

N_values = [6, 12, 24, 48, 96]

# Теоретические порядки методов

theoretical_orders = {
    "Левые прямоугольники": 1,
    "Средние прямоугольники": 2,
    "Трапеции": 2,
    "Симпсон 1/3": 4,
    "Симпсон 3/8": 4
}

errors = {name: [] for name in methods}
steps = []

for n in N_values:
    h = (b_interval - a) / n
    steps.append(h)

    for name, method in methods.items():
        value = method(P, a, b_interval, n)
        error = abs(I_exact - value)
        errors[name].append(error)

print("\nТаблица абсолютных погрешностей:")
print(f"{'n':<8}", end="")

for name in methods:
    print(f"{name:>23}", end="")

print()

for i, n in enumerate(N_values):
    print(f"{n:<8}", end="")

    for name in methods:
        print(f"{errors[name][i]:>23.6e}", end="")

    print()


# Аппроксимация методом наименьших квадратов

print("\nОценка фактического порядка сходимости:")
print("-" * 65)

# Порог для определения влияния машинной точности
roundoff_limit = 100 * np.finfo(float).eps * max(1.0, abs(I_exact))

estimated_orders = {}

for name in methods:
    h_array = np.array(steps)
    error_array = np.array(errors[name])

    # Исключаем нулевые ошибки и значения на уровне
    # машинной точности
    mask = error_array > roundoff_limit

    if np.count_nonzero(mask) >= 2:
        slope, intercept = np.polyfit(
            np.log(h_array[mask]),
            np.log(error_array[mask]),
            1
        )

        estimated_orders[name] = slope

        print(
            f"{name:<27} "
            f"Теоретический порядок: {theoretical_orders[name]} "
            f"| Фактический: {slope:.4f}"
        )
    else:
        estimated_orders[name] = None

        print(
            f"{name:<27} "
            f"Теоретический порядок: {theoretical_orders[name]} "
            f"| Ошибка на уровне машинной точности."
        )


# ============================================================
# 6. ГРАФИК 1. ГЕОМЕТРИЯ ИНТЕГРАЛА
# ============================================================

print("\nПостроение графика геометрии интеграла...")

x_plot = np.linspace(a, b_interval, 500)
y_plot = P(x_plot)

plt.figure(figsize=(10, 6))

plt.plot(
    x_plot,
    y_plot,
    linewidth=2.5,
    label="Интерполяционный многочлен P(x)"
)

plt.scatter(
    x_points,
    y_points,
    color="red",
    s=100,
    zorder=5,
    label="Исходные экспериментальные точки"
)

plt.fill_between(
    x_plot,
    y_plot,
    0,
    alpha=0.25,
    label=f"Площадь интеграла I = {I_exact:.4f}"
)

plt.axhline(0, color="black", linewidth=0.8)

plt.xlabel("x", fontsize=12)
plt.ylabel("P(x)", fontsize=12)

plt.title(
    "Геометрия определенного интеграла",
    fontsize=14
)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("graph_integral.png", dpi=300)
plt.show()


# ============================================================
# 7. ГРАФИК 2. СХОДИМОСТЬ В ЛОГАРИФМИЧЕСКОМ МАСШТАБЕ
# ============================================================

print("Построение графика сходимости...")

plt.figure(figsize=(11, 7))

h_array = np.array(steps)

for name in methods:
    error_array = np.array(errors[name])

    # Нулевые ошибки нельзя отображать в логарифмической шкале
    valid = error_array > 0

    if np.any(valid):
        plt.loglog(
            h_array[valid],
            error_array[valid],
            marker="o",
            linewidth=1.8,
            label=name
        )

    # Строим линию МНК только там, где ошибка выше
    # уровня машинной точности
    mask = error_array > roundoff_limit

    if np.count_nonzero(mask) >= 2:
        slope, intercept = np.polyfit(
            np.log(h_array[mask]),
            np.log(error_array[mask]),
            1
        )

        h_fit = np.linspace(
            np.min(h_array[mask]),
            np.max(h_array[mask]),
            100
        )

        error_fit = np.exp(intercept) * h_fit**slope

        plt.loglog(
            h_fit,
            error_fit,
            linestyle="--",
            alpha=0.8,
            label=f"МНК {name}: p={slope:.2f}"
        )

plt.xlabel("Шаг сетки h", fontsize=12)
plt.ylabel("Абсолютная погрешность ε", fontsize=12)

plt.title(
    "Сходимость методов численного интегрирования",
    fontsize=14
)

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig("graph_convergence.png", dpi=300)
plt.show()


# ============================================================
# 8. ИТОГОВЫЕ РЕЗУЛЬТАТЫ
# ============================================================

print("\n" + "=" * 65)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ РАСЧЕТА")
print("=" * 65)

print(f"\nИндивидуальный вариант N = {N}")

print(
    f"\nИнтерполяционный многочлен:\n"
    f"P(x) = {a3:.6f}x^3 "
    f"+ ({a2:.6f})x^2 "
    f"+ ({a1:.6f})x "
    f"+ ({a0:.6f})"
)

print(f"\nОпределитель матрицы: {det_A:.6f}")
print(f"Точный интеграл: {I_exact:.12f}")

print("\nРезультаты для n = 6:")

for name in methods:
    print(f"{name}: {results[(name, 6)]:.12f}")

print("\nРезультаты для n = 12:")

for name in methods:
    print(f"{name}: {results[(name, 12)]:.12f}")

print("\nГрафики сохранены:")
print("1. graph_integral.png")
print("2. graph_convergence.png")

print("\nРАСЧЕТ ЗАВЕРШЕН УСПЕШНО!")
print("=" * 65)

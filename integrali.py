import math

# Определение исследуемых функций

def sine_function(x):
    return math.sin(x)

def absolute_function(x):
    return abs(x - 1)

def root_function(x):
    return math.sqrt(x)

# Численное интегрирование методом трапеций

def trapezoid(func, left, right, steps):
    h = (right - left) / steps
    result = (func(left) + func(right)) / 2

    for i in range(1, steps):
        result += func(left + i * h)

    return result * h

# Численное интегрирование методом Симпсона

def simpson(func, left, right, steps):
    if steps % 2 != 0:
        raise ValueError("Количество разбиений должно быть чётным")

    h = (right - left) / steps
    result = func(left) + func(right)

    for i in range(1, steps):
        point = left + i * h

        if i % 2 == 0:
            result += 2 * func(point)
        else:
            result += 4 * func(point)

    return result * h / 3

# Вычисление порядка точности по правилу Рунге

def runge_order(first, second, third):
    diff1 = second - first
    diff2 = third - second

    if abs(diff2) < 1e-15 or abs(diff1) < 1e-15:
        return float("nan")

    return math.log2(abs(diff1 / diff2))

# Обработка одной функции

def analyze_function(title, func, a, b, start_steps):
    step_counts = [
        start_steps,
        start_steps * 2,
        start_steps * 4
    ]

    trapezoid_results = []
    simpson_results = []

    for steps in step_counts:
        trapezoid_results.append(
            trapezoid(func, a, b, steps)
        )

        simpson_results.append(
            simpson(func, a, b, steps)
        )

    order_trapezoid = runge_order(*trapezoid_results)
    order_simpson = runge_order(*simpson_results)

    print("\n" + "=" * 76)
    print(f"Исследуемая функция: {title}")
    print(f"Интервал интегрирования: [{a}, {b}]")
    print("=" * 76)

    print(
        f"{'Число шагов':<18}"
        f"{'Метод трапеций':<25}"
        f"{'Метод Симпсона':<25}"
    )

    print("-" * 76)

    for i, steps in enumerate(step_counts):
        print(
            f"{steps:<18}"
            f"{trapezoid_results[i]:<25.10f}"
            f"{simpson_results[i]:<25.10f}"
        )

    print("-" * 76)

    print(f"Порядок Рунге (трапеции): {order_trapezoid:.6f}")
    print(f"Порядок Рунге (Симпсон): {order_simpson:.6f}")
    print("=" * 76)

# Запуск вычислений

initial_steps = 10

functions = [
    ("sin(x)", sine_function, 0, math.pi),
    ("|x - 1|", absolute_function, 0, 2),
    ("sqrt(x)", root_function, 0, 1)
]

for name, function, left, right in functions:
    analyze_function(
        name,
        function,
        left,
        right,
        initial_steps
    )
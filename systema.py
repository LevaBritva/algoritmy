# ============================================================
# Часть 1. Аналитическое определение коэффициентов
# ============================================================

import sympy as sp

# Символьные величины: пределы интегрирования и искомые коэффициенты
a, b = sp.symbols('a b')
A, B, C = sp.symbols('A B C')

# Вспомогательная величина — длина промежутка
h_sym = b - a

# Уравнения, получаемые из условий точности на многочленах
equation_1 = sp.Eq(h_sym, A + B + C)

equation_2 = sp.Eq(
    h_sym**2 / 2,
    B * h_sym / 2 + C * h_sym
)

equation_3 = sp.Eq(
    h_sym**3 / 3,
    B * (h_sym / 2)**2 + C * h_sym**2
)

# Находим неизвестные коэффициенты
coeff_solution = sp.solve(
    (equation_1, equation_2, equation_3),
    (A, B, C)
)

# Печатаем результат
print("Обозначение: h = b - a, причём h != 0")
for sym in (A, B, C):
    print(f"{sym} = {sp.factor(coeff_solution[sym])}")

# Контрольная подстановка найденных значений обратно в уравнения
check_list = [
    sp.simplify((eq.lhs - eq.rhs).subs(coeff_solution))
    for eq in (equation_1, equation_2, equation_3)
]
print("Контроль (лев. часть минус прав. часть):", check_list)


# ============================================================
# Часть 2. Численное интегрирование функции sin(x)/x
# ============================================================

import numpy as np
from scipy.integrate import quad

# Подынтегральная функция
def f(x):
    return np.sin(x) / x

# Пределы интегрирования и число разбиений
a = 1
b = 2
segments = 6

# Длина одного элементарного отрезка
h = (b - a) / segments


# --- Квадратурные формулы на одном элементарном отрезке ---
def composite_rule(weights):
    """Обобщённое суммирование по всем отрезкам с заданными весами."""
    total = 0.0
    for idx in range(segments):
        start = a + idx * h
        mid = start + h / 2
        end = start + h

        if weights == 1:      # один узел — середина отрезка
            total += h * f(mid)
        elif weights == 2:    # два узла — трапеция
            total += h / 2 * (f(start) + f(end))
        elif weights == 3:    # три узла — формула Симпсона
            total += h / 6 * (
                f(start) + 4 * f(mid) + f(end)
            )
    return total


# --- Классический метод трапеций ---
def trapezoid_rule():
    acc = (f(a) + f(b)) / 2
    for j in range(1, segments):
        acc += f(a + j * h)
    return acc * h


# --- Классический метод Симпсона ---
def simpson_rule():
    acc = f(a) + f(b)
    for j in range(1, segments):
        node = a + j * h
        acc += (2 if j % 2 == 0 else 4) * f(node)
    return acc * h / 3


# --- Правило 3/8 ---
def rule_three_eighths():
    acc = f(a) + f(b)
    for j in range(1, segments):
        node = a + j * h
        acc += (2 if j % 3 == 0 else 3) * f(node)
    return acc * 3 * h / 8


# Эталонное значение интеграла
reference_value, _ = quad(f, a, b)

# Сводная таблица результатов
computed = {
    "Формула прямоугольников (1 узел)": composite_rule(1),
    "Формула прямоугольников (2 узла)": composite_rule(2),
    "Формула прямоугольников (3 узел)": composite_rule(3),
    "Метод трапеций": trapezoid_rule(),
    "Метод Симпсона": simpson_rule(),
    "Правило 3/8": rule_three_eighths(),
}

print("Эталон (SciPy quad):", round(reference_value, 10))

for label, approx in computed.items():
    deviation = abs(reference_value - approx)
    print(f"{label}: {approx:.10f}, отклонение = {deviation:.10f}")

# Определяем наиболее точный метод
most_accurate = min(computed, key=lambda lbl: abs(reference_value - computed[lbl]))
print("\nНаиболее точный метод:", most_accurate)
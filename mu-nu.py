import numpy as np
import matplotlib.pyplot as plt

def phi(x, mu, nu):
    return mu * np.sqrt(x) * (1 - nu * np.sqrt(x))

def calculate_iterations(mu, nu, x_start, tolerance=1e-6, limit=100):
    x = x_start
    sequence = [x]

    if x < 0:
        return x, 0, "Недопустимое начальное значение", sequence

    for step in range(1, limit + 1):
        next_x = phi(x, mu, nu)
        sequence.append(next_x)

        if next_x < 0:
            return next_x, step, "Выход из области определения", sequence

        if abs(next_x - x) <= tolerance:
            return next_x, step, "Сходимость достигнута", sequence

        x = next_x

    return x, limit, "Достигнут предел итераций", sequence

parameters = [
    (1, 1, 0.1),
    (1, 2, 0.1),
    (2, 0.5, 0.1),
    (1, 3.5, 0.01),
    (1, 4, 0.1)
]

for mu, nu, x0 in parameters:
    answer, count, status, sequence = calculate_iterations(mu, nu, x0)

    print(f"\nПараметр mu: {mu}")
    print(f"Параметр nu: {nu}")
    print(f"Начальное приближение: {x0}")
    print(f"Последнее значение: {answer:.8f}")
    print(f"Количество шагов: {count}")
    print(f"Итог: {status}")

    grid = np.linspace(0, 2, 400)
    function_values = phi(grid, mu, nu)

    # График функции и линии неподвижных точек
    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, function_values, label=r'$\varphi(x)$')
    plt.plot(grid, grid, '--', label=r'$y=x$')
    plt.title(f'Функция при mu={mu}, nu={nu}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Изменение приближений на каждом шаге
    plt.figure(figsize=(8, 4.5))
    plt.plot(
        np.arange(len(sequence)),
        sequence,
        marker='.',
        markersize=5
    )
    plt.title(f'Итерации при mu={mu}, nu={nu}')
    plt.xlabel('Номер шага')
    plt.ylabel(r'$x_k$')
    plt.grid()
    plt.tight_layout()
    plt.show()
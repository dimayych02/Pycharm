import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# Функции, для которых мы ищем корни
def f1(x, y):
    return 2 * x**2 - x*y - y**2 + 2*x - 2*y + 6
def f2(x, y):
    return y - x - 1
# Производные функций
def df1_dx(x, y):
    return 4 * x - y + 2
def df1_dy(x, y):
    return -x - 2 * y - 2
def df2_dx(x, y):
    return -1
def df2_dy(x, y):
    return 1
# Метод Ньютона для нахождения корней системы уравнений
def newton_method(start_x, start_y, max_iterations=100, epsilon=1e-6):
    x = start_x
    y = start_y
    for i in range(max_iterations):
        det = df1_dx(x, y) * df2_dy(x, y) - df2_dx(x, y) * df1_dy(x, y)
        delta_x = (-f1(x, y) * df2_dy(x, y) + f2(x, y) * df1_dy(x, y)) / det
        delta_y = (-f2(x, y) * df1_dx(x, y) + f1(x, y) * df2_dx(x, y)) / det
        x += delta_x
        y += delta_y
        if abs(delta_x) < epsilon and abs(delta_y) < epsilon:
            return x, y
    return None, None
# Визуализация результатов
def plot_results(x_values, y_values, xlim, ylim):
    plt.figure()
    plt.contourf(x_values, y_values, f1(x_values, y_values), levels=[-10, 0, 10], colors=['#FFAAAA', '#AAFFAA'])
    plt.colorbar()
    plt.contour(x_values, y_values, f2(x_values, y_values), levels=[0], colors='blue')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Newton Method')
    plt.show()
# Определение области для поиска корней
x = np.linspace(-2, 3, 100)
y = np.linspace(-2, 3, 100)
X, Y = np.meshgrid(x, y)
# Вызов метода Ньютона для каждой точки в области
roots = []
for i in range(len(X)):
    for j in range(len(Y)):
        root = newton_method(X[i][j], Y[i][j])
        if root not in roots and root != (None, None):
            roots.append(root)
# Визуализация результатов
if len(roots) > 0:
    roots = np.array(roots)
    plot_results(X, Y, [-2, 3], [-2, 3])
    print("Roots:", roots)
else:
    print("No roots found in the specified region")

    if __name__ == '__main__':
        plot_results(x_values, y_values, xlim, ylim)
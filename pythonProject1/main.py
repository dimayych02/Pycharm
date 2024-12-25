import numpy as np
import sympy as sp
from scipy import linalg
from sympy import exp as sym_exp
from sympy import sin, integrate, solve, cos, Function, dsolve
from sympy import symbols, Eq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import Subplot

def integralFunction():
    x = symbols('x')
    integrand = x * sin(2 * x)
    result = integrate(integrand, x)
    print("Значение интеграла:", result)


def complexIntegral():
    x, y, z = symbols('x y z')
    f = x ** 2 + y ** 2 + z ** 2
    result = integrate(f, (x, 1, 3), (y, 2, 4), (z, 5, 6))
    print(result)


def exponentGraphic():
    x = np.linspace(-10, 10, 100)
    x_sym = symbols('x')  # Создание символьной переменной x
    y_sym = sym_exp(x_sym)  # Вычисление экспоненты для символьной переменной x
    plt.plot(x, [y_sym.subs(x_sym, val) for val in x])  # Построение графика
    plt.xlabel('x')
    plt.ylabel('exp(x)')
    plt.title('График функции e(x)')
    plt.grid(True)
    plt.show()


def cosinusGraphic():
    x = symbols('x')
    # Вычисляем значения функции косинуса в заданном диапазоне
    x_values = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
    y_values = [cos(x_val) for x_val in x_values]
    # Задаем значения для меток по оси x
    x_ticks = [-3 * np.pi, -2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi]
    x_ticklabels = ['-3π', '-2π', '-π', '0', 'π', '2π', '3π']
    # Построение графика
    plt.plot(x_values, y_values)
    # Здадаем границы по осям
    plt.xlim(-4 * np.pi, 4 * np.pi)
    plt.ylim(-1, 1)
    plt.xticks(x_ticks, x_ticklabels)
    plt.xlabel('x')
    plt.ylabel('cos(x)')
    plt.title('График функции cos(x)')
    # Задаем сетку/нет
    plt.grid(True, linewidth=3)
    plt.show()


def ComplexGraphic(x):
    return np.cos(x ** 2 - 4 * x + 1)


x = np.linspace(-10, 10)  # Задаем диапазон значений x
y = ComplexGraphic(x)  # Вычисляем значения функции для каждого x
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('График функции y=cos(x^2-4x+1)')
plt.show()


def determinationFunction():
    A = [[1, 5, 3],
         [4, 5, 6],
         [7, 8, 9]]
    det = linalg.det(A)
    print("Определитель матрицы A: ", det)


def equationFunction():
    x, y, z = symbols('x y z')
    eq1 = Eq(2 * x + y + z, 1)
    eq2 = Eq(x - y + 2 * z, 3)
    eq3 = Eq(3 * x + 2 * y - z, 2)
    solution = solve((eq1, eq2, eq3), (x, y, z))
    print("Решение системы уравнений:")
    print("x =", solution[x])
    print("y =", solution[y])
    print("z =", solution[z])


def diffCalculus():
    t = sp.symbols('t')
    y = sp.Function('y')

    # Определяем уравнение
    # y' =-2y(t)
    equation = sp.Eq(y(t).diff(t), -2 * y(t))

    # Решаем уравнение
    solution = sp.dsolve(equation)

    print(solution)


# Определение функции, описывающей дифференциальное уравнение
def diff_eq():
    x = symbols('x')
    y = Function('y')(x)
    # Определение дифференциального уравнения
    diff_eq = Eq(y.diff(x, x) - 2 * y.diff(x) + y, 0)
    # Решение уравнения
    solution = dsolve(diff_eq)
    print(solution)


def DimensionThree(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
# Создание сетки значений x и y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
# Вычисление значений функции на сетке
Z = DimensionThree(X, Y)
# Создание 3D графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Построение поверхности
ax.plot_surface(X, Y, Z, color='k', rstride=5)
# Настройка осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Показать график
plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DimensionThree(x, y)

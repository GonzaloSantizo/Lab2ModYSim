import numpy as np
import matplotlib.pyplot as plt

def euler(f, y0, x0, h, x_end):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(1, len(x)):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])

    return x, y

def heun(f, y0, x0, h, x_end):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(1, len(x)):
        k1 = f(x[i - 1], y[i - 1])
        k2 = f(x[i], y[i - 1] + h * k1)
        y[i] = y[i - 1] + h * (k1 + k2) / 2

    return x, y

def taylor2(f, dfdy, y0, x0, h, x_end):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(1, len(x)):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1]) + (h**2 / 2) * dfdy(x[i - 1], y[i - 1])

    return x, y

def f1(x, y):
    return 22 * np.exp(x / 5) - 5 * x - 25

def dfdy1(x, y):
    return 22 / 5 * np.exp(x / 5) - 5

def y_exact1(x):
    return 110 * np.exp(x / 5) - (5 / 2) * x**2 - 25 * x - 3

def main():
    y0 = 82
    x0 = 0
    h = 0.1
    x_end = 2

    x_euler, y_euler = euler(f1, y0, x0, h, x_end)
    x_heun, y_heun = heun(f1, y0, x0, h, x_end)
    x_taylor2, y_taylor2 = taylor2(f1, dfdy1, y0, x0, h, x_end)
    x_exact = np.arange(x0, x_end + h, h)
    y_exact = y_exact1(x_exact)

    plt.plot(x_euler, y_euler, label='Euler Method')
    plt.plot(x_heun, y_heun, label='Heun Method')
    plt.plot(x_taylor2, y_taylor2, label='Taylor 2nd Order Method')
    plt.plot(x_exact, y_exact, label='Exact Solution', linestyle='dashed')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Numerical Methods for Solving ODEs')
    plt.show()

if __name__ == "__main__":
    main()
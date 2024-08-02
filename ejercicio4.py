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

def f2(t, y):
    return -np.sin(t)

def dfdy2(t, y):
    return -np.cos(t)

def y_exact2(t):
    return np.cos(t) + 1

def main():
    # First differential equation
    y0_1 = 82
    x0_1 = 0
    h_1 = 0.1
    x_end_1 = 2

    x_euler_1, y_euler_1 = euler(f1, y0_1, x0_1, h_1, x_end_1)
    x_heun_1, y_heun_1 = heun(f1, y0_1, x0_1, h_1, x_end_1)
    x_taylor2_1, y_taylor2_1 = taylor2(f1, dfdy1, y0_1, x0_1, h_1, x_end_1)
    x_exact_1 = np.arange(x0_1, x_end_1 + h_1, h_1)
    y_exact_1 = y_exact1(x_exact_1)

    # Second differential equation
    y0_2 = 2
    x0_2 = 0
    h_2 = 0.1
    x_end_2 = 2 * np.pi

    x_euler_2, y_euler_2 = euler(f2, y0_2, x0_2, h_2, x_end_2)
    x_heun_2, y_heun_2 = heun(f2, y0_2, x0_2, h_2, x_end_2)
    x_taylor2_2, y_taylor2_2 = taylor2(f2, dfdy2, y0_2, x0_2, h_2, x_end_2)
    x_exact_2 = np.arange(x0_2, x_end_2 + h_2, h_2)
    y_exact_2 = y_exact2(x_exact_2)

    # Plotting the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_euler_1, y_euler_1, label='Euler Method')
    plt.plot(x_heun_1, y_heun_1, label='Heun Method')
    plt.plot(x_taylor2_1, y_taylor2_1, label='Taylor 2nd Order Method')
    plt.plot(x_exact_1, y_exact_1, label='Exact Solution', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Funcion 1 (a)')

    # Vector field for the first differential equation
    X1, Y1 = np.meshgrid(np.linspace(x0_1, x_end_1, 20), np.linspace(min(y_exact_1), max(y_exact_1), 20))
    U1 = np.ones_like(X1)
    V1 = f1(X1, Y1)
    plt.quiver(X1, Y1, U1, V1, color='gray', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(x_euler_2, y_euler_2, label='Euler Method')
    plt.plot(x_heun_2, y_heun_2, label='Heun Method')
    plt.plot(x_taylor2_2, y_taylor2_2, label='Taylor 2nd Order Method')
    plt.plot(x_exact_2, y_exact_2, label='Exact Solution', linestyle='dashed')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('Funcion 2 (b)')

    # Vector field for the second differential equation
    X2, Y2 = np.meshgrid(np.linspace(x0_2, x_end_2, 20), np.linspace(min(y_exact_2), max(y_exact_2), 20))
    U2 = np.ones_like(X2)
    V2 = f2(X2, Y2)
    plt.quiver(X2, Y2, U2, V2, color='gray', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
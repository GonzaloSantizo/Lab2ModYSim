import numpy as np

def newton_multidimensional(F, x0, maxIter=100, tol=1e-7):
    """
    Encuentra los ceros de una función F: R^n -> R^n usando el método de Newton.

    Args:
        F: Función vectorial a encontrar sus ceros.
        x0: Vector inicial de búsqueda.
        maxIter: Número máximo de iteraciones (opcional).
        tol: Tolerancia para la convergencia (opcional).

    Returns:
        Lista de aproximaciones realizadas.
        Punto x* donde se encuentra el cero.
    """

    x = np.array(x0, dtype=float)
    iteraciones = [x.copy()]  # Almacenar iteraciones

    for _ in range(maxIter):
        # Calcular el Jacobiano de F en x
        J = np.zeros((len(x), len(x)))
        h = 1e-8  # Pequeño incremento para la derivada numérica
        for i in range(len(x)):
            for j in range(len(x)):
                x_h = x.copy()
                x_h[j] += h
                J[i, j] = (F(x_h)[i] - F(x)[i]) / h

        # Resolver el sistema lineal J * delta = -F(x)
        delta = np.linalg.solve(J, -F(x))
        x += delta

        iteraciones.append(x.copy())

        # Verificar convergencia
        if np.linalg.norm(delta) < tol:
            break

    return iteraciones, x

# Definición del sistema de ecuaciones
def F(x):
    return np.array([
        3*x[0] - np.cos(x[1]*x[2]) - 1/2,
        x[0]**2 - 81*(x[1] + 0.1)**2 + np.sin(x[2]) + 1.06,
        np.exp(-x[0]*x[1]) + 20*x[2] + (10*np.pi - 3)/3
    ])

# Punto inicial y ejecución del método de Newton
x0 = np.array([0.1, 0.1, 0.1])
iteraciones, x_sol = newton_multidimensional(F, x0)

# Imprimir resultados
for i, x in enumerate(iteraciones):
    print(f"Iteración {i+1}: x = {x}")

print(f"\nSolución encontrada: x* = {x_sol}")
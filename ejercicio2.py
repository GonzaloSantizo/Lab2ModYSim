import numpy as np

def newton_multidimensional(F, x0, maxIter=100, tol=1e-7):
    x = np.array(x0, dtype=float)
    iteraciones = [x.copy()]

    for _ in range(maxIter):
        J = np.zeros((len(x), len(x)))
        h = 1e-8
        for i in range(len(x)):
            for j in range(len(x)):
                x_h = x.copy()
                x_h[j] += h
                J[i, j] = (F(x_h)[i] - F(x)[i]) / h

        delta = np.linalg.solve(J, -F(x))
        x += delta
        iteraciones.append(x.copy())

        if np.linalg.norm(delta) < tol:
            break

    return iteraciones, x

def F(x):
    return np.array([
        3*x[0]**2 - x[1]**2,
        3*x[0]*x[1]**2 - x[0]**3 - 1
    ])

x0 = np.array([0.1, 0.1])
iteraciones, x_sol = newton_multidimensional(F, x0)
print("**************Ejercicio 2**************")
print("i) Solucion para la funcion F(x)")
print("***************************************")
for i, x in enumerate(iteraciones):
    print(f"Iteración {i+1}: x = {x}")
print("La solucion encontrada nos indica los 0s de la funcion F(x)")
print(f"Solución encontrada: x* = {x_sol}")
print("\n")

def G(x):
    return np.array([
        12*x[0] - 3*x[1]**2 - 4*x[2] - 7.17,
        x[0] + 10*x[1] - x[2] - 11.54,
        x[1]**3 - 7*x[2]**3 - 7.631
    ])

x0 = np.array([0.1, 0.1, 0.1])
iteraciones, x_sol = newton_multidimensional(G, x0)
print("**************Ejercicio 2**************")
print("ii) Solucion para la funcion G(x)")
print("***************************************")
for i, x in enumerate(iteraciones):
    print(f"Iteración {i+1}: x = {x}")
print("La solucion encontrada nos indica los 0s de la funcion G(x)")
print(f"Solución encontrada: x* = {x_sol}")

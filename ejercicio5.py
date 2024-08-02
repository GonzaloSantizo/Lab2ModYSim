import numpy as np
import matplotlib.pyplot as plt

def euler(f, y0, x0, h, x_end):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(1, len(x)):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])

    return x, y

import numpy as np
import matplotlib.pyplot as plt

# Definimos la función f(x, y) = dy/dx que representa la ecuación diferencial
def modelo_poblacion(x, y):
    return 0.0225 * y - 0.0003 * y**2  # Notar que usamos 'x' solo por convención, pero no se usa realmente

# Condiciones iniciales y parámetros
y0 = 25  # Población inicial
x0 = 0   # Tiempo inicial
h = 0.5  # Tamaño de paso (puedes cambiar a 1 si quieres)
x_end = 120  # Tiempo final (10 años en meses)

# Usamos la función euler para resolver la ecuación
x, y = euler(modelo_poblacion, y0, x0, h, x_end)

# Convertimos los tiempos a años
tiempos_años = x / 12 

# Calculamos la población límite (para la gráfica)
poblacion_limite = np.full_like(tiempos_años, 75) 

# Graficamos los resultados
plt.figure(figsize=(10, 6))
plt.plot(tiempos_años, y, label='Población Estimada', marker='o', linestyle='-')
plt.plot(tiempos_años, poblacion_limite, label='Población Límite', linestyle='--')
plt.xlabel('Tiempo (años)')
plt.ylabel('Población')
plt.title('Crecimiento de la Población de Animales')
plt.legend()
plt.grid(True)
plt.show()

# Calculamos y mostramos el porcentaje de la población límite alcanzado en 5 y 10 años
porcentaje_5_años = (y[int(5 * 12 / h)] / 75) * 100
porcentaje_10_años = (y[int(10 * 12 / h)] / 75) * 100
print(f"Porcentaje de la población límite alcanzado después de 5 años: {porcentaje_5_años:.2f}%")
print(f"Porcentaje de la población límite alcanzado después de 10 años: {porcentaje_10_años:.2f}%")

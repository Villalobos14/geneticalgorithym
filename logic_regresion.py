import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calcular_mse(X, y, weights):
    """Calcula el Error Cuadrático Medio (MSE)."""
    y_pred = np.dot(X, weights)
    mse = np.mean((y - y_pred) ** 2)
    return mse

def crossover(parent1, parent2):
    """Realiza el cruce entre dos padres."""
    punto_cruce = np.random.randint(1, len(parent1) - 1)
    hijo1 = np.concatenate([parent1[:punto_cruce], parent2[punto_cruce:]])
    hijo2 = np.concatenate([parent2[:punto_cruce], parent1[punto_cruce:]])
    return hijo1, hijo2

def mutate(individual, prob_mutacion):
    """Realiza la mutación en un individuo."""
    for i in range(len(individual)):
        if np.random.rand() < prob_mutacion:
            # Ajusta el rango de mutación a tu preferencia
            individual[i] += np.random.uniform(-0.5, 0.5)
    return individual

def seleccionar_padres(poblacion, fitness):
    """Selecciona los dos mejores padres según el fitness (menor MSE)."""
    indices = np.argsort(fitness)
    return poblacion[indices[:2]]

def genetic_algorithm_regresion(ruta_csv, params):
    """
    Ejecuta el algoritmo genético para regresión lineal.
    Retorna:
      - mejor_mse
      - mejor_pesos
      - coeficientes de scikit-learn
      - mse_sklearn
    """
    # Cargar el dataset
    data = pd.read_csv(ruta_csv, sep=";", header=0)
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # Agregar columna de bias (sesgo) en la primera columna
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Parámetros del algoritmo genético
    poblacion_inicial = params["poblacion_inicial"]
    poblacion_maxima = params["poblacion_maxima"]
    num_generaciones = params["num_generaciones"]
    prob_mutacion = params["prob_mutacion_gen"]

    # Inicialización de la población (pesos aleatorios)
    poblacion = np.random.uniform(-1, 1, (poblacion_inicial, X.shape[1]))
    mejor_mse = float("inf")
    mejor_pesos = None

    # Listas para almacenar la evolución de MSE y de los mejores pesos
    historial_mejor_mse = []
    historial_mejor_pesos = []

    for generacion in range(num_generaciones):
        # Evaluar fitness (MSE de cada individuo)
        fitness = np.array([calcular_mse(X, y, individuo) for individuo in poblacion])

        # Obtener el mejor individuo de la generación
        idx_mejor = np.argmin(fitness)
        if fitness[idx_mejor] < mejor_mse:
            mejor_mse = fitness[idx_mejor]
            mejor_pesos = poblacion[idx_mejor].copy()

        # Guardar valores en el historial
        historial_mejor_mse.append(mejor_mse)
        historial_mejor_pesos.append(mejor_pesos.copy())

        # Debugging: imprimir progreso (opcional)
        print(f"Generación {generacion + 1}/{num_generaciones} - Mejor MSE: {mejor_mse:.6f}")

        # Selección de padres (los 2 mejores según menor MSE)
        padres = seleccionar_padres(poblacion, fitness)

        # Generar nueva población a partir de cruces y mutaciones
        hijos = []
        while len(hijos) < poblacion_maxima:
            hijo1, hijo2 = crossover(padres[0], padres[1])
            hijos.append(mutate(hijo1, prob_mutacion))
            hijos.append(mutate(hijo2, prob_mutacion))

        # Limitamos la nueva población a la población máxima
        poblacion = np.array(hijos[:poblacion_maxima])

    # =============================
    # FINALIZADA LA EVOLUCIÓN GA
    # =============================

    # Obtener predicciones con el mejor individuo encontrado
    y_pred_ga = np.dot(X, mejor_pesos)

    # Ajuste con scikit-learn para comparar
    reg = LinearRegression().fit(X, y)
    y_pred_sklearn = reg.predict(X)
    mse_sklearn = mean_squared_error(y, y_pred_sklearn)

    # =============================
    # GENERAR GRÁFICAS
    # =============================
    # 1) Evolución de la aptitud (MSE) por generación
    plt.figure(figsize=(7, 5))
    plt.plot(historial_mejor_mse, label='Mejor MSE (GA)')
    plt.title("Evolución del MSE a lo largo de las generaciones")
    plt.xlabel("Generaciones")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2) Evolución de las betas (pesos) por generación
    #    Cada curva representará uno de los pesos (incluido el bias)
    plt.figure(figsize=(7, 5))
    num_pesos = X.shape[1]
    generaciones = range(num_generaciones)
    for i in range(num_pesos):
        pesos_i = [p[i] for p in historial_mejor_pesos]
        plt.plot(generaciones, pesos_i, label=f"W{i}")
    plt.title("Evolución de los pesos (betas) en el GA")
    plt.xlabel("Generaciones")
    plt.ylabel("Valor del peso")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3) Gráfica y deseada vs y calculada (GA vs scikit-learn)
    plt.figure(figsize=(7, 5))
    plt.plot(y, "o", label="Datos reales (y)", markersize=4)
    plt.plot(y_pred_ga, "r-", label="Predicción GA", linewidth=1)
    plt.plot(y_pred_sklearn, "g--", label="Predicción sklearn", linewidth=1)
    plt.legend()
    plt.xlabel("Índice de datos")
    plt.ylabel("Valor de salida")
    plt.title("Comparación entre GA y scikit-learn")
    plt.grid(True)
    plt.show()

    # Retornamos los valores para que la vista los muestre
    return mejor_mse, mejor_pesos, reg.coef_, mse_sklearn

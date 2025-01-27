import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def calcular_mse(X, y, weights):
    """Calcula el Error Cuadrático Medio (MSE)"""
    y_pred = np.dot(X, weights)
    mse = np.mean((y - y_pred) ** 2)
    return mse


def crossover(parent1, parent2):
    """Realiza el cruce entre dos padres"""
    punto_cruce = np.random.randint(1, len(parent1) - 1)
    hijo1 = np.concatenate([parent1[:punto_cruce], parent2[punto_cruce:]])
    hijo2 = np.concatenate([parent2[:punto_cruce], parent1[punto_cruce:]])
    return hijo1, hijo2


def mutate(individual, prob_mutacion):
    """Realiza la mutación en un individuo"""
    for i in range(len(individual)):
        if np.random.rand() < prob_mutacion:
            individual[i] += np.random.uniform(-0.5, 0.5)
    return individual


def seleccionar_padres(poblacion, fitness):
    """Selecciona los dos mejores padres según el fitness"""
    indices = np.argsort(fitness)
    return poblacion[indices[:2]]


def genetic_algorithm_regresion(ruta_csv, params):
    """Ejecuta el algoritmo genético para regresión lineal"""
    # Cargar el dataset
    data = pd.read_csv(ruta_csv, sep=";", header=0)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Agregar columna de bias (sesgo)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Parámetros del algoritmo genético
    poblacion_inicial = params["poblacion_inicial"]
    poblacion_maxima = params["poblacion_maxima"]
    num_generaciones = params["num_generaciones"]
    prob_mutacion = params["prob_mutacion_gen"]

    # Inicialización de la población
    poblacion = np.random.uniform(-1, 1, (poblacion_inicial, X.shape[1]))
    mejor_mse = float("inf")
    mejor_pesos = None

    for generacion in range(num_generaciones):
        # Evaluar fitness (MSE)
        fitness = np.array([calcular_mse(X, y, individuo) for individuo in poblacion])

        # Obtener el mejor individuo
        idx_mejor = np.argmin(fitness)
        if fitness[idx_mejor] < mejor_mse:
            mejor_mse = fitness[idx_mejor]
            mejor_pesos = poblacion[idx_mejor]

        # Debugging: imprimir progreso
        print(f"Generación {generacion + 1}/{num_generaciones}, Mejor MSE: {mejor_mse:.4f}")

        # Selección, cruce y mutación
        padres = seleccionar_padres(poblacion, fitness)
        hijos = []
        while len(hijos) < poblacion_maxima:
            hijo1, hijo2 = crossover(padres[0], padres[1])
            hijos.append(mutate(hijo1, prob_mutacion))
            hijos.append(mutate(hijo2, prob_mutacion))
        poblacion = np.array(hijos[:poblacion_maxima])

    # Graficar resultados
    y_pred_ga = np.dot(X, mejor_pesos)
    reg = LinearRegression().fit(X, y)
    y_pred_sklearn = reg.predict(X)

    mse_sklearn = mean_squared_error(y, y_pred_sklearn)
    print("\nResultados del ajuste de regresión:")
    print(f"Pesos obtenidos por GA: {mejor_pesos}")
    print(f"Pesos obtenidos por scikit-learn: {reg.coef_}")
    print(f"MSE con algoritmo genético: {mejor_mse:.4f}")
    print(f"MSE con scikit-learn: {mse_sklearn:.4f}")

    # Gráfica
    plt.figure()
    plt.plot(y, "o", label="Datos reales (y)", markersize=4)
    plt.plot(y_pred_ga, "r-", label="Predicción GA (y_pred)", linewidth=1)
    plt.plot(y_pred_sklearn, "g--", label="Predicción sklearn (y_pred)", linewidth=1)
    plt.legend()
    plt.xlabel("Índice de datos")
    plt.ylabel("Valores")
    plt.title("Comparación entre GA y scikit-learn")
    plt.show()

    return mejor_mse, mejor_pesos, reg.coef_, mse_sklearn

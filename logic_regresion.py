import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def mse(X, y, w):
    y_pred = np.dot(X, w)
    return np.mean((y - y_pred) ** 2)

def crossover(p1, p2):
    idx = np.random.randint(1, len(p1) - 1)
    return np.concatenate([p1[:idx], p2[idx:]]), np.concatenate([p2[:idx], p1[idx:]])

def mutate(ind, prob):
    for i in range(len(ind)):
        if np.random.rand() < prob:
            ind[i] += np.random.uniform(-0.5, 0.5)
    return ind

def select_parents(pop, scores):
    return pop[np.argsort(scores)[:2]]

def genetic_regression(csv, params):
    data = pd.read_csv(csv, sep=";", header=0)
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    popsize, maxpop, gens, mutprob = params.values()
    pop = np.random.uniform(-1, 1, (popsize, X.shape[1]))
    bestmse, bestw = float("inf"), None
    historymse, historyw = [], []
    
    for gen in range(gens):
        scores = np.array([mse(X, y, ind) for ind in pop])
        idxbest = np.argmin(scores)
        
        if scores[idxbest] < bestmse:
            bestmse, bestw = scores[idxbest], pop[idxbest].copy()
        
        historymse.append(bestmse)
        historyw.append(bestw.copy())
        print(f"Gen {gen + 1}/{gens} - MSE: {bestmse:.6f}")
        
        p1, p2 = select_parents(pop, scores)
        children = []
        while len(children) < maxpop:
            c1, c2 = crossover(p1, p2)
            children.append(mutate(c1, mutprob))
            children.append(mutate(c2, mutprob))
        
        pop = np.array(children[:maxpop])
    
    ypredga = np.dot(X, bestw)
    model = LinearRegression().fit(X, y)
    ypredsklearn = model.predict(X)
    msesklearn = mean_squared_error(y, ypredsklearn)
    
    plt.figure(figsize=(7, 5))
    plt.plot(historymse, label='Mejor MSE')
    plt.title("Evolución del MSE")
    plt.xlabel("Generaciones")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(7, 5))
    for i in range(X.shape[1]):
        plt.plot(range(gens), [w[i] for w in historyw], label=f"W{i}")
    plt.title("Evolución de los Pesos")
    plt.xlabel("Generaciones")
    plt.ylabel("Valor del Peso")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(7, 5))
    plt.plot(y, "o", label="Real", markersize=4)
    plt.plot(ypredga, "r-", label="GA", linewidth=1)
    plt.plot(ypredsklearn, "g--", label="sklearn", linewidth=1)
    plt.legend()
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.title("Comparación de Predicciones")
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(7, 5))
    plt.plot(np.abs(y - ypredga), "b-", label="|Y_d - Y_c|", linewidth=1)
    plt.xlabel("Índice")
    plt.ylabel("Diferencia Absoluta")
    plt.title("Error Absoluto entre Y_d y Y_c")
    plt.legend()
    plt.grid()
    plt.show()
    return bestmse, bestw, model.coef_, msesklearn





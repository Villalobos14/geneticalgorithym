import matplotlib.pyplot as plt
import os
import numpy as np
from sympy import symbols, lambdify
i = 0

def generar_segunda_grafica(x,y,mejor_x, mejor_y, peor_x, peor_y, generacion_actual, inicial_x, final_x, mejor_individuo, peor_individuo, funcion, tipo_problema):
    global i
    individuo_x = x
    individuo_y = y
    mejor_x_value = mejor_x
    mejor_y_value = mejor_y.y
    peor_x_value = peor_x
    peor_y_value = peor_y.y
    
    limite_inicial = inicial_x
    limite_final = final_x
    
    
    
    mejor_individuo_y = max(mejor_individuo)
    peor_individuo_y = min(peor_individuo)


    funcion_value = funcion
    tipo_problema_value = tipo_problema
    
    

    i += 1
    
    plt.clf()
    #plt.title(f"Función: {funcion_value} \n Generación {generacion_actual}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    
    plt.axhline(0, color='black',linewidth=0.5)
    
    img_folder_path = 'results/second-graph/img'
    
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)

    plt.scatter(individuo_x, individuo_y, label="Individuos", s=400, c="#45aaf2", alpha=0.4)
    plt.xlim(limite_inicial, limite_final)

    if tipo_problema_value == "Minimizacion":
        x_func = np.linspace(limite_inicial, limite_final, 200)
        x = symbols('x')
        expresion = lambdify(x, funcion_value, 'numpy')
        y_func= expresion(x_func)
        plt.plot(x_func, y_func)
        plt.scatter(mejor_x_value, mejor_y_value, label="Mejor", s=300, c="#20bf6b", alpha=0.4)
        plt.scatter(peor_x_value, peor_y_value, label="Peor", s=300, c="#eb3b5a", alpha=0.4)
        plt.legend()
        plt.ylim(mejor_individuo_y - 10, peor_individuo_y + 10)
    else:
        x_func = np.linspace(limite_inicial, limite_final, 200)
        x = symbols('x')
        expresion = lambdify(x, funcion_value, 'numpy')
        y_func= expresion(x_func)
        plt.plot(x_func, y_func)
        plt.scatter(mejor_x_value, mejor_y_value, label="Mejor", s=300, c="#20bf6b", alpha=0.4)
        plt.scatter(peor_x_value, peor_y_value, label="Peor", s=300, c="#eb3b5a", alpha=0.4)
        plt.legend()
        plt.ylim(peor_individuo_y -10 , mejor_individuo_y + 10)
    
    img_file_name = f'img_generacion_{i}.png'

    img_file_path = os.path.join(img_folder_path, img_file_name)
    plt.savefig(img_file_path)
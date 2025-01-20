import matplotlib.pyplot as plt
#import os
i = 0

def generar_graficas(mejor_individuo, peor_individuo, promedio, generacion_actual, num_generaciones):
    global i
    x = generacion_actual
    media = promedio
    mejor = mejor_individuo
    peor = peor_individuo
    generacion = num_generaciones
    i += 1
    
    plt.clf()
    plt.title("Evolución del fitness")
    plt.xlabel("Generación número: "+str(i))
    plt.ylabel("Eje Y")
    plt.grid(True)
    
    #img_folder_path = 'results/first-graph/img'

    #if not os.path.exists(img_folder_path):
    #    os.makedirs(img_folder_path)

    plt.plot(x, media, label='Promedio', color='#0fb9b1')
    plt.plot(x, mejor, label='Mejor individuo', color='#f7b731')
    plt.plot(x, peor, label='Peor individuo', color='#eb3b5a')
    
    plt.legend()

    #img_file_name = f'img_generacion_{i}.png'

    #img_file_path = os.path.join(img_folder_path, img_file_name)
    #plt.savefig(img_file_path)
    
    if i == generacion:
        plt.show()

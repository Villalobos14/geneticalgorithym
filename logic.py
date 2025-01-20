import math
import random
import tkinter as tk
from sympy import symbols, lambdify
from graphs.graphic import generar_graficas
from video import generar_video
from graphs.graphic2 import generar_segunda_grafica



class Individuo:
    identificador = 0
    def __init__(self, binario, i, x, y):
        Individuo.identificador += 1
        self.id = Individuo.identificador
        self.binario = binario
        self.i = i
        self.x = round(x, 4)
        self.y = round(y, 4)
    def __str__(self):
        return f"ID: {self.id}, i: {self.i}, Binario: {self.binario}, X: {self.x}, Y: {self.y}"


class Data:
   
    rango_punto_cruza = 1
    rango = 0
    rango_numero = 0
    resolucion = 0
    resolucion_deseada = 0
    limite_inferior = 0
    limite_superior = 0
    num_bits = 1

    
    poblacion_inicial = 0
    poblacion_maxima = 0
    poblacion_general = []

   
    prob_mutacion_ind = 0
    prob_mutacion_gen = 0
    num_generaciones = 0
    generacion_actual = 0

    
    tipo_problema_value = ""
    funcion = ""

    # Estrategias seleccionadas dinámicamente 3 1 1 2
    estrategia_formacion = "A6"  
    estrategia_mutacion = "M1"  
    estrategia_cruce = "C1"      
    estrategia_poda = "P2"       


class Estadisticas:
    mejor_individuo = None
    peor_individuo = None
    promedio = None
    mejor_individuo_arreglo = []
    peor_individuo_arreglo = []
    promedio_arreglo = []
    generacion_arreglo = []


def vaciarDatos():
    Data.rango_punto_cruza = 1
    Data.rango = 0
    Data.rango_numero = 0
    Data.resolucion = 0
    Data.resolucion_deseada = 0
    Data.limite_inferior = 0
    Data.limite_superior = 0
    Data.poblacion_inicial = 0
    Data.poblacion_maxima = 0
    Data.tipo_problema_value = ""
    Data.poblacion_general = []
    Data.prob_mutacion_ind = 0
    Data.prob_mutacion_gen = 0
    Data.num_generaciones = 0
    Data.generacion_actual = 0
    Data.funcion = ""
    Data.num_bits = 1

    Estadisticas.mejor_individuo = None
    Estadisticas.peor_individuo = None
    Estadisticas.promedio = None
    Estadisticas.mejor_individuo_arreglo = []
    Estadisticas.peor_individuo_arreglo = []
    Estadisticas.promedio_arreglo = []
    Estadisticas.generacion_arreglo = []



def calcular_funcion(funcion, valor_x):
    """Evalúa la función simbólica en el valor_x."""
    x = symbols('x')
    expresion = lambdify(x, funcion, 'numpy')
    resultado = expresion(valor_x)
    return resultado

def calcular_valor_x(num_generado):
    """Convierte el número entero (binario) a un valor de x en el rango [lim_inf, lim_sup]."""
    if Data.limite_inferior >= Data.limite_superior:
        valor_x = Data.limite_superior + num_generado * Data.resolucion
        return valor_x
    valor_x = Data.limite_inferior + num_generado * Data.resolucion
    return valor_x

def calcular_datos():
    """Establece la resolución y el número de bits necesarios para representar el dominio."""
    Data.rango = Data.limite_superior - Data.limite_inferior
    num_saltos = Data.rango / Data.resolucion_deseada
    num_puntos = num_saltos + 1

    # Cálculo de bits
    Data.num_bits = math.log2(abs(num_puntos))
    if Data.num_bits % 1 != 0:
        Data.num_bits = math.ceil(Data.num_bits)
    else:
        Data.num_bits = int(Data.num_bits)

    Data.resolucion = Data.rango / (2 ** Data.num_bits)
    if Data.resolucion % 1 == 0:
        Data.resolucion = int(Data.resolucion)
    else:
        Data.resolucion = round(Data.resolucion, 4)

    Data.rango_numero = 2**Data.num_bits - 1
    Data.rango_punto_cruza = len(bin(Data.rango_numero)[2:])  # Para cruza


def generar_primer_poblacion():
    """Genera la población inicial de manera aleatoria."""
    for _ in range(Data.poblacion_inicial):
        num_generado = random.randint(1, Data.rango_numero)
        num_generado_binario = format(num_generado, f"0{Data.num_bits}b")
        valor_x = calcular_valor_x(num_generado)
        valor_y = calcular_funcion(Data.funcion, valor_x)
        individuo = Individuo(
            i=num_generado,
            binario=num_generado_binario,
            x=valor_x,
            y=valor_y
        )
        Data.poblacion_general.append(individuo)

#parejas

def formacion_A1():
    """
    Para cada individuo, generar una cantidad aleatoria m entre [0, n],
    elegir aleatoriamente los m individuos que se cruzarán con él.
    (Se puede omitir self-pareja).
    """
    # Se asume Data.n para el rango de m (puedes crear un Data.parametros o algo similar).
    # Usaremos un 'n' fijo, aquí un valor de ejemplo = 3
    n = 3  # Ajusta según necesites
    poblacion = Data.poblacion_general[:]
    parejas = []
    for ind in poblacion:
        m = random.randint(0, n)
        # Filtrar para evitar emparejarse a sí mismo
        candidatos = [c for c in poblacion if c != ind]
        if m > 0 and len(candidatos) > 0:
            seleccionados = random.sample(candidatos, min(m, len(candidatos)))
            for s in seleccionados:
                parejas.append((ind, s))
    return parejas

def formacion_A2():
    """
    Para cada individuo, se evalúa si se cruzará según un umbral Pc;
    en caso de que sí, elegir aleatoriamente la o las parejas.
    """
    # Supongamos Pc es Data.pc (probabilidad de cruzarse).
    Pc = 0.5  # Ajusta según necesidad
    poblacion = Data.poblacion_general[:]
    parejas = []
    for ind in poblacion:
        if random.random() <= Pc:
            # elige 1 o varias parejas, ejemplo: 1
            candidato = random.choice([c for c in poblacion if c != ind])
            parejas.append((ind, candidato))
    return parejas

def formacion_A3():
    """
    La población se ordena y particiona en dos; los individuos de la partición
    con mejor aptitud se cruzan con algunos o todos los individuos de la partición menos apta.
    """
    flag = (Data.tipo_problema_value != "Minimizacion")
    individuos_ordenados = sorted(Data.poblacion_general, key=lambda x: x.y, reverse=flag)
    
    mitad = int(len(individuos_ordenados) / 2)
    particion_mejor_aptitud = individuos_ordenados[:mitad]
    particion_menor_aptitud = individuos_ordenados[mitad:]
    
    parejas = []
    for mejor_individuo in particion_mejor_aptitud:
        for individuo in particion_menor_aptitud:
            parejas.append((mejor_individuo, individuo))
    return parejas

def formacion_A4():
    """
    La población se ordena y se selecciona un porcentaje de los mejores individuos;
    los individuos de la selección se cruzan con algunos o todos los individuos.
    """
    porcentaje = 0.3  # Ajusta el porcentaje que gustes
    flag = (Data.tipo_problema_value != "Minimizacion")
    individuos_ordenados = sorted(Data.poblacion_general, key=lambda x: x.y, reverse=flag)
    
    num_mejores = max(1, int(len(individuos_ordenados) * porcentaje))
    mejores = individuos_ordenados[:num_mejores]
    resto = individuos_ordenados[num_mejores:]
    
    parejas = []
    for mejor in mejores:
        for ind in resto:
            parejas.append((mejor, ind))
    return parejas

def formacion_A5():
    """
    Todos con todos.
    """
    poblacion = Data.poblacion_general[:]
    parejas = []
    for i in range(len(poblacion)):
        for j in range(i + 1, len(poblacion)):
            parejas.append((poblacion[i], poblacion[j]))
    return parejas

def formacion_A6():
    """
    Ruleta, previa transformación de los datos. Si es maximización,
    el peor fitness se transforma a 0 con una transformación lineal.
    Para minimización, multiplicar por -1 y luego transformar a 0.
    """
    # Para simplificar, haremos ruleta en un contexto de aptitud no negativa.
    # 1. Transformar fitness a valores >= 0
    # 2. Ruleta para seleccionar parejas
    poblacion = Data.poblacion_general[:]
    flag = (Data.tipo_problema_value != "Minimizacion")
    fitness_vals = [ind.y for ind in poblacion]

    # Transformación
    if flag:
        # Maximizacion: restamos el fitness del peor para que quede en 0
        min_fit = min(fitness_vals)
        offset = -min_fit if min_fit < 0 else 0
        fitness_transformada = [f + offset for f in fitness_vals]
    else:
        # Minimizacion: multiplicar por -1 y luego offset
        fitness_invertida = [-f for f in fitness_vals]
        min_fit = min(fitness_invertida)
        offset = -min_fit if min_fit < 0 else 0
        fitness_transformada = [fi + offset for fi in fitness_invertida]

    total_fit = sum(fitness_transformada)
    
    if total_fit == 0:
        # Si todo es cero, forzamos un random pairing
        return formacion_A5()  # Por ejemplo, todos con todos

    # Ruleta
    def seleccionar_por_ruleta():
        r = random.uniform(0, total_fit)
        acumulado = 0
        for i, f in enumerate(fitness_transformada):
            acumulado += f
            if acumulado >= r:
                return poblacion[i]
        return poblacion[-1]

    parejas = []
    # Ejemplo: generamos tantas parejas como la mitad del tamaño de la población
    # para no exagerar la cantidad de parejas.
    for _ in range(len(poblacion)//2):
        ind1 = seleccionar_por_ruleta()
        ind2 = seleccionar_por_ruleta()
        if ind1 != ind2:
            parejas.append((ind1, ind2))
    return parejas

estrategias_formacion = {
    "A1": formacion_A1,
    "A2": formacion_A2,
    "A3": formacion_A3,
    "A4": formacion_A4,
    "A5": formacion_A5,
    "A6": formacion_A6,
}

#cruza

def cruce_C1(ind1, ind2):
    """
    Un punto de cruza aleatorio.
    """
    punto_cruza = random.randint(1, Data.rango_punto_cruza - 1)
    parte1 = ind1.binario[:punto_cruza]
    parte2 = ind1.binario[punto_cruza:]
    parte3 = ind2.binario[:punto_cruza]
    parte4 = ind2.binario[punto_cruza:]
    
    nuevo_individuo1 = parte1 + parte4
    nuevo_individuo2 = parte3 + parte2
    return nuevo_individuo1, nuevo_individuo2

def cruce_C2(ind1, ind2):
    """
    Múltiples puntos de cruza; se elige aleatoriamente cuántos.
    Luego, para cada punto, se intercambian segmentos.
    """
    longitud = Data.rango_punto_cruza
    num_puntos = random.randint(1, longitud - 1)  # cuántos puntos de cruza
    puntos = sorted(random.sample(range(1, longitud), num_puntos))

    bin1 = list(ind1.binario)
    bin2 = list(ind2.binario)

    # Intercambio por cada punto
    for i, p in enumerate(puntos):
        if i % 2 == 0:  # Para alternar
            bin1[p:], bin2[p:] = bin2[p:], bin1[p:]
    
    return "".join(bin1), "".join(bin2)

def cruce_C3(ind1, ind2):
    """
    Un solo punto de cruza fijo (por ejemplo, a la mitad).
    """
    punto_fijo = Data.rango_punto_cruza // 2
    parte1 = ind1.binario[:punto_fijo]
    parte2 = ind1.binario[punto_fijo:]
    parte3 = ind2.binario[:punto_fijo]
    parte4 = ind2.binario[punto_fijo:]
    
    nuevo_individuo1 = parte1 + parte4
    nuevo_individuo2 = parte3 + parte2
    return nuevo_individuo1, nuevo_individuo2


estrategias_cruce = {
    "C1": cruce_C1,
    "C2": cruce_C2,
    "C3": cruce_C3,
}
#mutacion

def mutacion_M1(binario):
    """
    (100%) Negación del bit.
    Se realiza si el individuo entra a mutar (prob_mutacion_ind),
    y luego se hace bit a bit con prob_mutacion_gen.
    """
    binario_separado = list(binario)
    for i in range(len(binario_separado)):
        if random.random() <= Data.prob_mutacion_gen:
            binario_separado[i] = '1' if binario_separado[i] == '0' else '0'
    return "".join(binario_separado)

def mutacion_M2(binario):
    """
    (100%) Intercambio de posición de bits.
    Si el individuo muta, elegir aleatoriamente la posición a la que se intercambiará.
    """
    binario_separado = list(binario)
    # Escogemos una posición i y otra j para intercambiar
    i = random.randint(0, len(binario_separado)-1)
    j = random.randint(0, len(binario_separado)-1)
    binario_separado[i], binario_separado[j] = binario_separado[j], binario_separado[i]
    return "".join(binario_separado)

estrategias_mutacion = {
    "M1": mutacion_M1,
    "M2": mutacion_M2,
}

#poda

def poda_P1():
    """
    Mantener los mejores individuos (80%).
    Se asume que reduce la población al 80% o algún criterio similar.
    """
    # Quita duplicados primero
    unique_poblacion = eliminar_repetidos(Data.poblacion_general)

    # Ordenamos
    flag = (Data.tipo_problema_value != "Minimizacion")
    individuos_ordenados = sorted(unique_poblacion, key=lambda x: x.y, reverse=flag)

    # Mantener cierto porcentaje
    maximo = Data.poblacion_maxima
    corte = int(0.8 * maximo) if maximo > 0 else len(individuos_ordenados)
    Data.poblacion_general = individuos_ordenados[:corte]

def poda_P2():
    """
    Eliminación aleatoria asegurando mantener al mejor individuo de la población.
    """
    # Quita duplicados primero
    unique_poblacion = eliminar_repetidos(Data.poblacion_general)

    flag = (Data.tipo_problema_value != "Minimizacion")
    individuos_ordenados = sorted(unique_poblacion, key=lambda x: x.y, reverse=flag)

    # El mejor individuo siempre se mantiene
    if len(individuos_ordenados) == 0:
        Data.poblacion_general = []
        return

    mejor = individuos_ordenados[0]
    if len(individuos_ordenados) > 1:
        resto = individuos_ordenados[1:]
    else:
        resto = []

    maximo = Data.poblacion_maxima
    # Seleccionamos aleatoriamente el resto, pero maximo-1
    if maximo > 1:
        seleccion = random.sample(resto, min(len(resto), maximo-1))
        Data.poblacion_general = [mejor] + seleccion
    else:
        # Si la población máxima es 1, solo se mantiene el mejor
        Data.poblacion_general = [mejor]

def poda_P3():
    """
    Dados los valores del fitness, generar clases; luego por clase mantener
    algunos individuos de manera aleatoria.
    (Aquí haremos un ejemplo sencillo de "clustering" en 3 clases.)
    """
    unique_poblacion = eliminar_repetidos(Data.poblacion_general)
    # Ordenamos
    flag = (Data.tipo_problema_value != "Minimizacion")
    individuos_ordenados = sorted(unique_poblacion, key=lambda x: x.y, reverse=flag)

    # Dividimos en 3 clases de forma simple (mejores, medios, peores)
    if len(individuos_ordenados) < 3:
        Data.poblacion_general = individuos_ordenados
        return

    tercio = len(individuos_ordenados) // 3
    clase1 = individuos_ordenados[:tercio]
    clase2 = individuos_ordenados[tercio: 2*tercio]
    clase3 = individuos_ordenados[2*tercio:]

    # Mantener algunos de cada clase
    seleccion_clase1 = random.sample(clase1, max(1, len(clase1)//2)) if len(clase1) > 1 else clase1
    seleccion_clase2 = random.sample(clase2, max(1, len(clase2)//2)) if len(clase2) > 1 else clase2
    seleccion_clase3 = random.sample(clase3, max(1, len(clase3)//2)) if len(clase3) > 1 else clase3

    nueva_poblacion = seleccion_clase1 + seleccion_clase2 + seleccion_clase3

    # Ajuste a poblacion_maxima
    maximo = Data.poblacion_maxima
    if len(nueva_poblacion) > maximo:
        nueva_poblacion = random.sample(nueva_poblacion, maximo)

    Data.poblacion_general = nueva_poblacion

estrategias_poda = {
    "P1": poda_P1,
    "P2": poda_P2,
    "P3": poda_P3,
}


def eliminar_repetidos(poblacion):
    """
    Elimina individuos repetidos basándose en su i (decimal).
    Mantiene el primero que aparece.
    """
    conjunto_i = set()
    poblacion_sin_repetidos = []
    for individuo in poblacion:
        if individuo.i not in conjunto_i:
            conjunto_i.add(individuo.i)
            poblacion_sin_repetidos.append(individuo)
    return poblacion_sin_repetidos

def guardar_nuevos_individuos(binario1, binario2):
    """
    Crea dos nuevos Individuos a partir de sus representaciones binarias,
    y los agrega a la población general.
    """
    numero_decimal1 = int(binario1, 2)
    numero_decimal2 = int(binario2, 2)
    if Data.limite_inferior >= Data.limite_superior:
        x1 = Data.limite_superior + numero_decimal1 * Data.resolucion
        x2 = Data.limite_superior + numero_decimal2 * Data.resolucion
    else:
        x1 = Data.limite_inferior + numero_decimal1 * Data.resolucion
        x2 = Data.limite_inferior + numero_decimal2 * Data.resolucion
    
    y1 = calcular_funcion(Data.funcion, x1)
    y2 = calcular_funcion(Data.funcion, x2)
    
    individuo1 = Individuo(i=numero_decimal1, binario=binario1, x=x1, y=y1)
    individuo2 = Individuo(i=numero_decimal2, binario=binario2, x=x2, y=y2)
    
    Data.poblacion_general.append(individuo1)
    Data.poblacion_general.append(individuo2)


def generar_estadisticas():
    if Data.tipo_problema_value == "Minimizacion":
        mejor_individuo = min(Data.poblacion_general, key=lambda individuo: individuo.y)
        peor_individuo = max(Data.poblacion_general, key=lambda individuo: individuo.y)
    else:
        mejor_individuo = max(Data.poblacion_general, key=lambda individuo: individuo.y)
        peor_individuo = min(Data.poblacion_general, key=lambda individuo: individuo.y)
    
    promedio = sum(individuo.y for individuo in Data.poblacion_general) / len(Data.poblacion_general)
    
    Estadisticas.mejor_individuo = mejor_individuo
    Estadisticas.peor_individuo = peor_individuo
    Estadisticas.promedio = promedio

    Estadisticas.mejor_individuo_arreglo.append(Estadisticas.mejor_individuo.y)
    Estadisticas.peor_individuo_arreglo.append(Estadisticas.peor_individuo.y)
    Estadisticas.promedio_arreglo.append(Estadisticas.promedio)
    Estadisticas.generacion_arreglo.append(Data.generacion_actual)

    
    valores_x = [individuo.x for individuo in Data.poblacion_general]
    valores_y = [individuo.y for individuo in Data.poblacion_general]
    
    if Data.tipo_problema_value == "Minimizacion":
        mejor_y = min(Data.poblacion_general, key=lambda individuo: individuo.y)
        mejor_x = mejor_y.x
        peor_y = max(Data.poblacion_general, key=lambda individuo: individuo.y)
        peor_x = peor_y.x
    else:
        mejor_y = max(Data.poblacion_general, key=lambda individuo: individuo.y)
        mejor_x = mejor_y.x
        peor_y = min(Data.poblacion_general, key=lambda individuo: individuo.y)
        peor_x = peor_y.x


    generar_segunda_grafica(
        valores_x, valores_y,
        mejor_x, mejor_y,
        peor_x, peor_y,
        Data.generacion_actual,
        Data.limite_inferior,
        Data.limite_superior,
        Estadisticas.mejor_individuo_arreglo,
        Estadisticas.peor_individuo_arreglo,
        Data.funcion,
        Data.tipo_problema_value
    )


    generar_graficas(
        Estadisticas.mejor_individuo_arreglo,
        Estadisticas.peor_individuo_arreglo,
        Estadisticas.promedio_arreglo,
        Estadisticas.generacion_arreglo,
        Data.num_generaciones
    )

#implementacion


def optimizacion():
    """
    1. Formación de parejas (usando estrategia dinámica)
    2. Cruza y mutación por cada pareja
    3. Poda final
    """
    # 1. Llamamos a la estrategia de formación que esté seleccionada
    estrategia_formacion = estrategias_formacion[Data.estrategia_formacion]
    parejas = estrategia_formacion()

    # 2. Obtenemos la estrategia de cruce y mutación
    estrategia_cruce = estrategias_cruce[Data.estrategia_cruce]
    estrategia_mutacion = estrategias_mutacion[Data.estrategia_mutacion]

    # Por cada pareja, cruzamos y mutamos
    for (ind1, ind2) in parejas:
        # Cruza
        nuevo_ind1, nuevo_ind2 = estrategia_cruce(ind1, ind2)

        # Mutación a nivel individuo (probabilidad prob_mutacion_ind)
        # Si el individuo "entra a mutar", aplicamos la mutación bit a bit
        if random.random() <= Data.prob_mutacion_ind:
            nuevo_ind1 = estrategia_mutacion(nuevo_ind1)
        if random.random() <= Data.prob_mutacion_ind:
            nuevo_ind2 = estrategia_mutacion(nuevo_ind2)

        # Guardar en la población
        guardar_nuevos_individuos(nuevo_ind1, nuevo_ind2)

    # 3. Poda
    estrategia_poda = estrategias_poda[Data.estrategia_poda]
    estrategia_poda()

def genetic_algorithm(parametros):
    """
    Ejecuta el algoritmo genético completo:
    1. Limpia datos
    2. Asigna parámetros
    3. Calcula representación y genera población
    4. Itera sobre generaciones
    5. Genera estadísticos y gráficas
    6. Muestra mejor individuo
    """

    vaciarDatos()
    
    Data.poblacion_inicial = int(parametros.p_inicial)
    Data.poblacion_maxima = int(parametros.p_max)
    Data.resolucion_deseada = float(parametros.res)
    Data.limite_inferior = float(parametros.lim_inf)
    Data.limite_superior = float(parametros.lim_sup)
    Data.prob_mutacion_ind = float(parametros.prob_ind)
    Data.prob_mutacion_gen = float(parametros.prob_gen)
    Data.tipo_problema_value = parametros.tipo_problema
    Data.num_generaciones = int(parametros.num_generaciones)
    Data.funcion = parametros.funcion

    
    
    
    calcular_datos()
    generar_primer_poblacion()

    for generacion in range(1, Data.num_generaciones + 1):
        Data.generacion_actual = generacion
        optimizacion()
        print(f"Generación: {Data.generacion_actual}")
        generar_estadisticas()

    generar_video(Data.num_generaciones)


    imprimir_mejor_individuo()

def imprimir_mejor_individuo():
    mensaje = (
        f"El mejor individuo es:\n{Estadisticas.mejor_individuo}\n\n"
        "Revisa la carpeta results/first-graph/video para ver el video de la evolución del fitness."
    )
    ventana_alerta = tk.Tk()
    ventana_alerta.title("Felicidades")

    etiqueta_mensaje = tk.Label(ventana_alerta, text=mensaje)
    etiqueta_mensaje.pack(padx=10, pady=10)
    
    ventana_alerta.mainloop()

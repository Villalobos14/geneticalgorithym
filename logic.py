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

    # Por defecto asignamos las nuevas estrategias requeridas
    estrategia_formacion = "FormacionAleatorioContiguidad"
    estrategia_cruce = "CruzaHibridacionParcial"
    estrategia_mutacion = "M2"            # Intercambio (bit a bit)
    estrategia_poda = "PodaEquitativaAleatoria"

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
    x = symbols('x')
    expresion = lambdify(x, funcion, 'numpy')
    return expresion(valor_x)

def calcular_valor_x(num_generado):
    if Data.limite_inferior >= Data.limite_superior:
        return Data.limite_superior + num_generado * Data.resolucion
    return Data.limite_inferior + num_generado * Data.resolucion

def calcular_datos():
    Data.rango = Data.limite_superior - Data.limite_inferior
    num_saltos = Data.rango / Data.resolucion_deseada
    num_puntos = num_saltos + 1
    Data.num_bits = math.log2(abs(num_puntos))
    if Data.num_bits % 1 != 0:
        Data.num_bits = math.ceil(Data.num_bits)
    else:
        Data.num_bits = int(Data.num_bits)

    Data.resolucion = Data.rango / (2**Data.num_bits)
    if Data.resolucion % 1 == 0:
        Data.resolucion = int(Data.resolucion)
    else:
        Data.resolucion = round(Data.resolucion, 4)

    Data.rango_numero = 2**Data.num_bits - 1
    Data.rango_punto_cruza = len(bin(Data.rango_numero)[2:])

def generar_primer_poblacion():
    for _ in range(Data.poblacion_inicial):
        num_generado = random.randint(1, Data.rango_numero)
        binario = format(num_generado, f"0{Data.num_bits}b")
        x = calcular_valor_x(num_generado)
        y = calcular_funcion(Data.funcion, x)
        individuo = Individuo(binario, num_generado, x, y)
        Data.poblacion_general.append(individuo)

###############################################################################
#             NUEVA ESTRATEGIA DE FORMACIÓN: ALEATORIO + CONTIGÜIDAD
###############################################################################

def formacion_aleatorio_contiguidad():
    """
    1) Seleccionar aleatoriamente la mitad de la población.
    2) Ordenar ese subconjunto de menor a mayor (según y).
    3) Formar parejas por contigüidad (1-2, 3-4, etc.)
    4) Cada pareja se concreta con prob de cruza Pc.
    """
    Pc = 0.5  # O la que desees
    p = Data.poblacion_general[:]
    if len(p) < 2:
        return []

    # a) Elegir aleatoriamente la mitad
    mitad = len(p)//2
    subset = random.sample(p, max(2, mitad))  # mínimo 2 individuos
    flag = (Data.tipo_problema_value != "Minimizacion")
    # b) Ordenarlos
    subset_ordenado = sorted(subset, key=lambda x: x.y, reverse=flag)

    # c) Formar parejas por contigüidad
    parejas = []
    for i in range(0, len(subset_ordenado)-1, 2):
        if random.random() <= Pc:
            parejas.append((subset_ordenado[i], subset_ordenado[i+1]))

    return parejas


def cruce_hibridacion_parcial(ind1, ind2):
    """
    "Hibridación parcial":
    - Longitud = n
    - Definir aleatoriamente k posiciones (0 < k < n/2)
    - Solo se intercambia la información en esas posiciones
    """
    n = len(ind1.binario)
    max_k = n // 2
    if max_k < 1:
        max_k = 1
    k = random.randint(1, max_k)  # 0 < k < n/2
    posiciones = random.sample(range(n), k)

    b1 = list(ind1.binario)
    b2 = list(ind2.binario)

    for pos in posiciones:
        b1[pos], b2[pos] = b2[pos], b1[pos]

    return "".join(b1), "".join(b2)



def mutacion_M2(binario):
    """
    "Intercambio bit a bit"
    Se evalúa bit a bit; si el bit muta, se intercambia con otra posición aleatoria.
    """
    b = list(binario)
    for i in range(len(b)):
        if random.random() <= Data.prob_mutacion_gen:
            j = random.randint(0, len(b)-1)
            b[i], b[j] = b[j], b[i]
    return "".join(b)

def mutacion_M1(binario):
    """
    Negación de bit
    """
    b = list(binario)
    for i in range(len(b)):
        if random.random() <= Data.prob_mutacion_gen:
            b[i] = '1' if b[i] == '0' else '0'
    return "".join(b)



def eliminar_repetidos(poblacion):
    conj = set()
    nueva = []
    for ind in poblacion:
        if ind.i not in conj:
            conj.add(ind.i)
            nueva.append(ind)
    return nueva

def poda_equitativa_aleatoria():
    """
    1) Mantener solo una copia de clones
    2) Mantener al mejor
    3) Si excede la población máx, se divide en dos grupos (mejor y peor).
       Si el excedente es k, de cada grupo se elimina aleatoriamente k/2 individuos.
    """
    unicos = eliminar_repetidos(Data.poblacion_general)
    Data.poblacion_general = unicos

    if len(Data.poblacion_general) == 0:
        return

    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(Data.poblacion_general, key=lambda x: x.y, reverse=flag)

    mejor = orden[0]

    resto = orden[1:]
    Data.poblacion_general = [mejor] + resto

    excedente = len(Data.poblacion_general) - Data.poblacion_maxima
    if excedente <= 0:
        return

    mitad = len(Data.poblacion_general)//2
    grupo_mejor = Data.poblacion_general[:mitad]
    grupo_peor = Data.poblacion_general[mitad:]

    a_eliminar_mejor = excedente//2
    a_eliminar_peor = excedente - a_eliminar_mejor  

    if a_eliminar_mejor > len(grupo_mejor)-1:
        a_eliminar_mejor = max(0, len(grupo_mejor)-1)  
    if a_eliminar_peor > len(grupo_peor):
        a_eliminar_peor = len(grupo_peor)

    if a_eliminar_mejor > 0 and len(grupo_mejor) > 1:
       
        sub_mejor = grupo_mejor[1:] 
        if len(sub_mejor) > 0:
            eliminar_sub = random.sample(sub_mejor, min(a_eliminar_mejor, len(sub_mejor)))
            grupo_mejor = [grupo_mejor[0]] + [x for x in sub_mejor if x not in eliminar_sub]

    if a_eliminar_peor > 0 and len(grupo_peor) > 0:
        eliminar_sub = random.sample(grupo_peor, min(a_eliminar_peor, len(grupo_peor)))
        grupo_peor = [x for x in grupo_peor if x not in eliminar_sub]

    Data.poblacion_general = grupo_mejor + grupo_peor



def formacion_A1():
    """
    Aleatoria por grupos (original)
    """
    n = 3
    p = Data.poblacion_general[:]
    parejas = []
    for ind in p:
        m = random.randint(0, n)
        cands = [c for c in p if c != ind]
        if m > 0 and len(cands) > 0:
            sel = random.sample(cands, min(m, len(cands)))
            for s in sel:
                parejas.append((ind, s))
    return parejas

def formacion_A5():
    """
    Todos con todos (original).
    """
    p = Data.poblacion_general[:]
    parejas = []
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            parejas.append((p[i], p[j]))
    return parejas

def cruce_C2(ind1, ind2):
    """
    Multipunto (ejemplo original).
    """
    longitud = Data.rango_punto_cruza
    num_puntos = random.randint(1, longitud - 1)
    puntos = sorted(random.sample(range(1, longitud), num_puntos))

    bin1 = list(ind1.binario)
    bin2 = list(ind2.binario)

    for i, p in enumerate(puntos):
        if i % 2 == 0:  
            bin1[p:], bin2[p:] = bin2[p:], bin1[p:]
    return "".join(bin1), "".join(bin2)

def mutacion_M1(binario):
    """
    Negación bit a bit (original).
    """
    b = list(binario)
    for i in range(len(b)):
        if random.random() <= Data.prob_mutacion_gen:
            b[i] = '1' if b[i] == '0' else '0'
    return "".join(b)

def poda_P2():
    """
    Aleatoria conservando el mejor (original).
    """
    unicos = eliminar_repetidos(Data.poblacion_general)
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(unicos, key=lambda x: x.y, reverse=flag)

    if len(orden) == 0:
        Data.poblacion_general = []
        return

    mejor = orden[0]
    resto = orden[1:]
    maximo = Data.poblacion_maxima

    if maximo > 1:
        sel = random.sample(resto, min(len(resto), maximo-1))
        Data.poblacion_general = [mejor] + sel
    else:
        Data.poblacion_general = [mejor]


#dicionario de estrategias

estrategias_formacion = {
    "FormacionAleatorioContiguidad": formacion_aleatorio_contiguidad,
    "A1": formacion_A1,
    "A5": formacion_A5
    
}

estrategias_cruce = {
    "CruzaHibridacionParcial": cruce_hibridacion_parcial,
    "C2": cruce_C2  
    
}

estrategias_mutacion = {
    "M1": mutacion_M1,
    "M2": mutacion_M2
}

def poda_P1():
    """
    Mantener 80% mejores (original).
    """
    unicos = eliminar_repetidos(Data.poblacion_general)
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(unicos, key=lambda x: x.y, reverse=flag)
    maximo = Data.poblacion_maxima
    corte = int(0.8 * maximo) if maximo > 0 else len(orden)
    Data.poblacion_general = orden[:corte]

estrategias_poda = {
    "PodaEquitativaAleatoria": poda_equitativa_aleatoria,
    "P2": poda_P2,
    "P1": poda_P1
}

#stats

def generar_estadisticas():
    if Data.tipo_problema_value == "Minimizacion":
        mejor_individuo = min(Data.poblacion_general, key=lambda x: x.y)
        peor_individuo = max(Data.poblacion_general, key=lambda x: x.y)
    else:
        mejor_individuo = max(Data.poblacion_general, key=lambda x: x.y)
        peor_individuo = min(Data.poblacion_general, key=lambda x: x.y)

    promedio = sum(ind.y for ind in Data.poblacion_general) / len(Data.poblacion_general)

    Estadisticas.mejor_individuo = mejor_individuo
    Estadisticas.peor_individuo = peor_individuo
    Estadisticas.promedio = promedio

    Estadisticas.mejor_individuo_arreglo.append(mejor_individuo.y)
    Estadisticas.peor_individuo_arreglo.append(peor_individuo.y)
    Estadisticas.promedio_arreglo.append(promedio)
    Estadisticas.generacion_arreglo.append(Data.generacion_actual)

    vx = [ind.x for ind in Data.poblacion_general]
    vy = [ind.y for ind in Data.poblacion_general]

    if Data.tipo_problema_value == "Minimizacion":
        my = min(Data.poblacion_general, key=lambda x: x.y)
        mx = my.x
        py = max(Data.poblacion_general, key=lambda x: x.y)
        px = py.x
    else:
        my = max(Data.poblacion_general, key=lambda x: x.y)
        mx = my.x
        py = min(Data.poblacion_general, key=lambda x: x.y)
        px = py.x

    generar_segunda_grafica(
        vx, vy,
        mx, my,
        px, py,
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


def optimizacion():

    formacion = estrategias_formacion[Data.estrategia_formacion]
    cruce = estrategias_cruce[Data.estrategia_cruce]
    mutacion = estrategias_mutacion[Data.estrategia_mutacion]
    poda = estrategias_poda[Data.estrategia_poda]

    parejas = formacion()
    for (ind1, ind2) in parejas:
        hijo1, hijo2 = cruce(ind1, ind2)
        if random.random() <= Data.prob_mutacion_ind:
            hijo1 = mutacion(hijo1)
        if random.random() <= Data.prob_mutacion_ind:
            hijo2 = mutacion(hijo2)

        guardar_nuevos_individuos(hijo1, hijo2)

    poda()


def guardar_nuevos_individuos(binario1, binario2):
    d1 = int(binario1, 2)
    d2 = int(binario2, 2)
    if Data.limite_inferior >= Data.limite_superior:
        x1 = Data.limite_superior + d1*Data.resolucion
        x2 = Data.limite_superior + d2*Data.resolucion
    else:
        x1 = Data.limite_inferior + d1*Data.resolucion
        x2 = Data.limite_inferior + d2*Data.resolucion

    y1 = calcular_funcion(Data.funcion, x1)
    y2 = calcular_funcion(Data.funcion, x2)

    i1 = Individuo(binario1, d1, x1, y1)
    i2 = Individuo(binario2, d2, x2, y2)
    Data.poblacion_general.append(i1)
    Data.poblacion_general.append(i2)


def genetic_algorithm(parametros):
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

   
    Data.estrategia_formacion = "FormacionAleatorioContiguidad"
    Data.estrategia_cruce = "CruzaHibridacionParcial"
    Data.estrategia_mutacion = "M2"  
    Data.estrategia_poda = "PodaEquitativaAleatoria"

    calcular_datos()
    generar_primer_poblacion()

    for gen in range(1, Data.num_generaciones + 1):
        Data.generacion_actual = gen
        optimizacion()
        print(f"Generación: {gen}")
        generar_estadisticas()

    generar_video(Data.num_generaciones)
    imprimir_mejor_individuo()

#resultados en vista

def imprimir_mejor_individuo():
    msg = (
        f"El mejor individuo es:\n{Estadisticas.mejor_individuo}\n\n"
        "Revisa la carpeta results/first-graph/video para ver el video de la evolución del fitness."
    )
    ventana_alerta = tk.Tk()
    ventana_alerta.title("Felicidades")
    etiqueta_mensaje = tk.Label(ventana_alerta, text=msg)
    etiqueta_mensaje.pack(padx=10, pady=10)
    ventana_alerta.mainloop()





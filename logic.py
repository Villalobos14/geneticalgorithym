import math
import random
import tkinter as tk
from sympy import symbols, lambdify
from graphs.graphic import generar_graficas
from video import generar_video
from graphs.graphic2 import generar_segunda_grafica

###############################################################################
#                       CLASES Y ESTRUCTURAS DE DATOS
###############################################################################

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

    # Estrategias: se seleccionan dinámicamente
    estrategia_formacion = "A6"           # Por defecto
    estrategia_cruce = "C1"               # Por defecto
    estrategia_mutacion = "M1"            # Por defecto
    estrategia_poda = "P2"                # Por defecto

class Estadisticas:
    mejor_individuo = None
    peor_individuo = None
    promedio = None
    mejor_individuo_arreglo = []
    peor_individuo_arreglo = []
    promedio_arreglo = []
    generacion_arreglo = []

###############################################################################
#                          LIMPIEZA DE DATOS
###############################################################################

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

###############################################################################
#                          FUNCIONES DE CÁLCULO
###############################################################################

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

    Data.resolucion = Data.rango / (2 ** Data.num_bits)
    if Data.resolucion % 1 == 0:
        Data.resolucion = int(Data.resolucion)
    else:
        Data.resolucion = round(Data.resolucion, 4)

    Data.rango_numero = 2**Data.num_bits - 1
    Data.rango_punto_cruza = len(bin(Data.rango_numero)[2:])

###############################################################################
#                     GENERACIÓN DE LA POBLACIÓN INICIAL
###############################################################################

def generar_primer_poblacion():
    for _ in range(Data.poblacion_inicial):
        num_generado = random.randint(1, Data.rango_numero)
        binario = format(num_generado, f"0{Data.num_bits}b")
        x = calcular_valor_x(num_generado)
        y = calcular_funcion(Data.funcion, x)
        individuo = Individuo(binario=binario, i=num_generado, x=x, y=y)
        Data.poblacion_general.append(individuo)

###############################################################################
#                      ESTRATEGIAS DE FORMACIÓN DE PAREJAS
###############################################################################
# EQUIVALENCIAS
# A1 = formacion_A1()
# A2 = formacion_A2()
# A3 = formacion_A3()
# A4 = formacion_A4()
# A5 = formacion_A5()
# A6 = formacion_A6()

def formacion_A1():
    """
    Estrategia "FormacionAleatoriaPorGrupos"
    Para cada individuo, generar un número aleatorio m entre [0, n],
    y elegir aleatoriamente esos m individuos como parejas.
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

def formacion_A2():
    """
    Estrategia "FormacionUmbralProb"
    Cada individuo, con probabilidad Pc, elige 1 pareja aleatoria.
    """
    Pc = 0.5
    p = Data.poblacion_general[:]
    parejas = []
    for ind in p:
        if random.random() <= Pc:
            candidato = random.choice([x for x in p if x != ind])
            parejas.append((ind, candidato))
    return parejas

def formacion_A3():
    """
    Estrategia "FormacionParticionMitad"
    Ordenar y partir la población en mitad "mejor" y mitad "peor"
    luego cruzar cada uno de los mejores con cada uno de los peores.
    """
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(Data.poblacion_general, key=lambda x: x.y, reverse=flag)
    mitad = int(len(orden)/2)
    mejor = orden[:mitad]
    peor = orden[mitad:]
    parejas = []
    for m in mejor:
        for p in peor:
            parejas.append((m, p))
    return parejas

def formacion_A4():
    """
    Estrategia "FormacionPorcentajeElite"
    Ordena la población, toma un porcentaje (ej 0.3) de los mejores y los cruza con el resto.
    """
    porcentaje = 0.3
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(Data.poblacion_general, key=lambda x: x.y, reverse=flag)
    num_mejores = max(1, int(len(orden)*porcentaje))
    mejores = orden[:num_mejores]
    resto = orden[num_mejores:]
    parejas = []
    for m in mejores:
        for r in resto:
            parejas.append((m, r))
    return parejas

def formacion_A5():
    """
    Estrategia "FormacionTodosConTodos"
    Cada individuo se cruza con todos los demás.
    """
    p = Data.poblacion_general[:]
    parejas = []
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            parejas.append((p[i], p[j]))
    return parejas

def formacion_A6():
    """
    Estrategia "FormacionRuleta"
    Transforma la aptitud para que sea no negativa.
    Luego selecciona parejas por ruleta.
    """
    p = Data.poblacion_general[:]
    flag = (Data.tipo_problema_value != "Minimizacion")
    fitness_vals = [ind.y for ind in p]

    if flag:
        min_fit = min(fitness_vals)
        offset = -min_fit if min_fit < 0 else 0
        fit_transform = [f + offset for f in fitness_vals]
    else:
        invert = [-f for f in fitness_vals]
        min_fit = min(invert)
        offset = -min_fit if min_fit < 0 else 0
        fit_transform = [v + offset for v in invert]

    total_fit = sum(fit_transform)
    if total_fit == 0:
        # Si todo es 0, fallback a "todos con todos"
        return formacion_A5()

    def ruleta():
        r = random.uniform(0, total_fit)
        acum = 0
        for i, f in enumerate(fit_transform):
            acum += f
            if acum >= r:
                return p[i]
        return p[-1]

    parejas = []
    for _ in range(len(p)//2):
        ind1 = ruleta()
        ind2 = ruleta()
        if ind1 != ind2:
            parejas.append((ind1, ind2))
    return parejas

def formacion_mejor_tercio_domina():
    """
    ESTRATEGIA NUEVA SEGÚN ENUNCIADO:
    - Dividir la población en 3 grupos: mejor, intermedio y peor.
    - mejor -> puede cruzar con toda la población
    - intermedio -> con intermedio y peor
    - peor -> solo con peor
    - Todo con una probabilidad de cruza Pc
    """
    Pc = 0.5
    p = Data.poblacion_general[:]
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(p, key=lambda x: x.y, reverse=flag)
    tercio = len(orden)//3

    if tercio == 0:
        # Fallback si la población < 3
        return formacion_A5()

    grupo_mejor = orden[:tercio]
    grupo_intermedio = orden[tercio:2*tercio]
    grupo_peor = orden[2*tercio:]

    parejas = []

    # mejor con toda la población (excepto sí mismo)
    for ind in grupo_mejor:
        for otro in p:
            if otro != ind and random.random() <= Pc:
                parejas.append((ind, otro))

    # intermedio con intermedio y peor
    for ind in grupo_intermedio:
        for otro in (grupo_intermedio + grupo_peor):
            if otro != ind and random.random() <= Pc:
                parejas.append((ind, otro))

    # peor con peor
    for ind in grupo_peor:
        for otro in grupo_peor:
            if otro != ind and random.random() <= Pc:
                parejas.append((ind, otro))

    return parejas

###############################################################################
#                       DICCIONARIO DE ESTRATEGIAS DE FORMACIÓN
###############################################################################

estrategias_formacion = {
    "A1": formacion_A1,
    "A2": formacion_A2,
    "A3": formacion_A3,
    "A4": formacion_A4,
    "A5": formacion_A5,
    "A6": formacion_A6,
    "FormacionMejorTercioDomina": formacion_mejor_tercio_domina
}

###############################################################################
#                          ESTRATEGIAS DE CRUZA
###############################################################################
# EQUIVALENCIAS
# C1 = cruce_C1()
# C2 = cruce_C2()
# C3 = cruce_C3()

def cruce_C1(ind1, ind2):
    """
    Estrategia "CruzaPuntoAleatorio"
    Un punto de cruza aleatorio único.
    """
    punto_cruza = random.randint(1, Data.rango_punto_cruza - 1)
    s1 = ind1.binario[:punto_cruza]
    s2 = ind1.binario[punto_cruza:]
    t1 = ind2.binario[:punto_cruza]
    t2 = ind2.binario[punto_cruza:]

    hijo1 = s1 + t2
    hijo2 = t1 + s2
    return hijo1, hijo2

def cruce_C2(ind1, ind2):
    """
    Estrategia "CruzaMultipunto"
    Se elige aleatoriamente cuántos puntos de cruza y se alternan segmentos.
    """
    longitud = Data.rango_punto_cruza
    num_puntos = random.randint(1, longitud - 1)
    puntos = sorted(random.sample(range(1, longitud), num_puntos))

    bin1 = list(ind1.binario)
    bin2 = list(ind2.binario)

    for i, p in enumerate(puntos):
        if i % 2 == 0:  # alterna
            bin1[p:], bin2[p:] = bin2[p:], bin1[p:]
    return "".join(bin1), "".join(bin2)

def cruce_C3(ind1, ind2):
    """
    Estrategia "CruzaPuntoFijo"
    Un solo punto de cruza fijo (por ejemplo, la mitad).
    """
    punto_fijo = Data.rango_punto_cruza // 2
    s1 = ind1.binario[:punto_fijo]
    s2 = ind1.binario[punto_fijo:]
    t1 = ind2.binario[:punto_fijo]
    t2 = ind2.binario[punto_fijo:]

    hijo1 = s1 + t2
    hijo2 = t1 + s2
    return hijo1, hijo2

def invertir_cadena(cad):
    return cad[::-1]

def cruza_punto_aleatorio_inverso(ind1, ind2):
    """
    ESTRATEGIA NUEVA SEGÚN ENUNCIADO:
    - Un punto de cruza aleatorio p
    - Hijo1 = s1 + inverso(t2)
    - Hijo2 = t1 + inverso(s2)
    """
    punto_cruza = random.randint(1, Data.rango_punto_cruza - 1)
    s1 = ind1.binario[:punto_cruza]
    s2 = ind1.binario[punto_cruza:]
    t1 = ind2.binario[:punto_cruza]
    t2 = ind2.binario[punto_cruza:]

    hijo1 = s1 + invertir_cadena(t2)
    hijo2 = t1 + invertir_cadena(s2)
    return hijo1, hijo2

###############################################################################
#                        DICCIONARIO DE ESTRATEGIAS DE CRUZA
###############################################################################

estrategias_cruce = {
    "C1": cruce_C1,
    "C2": cruce_C2,
    "C3": cruce_C3,
    "CruzaPuntoAleatorioInverso": cruza_punto_aleatorio_inverso
}

###############################################################################
#                          ESTRATEGIAS DE MUTACIÓN
###############################################################################
# EQUIVALENCIAS
# M1 = mutacion_M1()
# M2 = mutacion_M2()

def mutacion_M1(binario):
    """
    Estrategia "MutacionNegacionBit"
    Se evalúa bit a bit con prob_mutacion_gen y se niega el bit (0->1 / 1->0).
    """
    b = list(binario)
    for i in range(len(b)):
        if random.random() <= Data.prob_mutacion_gen:
            b[i] = '1' if b[i] == '0' else '0'
    return "".join(b)

def mutacion_M2(binario):
    """
    ESTRATEGIA SEGÚN EL NUEVO ENUNCIADO DE "Intercambio":
    - Se evalúa si el individuo muta (eso se hace en el flujo principal).
    - Si el individuo entra a mutar, se evalúa cada bit con prob_mutacion_gen,
      y si el bit muta, se intercambia con otra posición aleatoria del individuo.
    """
    b = list(binario)
    # para cada bit, si "muta", hacemos un intercambio
    for i in range(len(b)):
        if random.random() <= Data.prob_mutacion_gen:
            j = random.randint(0, len(b)-1)
            b[i], b[j] = b[j], b[i]
    return "".join(b)

estrategias_mutacion = {
    "M1": mutacion_M1,
    "M2": mutacion_M2
}

###############################################################################
#                          ESTRATEGIAS DE PODA
###############################################################################
# EQUIVALENCIAS
# P1 = poda_P1
# P2 = poda_P2
# P3 = poda_P3

def eliminar_repetidos(poblacion):
    conjunto_i = set()
    poblacion_sin_repetidos = []
    for ind in poblacion:
        if ind.i not in conjunto_i:
            conjunto_i.add(ind.i)
            poblacion_sin_repetidos.append(ind)
    return poblacion_sin_repetidos

def poda_P1():
    """
    Estrategia "PodaMantenerMejores"
    Mantiene ~80% de la población ordenada por aptitud (quita duplicados).
    """
    unicos = eliminar_repetidos(Data.poblacion_general)
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(unicos, key=lambda x: x.y, reverse=flag)

    maximo = Data.poblacion_maxima
    corte = int(0.8 * maximo) if maximo > 0 else len(orden)
    Data.poblacion_general = orden[:corte]

def poda_P2():
    """
    Estrategia "PodaAleatoriaConservandoMejor"
    1) Quita duplicados
    2) Mantiene al mejor
    3) El resto se selecciona aleatoriamente para no pasar la población máxima
    """
    unicos = eliminar_repetidos(Data.poblacion_general)
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(unicos, key=lambda x: x.y, reverse=flag)

    if len(orden) == 0:
        Data.poblacion_general = []
        return

    mejor = orden[0]
    resto = orden[1:] if len(orden) > 1 else []
    maximo = Data.poblacion_maxima

    if maximo > 1:
        sel = random.sample(resto, min(len(resto), maximo-1))
        Data.poblacion_general = [mejor] + sel
    else:
        Data.poblacion_general = [mejor]

def poda_P3():
    """
    Estrategia "PodaAgruparPorClases"
    Divide la población (ordenada) en 3 partes y selecciona al azar en cada clase.
    """
    unicos = eliminar_repetidos(Data.poblacion_general)
    flag = (Data.tipo_problema_value != "Minimizacion")
    orden = sorted(unicos, key=lambda x: x.y, reverse=flag)

    if len(orden) < 3:
        Data.poblacion_general = orden
        return

    tercio = len(orden)//3
    c1 = orden[:tercio]
    c2 = orden[tercio: 2*tercio]
    c3 = orden[2*tercio:]

    s1 = random.sample(c1, max(1, len(c1)//2)) if len(c1) > 1 else c1
    s2 = random.sample(c2, max(1, len(c2)//2)) if len(c2) > 1 else c2
    s3 = random.sample(c3, max(1, len(c3)//2)) if len(c3) > 1 else c3

    nueva_poblacion = s1 + s2 + s3

    maximo = Data.poblacion_maxima
    if len(nueva_poblacion) > maximo:
        nueva_poblacion = random.sample(nueva_poblacion, maximo)

    Data.poblacion_general = nueva_poblacion

estrategias_poda = {
    "P1": poda_P1,
    "P2": poda_P2,
    "P3": poda_P3
}

###############################################################################
#                         CÁLCULO DE ESTADÍSTICAS Y GRÁFICAS
###############################################################################

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
        vx, vy, mx, my, px, py,
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

###############################################################################
#                    CICLO DE OPTIMIZACIÓN (POR GENERACIÓN)
###############################################################################

def optimizacion():
    formacion = estrategias_formacion[Data.estrategia_formacion]
    cruce = estrategias_cruce[Data.estrategia_cruce]
    mutacion = estrategias_mutacion[Data.estrategia_mutacion]
    poda = estrategias_poda[Data.estrategia_poda]

    parejas = formacion()

    for ind1, ind2 in parejas:
        hijo1, hijo2 = cruce(ind1, ind2)
        # Prob de que el individuo entre a mutar
        if random.random() <= Data.prob_mutacion_ind:
            hijo1 = mutacion(hijo1)
        if random.random() <= Data.prob_mutacion_ind:
            hijo2 = mutacion(hijo2)

        guardar_nuevos_individuos(hijo1, hijo2)

    poda()

###############################################################################
#                              GUARDAR DESCENDIENTES
###############################################################################

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

    ind1 = Individuo(binario=binario1, i=d1, x=x1, y=y1)
    ind2 = Individuo(binario=binario2, i=d2, x=x2, y=y2)
    Data.poblacion_general.append(ind1)
    Data.poblacion_general.append(ind2)

###############################################################################
#                        ALGORITMO GENÉTICO (FUNCION PRINCIPAL)
###############################################################################

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

    calcular_datos()
    generar_primer_poblacion()

    for gen in range(1, Data.num_generaciones + 1):
        Data.generacion_actual = gen
        optimizacion()
        print(f"Generación: {Data.generacion_actual}")
        generar_estadisticas()

    generar_video(Data.num_generaciones)
    imprimir_mejor_individuo()

###############################################################################
#                          MOSTRAR MEJOR INDIVIDUO
###############################################################################

def imprimir_mejor_individuo():
    msg = (
        f"El mejor individuo es:\n{Estadisticas.mejor_individuo}\n\n"
        "Revisa la carpeta results/first-graph/video para ver el video de la evolución del fitness."
    )
    ventana_alerta = tk.Tk()
    ventana_alerta.title("RESULTADOS")
    etiqueta_mensaje = tk.Label(ventana_alerta, text=msg)
    etiqueta_mensaje.pack(padx=10, pady=10)
    ventana_alerta.mainloop()

###############################################################################
#                      TABLA DE EQUIVALENCIAS (COMENTARIO)
###############################################################################
"""
# FORMACION (A1–A6):
# A1 = "FormacionAleatoriaPorGrupos"
# A2 = "FormacionUmbralProb"
# A3 = "FormacionParticionMitad"
# A4 = "FormacionPorcentajeElite"
# A5 = "FormacionTodosConTodos"
# A6 = "FormacionRuleta"
# FormacionMejorTercioDomina = "El mejor tercio domina" (Nueva según enunciado)

# CRUZA (C1–C3):
# C1 = "CruzaPuntoAleatorio"
# C2 = "CruzaMultipunto"
# C3 = "CruzaPuntoFijo"
# CruzaPuntoAleatorioInverso = "Un punto + inversión de la subcadena" (Nueva)

# MUTACION (M1–M2):
# M1 = "MutacionNegacionBit"
# M2 = "MutacionIntercambio" (actualizada para el enunciado: cada bit que muta hace un swap)

# PODA (P1–P3):
# P1 = "PodaMantenerMejores"
# P2 = "PodaAleatoriaConservandoMejor" (tal cual enunciado)
# P3 = "PodaAgruparPorClases"
"""

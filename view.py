import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from logic_regresion import genetic_algorithm_regresion

def seleccionar_archivo():
    """Selecciona un archivo CSV desde un diálogo."""
    ruta = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("Archivo CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )
    if ruta:
        entry_csv.delete(0, tk.END)
        entry_csv.insert(0, ruta)

def ejecutar_ga():
    """Ejecuta el algoritmo genético y muestra los resultados."""
    ruta_csv = entry_csv.get()
    if not ruta_csv:
        messagebox.showwarning("Aviso", "Por favor selecciona un archivo CSV primero.")
        return

    try:
        params = {
            "poblacion_inicial": int(spin_p_inicial.get()),
            "poblacion_maxima": int(spin_p_max.get()),
            "num_generaciones": int(spin_generaciones.get()),
            "prob_mutacion_gen": float(spin_prob_gen.get())
        }

        mejor_mse, mejor_pesos, pesos_sklearn, mse_sklearn = genetic_algorithm_regresion(ruta_csv, params)

        # Mostrar resultados en messagebox
        resultados = (
            f"Resultados del ajuste de regresión:\n\n"
            f"Mejor MSE (GA): {mejor_mse:.6f}\n"
            f"Pesos obtenidos por GA: {mejor_pesos}\n\n"
            f"Pesos obtenidos por scikit-learn: {pesos_sklearn}\n"
            f"MSE con scikit-learn: {mse_sklearn:.6f}\n"
        )
        messagebox.showinfo("Resultados", resultados)

    except Exception as e:
        messagebox.showerror("Error", f"Hubo un problema: {str(e)}")

# Interfaz gráfica (Tkinter)
root = tk.Tk()
root.title("GA - Regresión")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(row=0, column=0, sticky="NSEW")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Etiqueta y campo para seleccionar archivo CSV
ttk.Label(mainframe, text="Archivo CSV:").grid(row=0, column=0, sticky="W")
entry_csv = ttk.Entry(mainframe, width=40)
entry_csv.grid(row=0, column=1, sticky="WE")
ttk.Button(mainframe, text="Examinar...", command=seleccionar_archivo).grid(row=0, column=2, padx=5)

# Población inicial
ttk.Label(mainframe, text="Población inicial:").grid(row=1, column=0, sticky="W")
spin_p_inicial = tk.Spinbox(mainframe, from_=10, to=500, width=10)
spin_p_inicial.grid(row=1, column=1, sticky="W")

# Población máxima
ttk.Label(mainframe, text="Población máxima:").grid(row=2, column=0, sticky="W")
spin_p_max = tk.Spinbox(mainframe, from_=10, to=500, width=10)
spin_p_max.grid(row=2, column=1, sticky="W")

# Número de generaciones
ttk.Label(mainframe, text="Generaciones:").grid(row=3, column=0, sticky="W")
spin_generaciones = tk.Spinbox(mainframe, from_=10, to=1000, width=10)
spin_generaciones.grid(row=3, column=1, sticky="W")

# Probabilidad de mutación
ttk.Label(mainframe, text="Prob. de mutación:").grid(row=4, column=0, sticky="W")
spin_prob_gen = tk.Spinbox(mainframe, from_=0.0, to=1.0, increment=0.01, width=10)
spin_prob_gen.grid(row=4, column=1, sticky="W")

# Botón para ejecutar el GA
ttk.Button(mainframe, text="Ejecutar GA", command=ejecutar_ga).grid(row=5, column=1, pady=10)

root.mainloop()

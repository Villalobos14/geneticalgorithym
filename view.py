import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from logic_regresion import genetic_regression

def seleccionar_archivo():
    ruta = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("Archivo CSV", ".csv"), ("Todos los archivos", ".*")]
    )
    if ruta:
        entry_csv.delete(0, tk.END)
        entry_csv.insert(0, ruta)

def ejecutar_ga():
    ruta_csv = entry_csv.get()
    if not ruta_csv:
        messagebox.showwarning("Aviso", "Por favor selecciona un archivo CSV primero.")
        return
    try:
        params = {
            "popsize": int(spin_p_inicial.get()),
            "maxpop": int(spin_p_max.get()),
            "gens": int(spin_generaciones.get()),
            "mutprob": float(spin_prob_gen.get())
        }
        bestmse, bestw, weights_sklearn, mse_sklearn = genetic_regression(ruta_csv, params)
        
        resultados = (
            f"Resultados del ajuste de regresión:\n\n"
            f"Mejor MSE (GA): {bestmse:.6f}\n"
            f"Pesos obtenidos por GA: {bestw}\n\n"
            f"Pesos obtenidos por scikit-learn: {weights_sklearn}\n"
            f"MSE con scikit-learn: {mse_sklearn:.6f}\n"
        )
        messagebox.showinfo("Resultados", resultados)
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un problema: {str(e)}")

# Configuración de la ventana principal
root = tk.Tk()
root.title("Algoritmo Genético - Regresión Lineal")
root.geometry("600x400")
root.configure(bg='#f0f0f0')

# Frame principal
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Frame para el archivo
file_frame = ttk.LabelFrame(main_frame, text="Selección de Archivo", padding="10")
file_frame.pack(fill=tk.X, pady=(0, 10))

entry_csv = ttk.Entry(file_frame, width=50)
entry_csv.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
ttk.Button(file_frame, text="Examinar...", command=seleccionar_archivo).pack(side=tk.LEFT)

# Frame para los parámetros
params_frame = ttk.LabelFrame(main_frame, text="Parámetros del Algoritmo", padding="10")
params_frame.pack(fill=tk.BOTH, expand=True)

# Grid de parámetros
params = [
    ("Población inicial:", "spin_p_inicial", 10, 500),
    ("Población máxima:", "spin_p_max", 10, 500),
    ("Generaciones:", "spin_generaciones", 10, 1000),
    ("Probabilidad de mutación:", "spin_prob_gen", 0.0, 1.0, 0.01)
]

for i, (label_text, var_name, from_, to, *args) in enumerate(params):
    ttk.Label(params_frame, text=label_text).grid(row=i, column=0, sticky="W", padx=5, pady=5)
    if len(args) > 0:  # Para la probabilidad de mutación
        spinbox = tk.Spinbox(params_frame, from_=from_, to=to, increment=args[0], width=15)
    else:
        spinbox = tk.Spinbox(params_frame, from_=from_, to=to, width=15)
    spinbox.grid(row=i, column=1, sticky="W", padx=5, pady=5)
    globals()[var_name] = spinbox

# Frame para el botón de ejecución
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill=tk.X, pady=20)

execute_button = ttk.Button(button_frame, text="Ejecutar Algoritmo Genético", command=ejecutar_ga)
execute_button.pack()

# Estilo
style = ttk.Style()
style.configure('TLabel', font=('Arial', 10))
style.configure('TButton', font=('Arial', 10))
style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))

root.mainloop()
import tkinter as tk
from tkinter import ttk
import co2_vorhersage
import produktion_vohersage
import produktionsvorhersage
import stahl_ml_prognose

def run_co2_prediction():
    result = co2_vorhersage.predict()  # Beispiel: Funktion aus dem Skript
    result_label.config(text=f"CO₂ Vorhersage: {result:.2f}")

def run_production_prediction():
    result = produktion_vohersage.predict()  # Funktion aus anderem Modul
    result_label.config(text=f"Produktions-Vorhersage: {result:.2f}")

root = tk.Tk()
root.title("Stahl Prognosen")

ttk.Button(root, text="CO₂ Vorhersage", command=run_co2_prediction).grid(column=0, row=0)
ttk.Button(root, text="Produktions-Vorhersage", command=run_production_prediction).grid(column=0, row=1)

result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=2)

root.mainloop()

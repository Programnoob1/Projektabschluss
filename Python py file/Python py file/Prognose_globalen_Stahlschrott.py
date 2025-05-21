#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CSV-Datei einlesen (Pfad anpassen)
dateipfad = r"C:\Users\User\Desktop\Abschlussprojekte\Weiterbildung\CSV Datei\Python\Prognose der globalen Stahlschrott.csv"
df = pd.read_csv(dateipfad,encoding="latin1")  # Falls die Trennung Tabulator ist

# Spaltennamen bereinigen
df.columns = df.columns.str.strip()

# Prüfen, ob die erwarteten Spalten existieren
print("Verfügbare Spalten:", df.columns)

# Sicherstellen, dass die Spalten existieren
if all(col in df.columns for col in ["Jahr", "Angebot", "Nachfrage", "Nachfrageüberhang"]):
    jahr = df["Jahr"].values.reshape(-1, 1)
    nachfrage = df["Nachfrage"].values
    nachfrageueberhang = df["Nachfrageüberhang"].values
else:
    print("Fehler: Mindestens eine erwartete Spalte fehlt!")

# Machine-Learning-Modelle für Prognosen
model_nachfrage = LinearRegression()
model_nachfrage.fit(jahr, nachfrage)

model_nachfrageueberhang = LinearRegression()
model_nachfrageueberhang.fit(jahr, nachfrageueberhang)

# Prognose für zukünftige Jahre
jahre_zukunft = np.array([2035, 2040, 2045]).reshape(-1, 1)
prognose_nachfrage = model_nachfrage.predict(jahre_zukunft)
prognose_nachfrageueberhang = model_nachfrageueberhang.predict(jahre_zukunft)

# Ergebnisse anzeigen
print("\nPrognosen für globale Stahlschrott-Nachfrage und Nachfrageüberhang:")
for j, p_n, p_u in zip(jahre_zukunft.flatten(), prognose_nachfrage, prognose_nachfrageueberhang):
    print(f"Jahr {j}: Nachfrage = {p_n:.2f} Mio. t, Nachfrageüberhang = {p_u:.2f} Mio. t")

# Visualisierung
plt.figure(figsize=(10, 5))
plt.scatter(jahr, nachfrage, color="blue", label="Historische Nachfrage")
plt.plot(jahre_zukunft, prognose_nachfrage, color="red", linestyle="dashed", label="Prognose Nachfrage")

plt.scatter(jahr, nachfrageueberhang, color="green", label="Historischer Nachfrageüberhang")
plt.plot(jahre_zukunft, prognose_nachfrageueberhang, color="orange", linestyle="dashed", label="Prognose Nachfrageüberhang")

plt.xlabel("Jahr")
plt.ylabel("Mio. t")
plt.title("Prognose der globalen Stahlschrott-Nachfrage & Nachfrageüberhang")
plt.legend()
plt.show()


# In[ ]:





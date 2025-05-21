import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Daten laden
df = pd.read_csv("C:/Users/User/Documents/Abschlussprojekt/Erzeugung/gesamtproduktion.csv")

# Eingaben und Ziel definieren
X = df["Jahr"].values.reshape(-1, 1)
y = df["Produktion_Mio_Tonnen"].values

# Modell trainieren
model = LinearRegression()
model.fit(X, y)

# Vorhersage bis 2035
future_years = np.arange(2024, 2036).reshape(-1, 1)
predictions = model.predict(future_years)

# Visualisierung
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Echte Daten")
plt.plot(future_years, predictions, color="red", label="Prognose")
plt.xlabel("Jahr")
plt.ylabel("Produktion (Mio. Tonnen)")
plt.title("Prognose der weltweiten Stahlproduktion")
plt.legend()
plt.grid(True)
plt.show()

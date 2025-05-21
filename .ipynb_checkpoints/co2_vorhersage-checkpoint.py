import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Daten laden
df = pd.read_csv("C:/Users/User/Documents/Abschlussprojekt/Erzeugung/co2_stahl_1990-2023.csv")

# Features & Ziel
X = df[["Jahr", "Produktion_Mio_t", "Strompreis", "Eisenschrott_Quote", "CO2_Preis"]]
y = df["CO2_pro_t"]

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Polynomiale Features (Grad 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly, X_test_poly = poly.fit_transform(X_train), poly.transform(X_test)

# Modell trainieren
model = LinearRegression().fit(X_train_poly, y_train)

# Güte
pred_test = model.predict(X_test_poly)
print("MAE:", round(mean_absolute_error(y_test, pred_test), 2), "kg CO₂/t")

# Zukunfts-Szenario 2024-2035
future = pd.DataFrame({
    "Jahr": range(2024, 2036),
    "Produktion_Mio_t": np.linspace(1_870, 2_000, 12),   # moderate Wachstumsannahme
    "Strompreis": np.linspace(90, 110, 12),              # leicht steigend
    "Eisenschrott_Quote": np.linspace(30, 45, 12),       # mehr EAF
    "CO2_Preis": np.linspace(90, 150, 12)                # strenger ETS
})
pred_future = model.predict(poly.transform(future))

future["CO2_pro_t_Pred"] = pred_future
future["Gesamt_CO2_Mio_t"] = future["CO2_pro_t_Pred"] * future["Produktion_Mio_t"] / 1000

print(future[["Jahr", "CO2_pro_t_Pred", "Gesamt_CO2_Mio_t"]])

# Plot
plt.plot(df["Jahr"], df["CO2_pro_t"], label="Historisch")
plt.plot(future["Jahr"], future["CO2_pro_t_Pred"], '--', label="Prognose")
plt.ylabel("kg CO₂ pro t Rohstahl")
plt.grid(); plt.legend(); plt.show()

#df1 = pd.DataFrame(future)
#df1.to_csv("C:/Users/User/Documents/Abschlussprojekt/Erzeugung/co2_prognose_2024-2035.csv", index=False, sep=",")

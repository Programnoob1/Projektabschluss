import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Daten laden
df = pd.read_csv(r"C:\Users\hogu2\Desktop\Projektabschluss\verbrauch_pro__kopf\co2_stahl_1990-2023.csv")

# Features & Ziel
X = df[["Jahr", "Produktion_Mio_t", "Strompreis", "Eisenschrott_Quote", "CO2_Preis"]]
y = df["CO2_pro_t"]

# Feature-Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True, random_state=42)

# Polynomiale Features (Grad 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly, X_test_poly = poly.fit_transform(X_train), poly.transform(X_test)

# Ridge Regression für bessere Generalisierung
model = Ridge(alpha=1.0).fit(X_train_poly, y_train)

# Güte-Metriken
pred_test = model.predict(X_test_poly)
mae = round(mean_absolute_error(y_test, pred_test), 2)
r2 = round(r2_score(y_test, pred_test), 3)

print(f"MAE: {mae} kg CO₂/t")
print(f"R²: {r2}")

# Zukunfts-Szenario 2024-2035
future = pd.DataFrame({
    "Jahr": range(2024, 2036),
    "Produktion_Mio_t": np.linspace(1_870, 2_000, 12),
    "Strompreis": np.linspace(90, 110, 12),
    "Eisenschrott_Quote": np.linspace(30, 45, 12),
    "CO2_Preis": np.linspace(90, 150, 12)
})
future_scaled = scaler.transform(future)
pred_future = model.predict(poly.transform(future_scaled))

future["CO2_pro_t_Pred"] = pred_future
future["Gesamt_CO2_Mio_t"] = future["CO2_pro_t_Pred"] * future["Produktion_Mio_t"] / 1000

print(future[["Jahr", "CO2_pro_t_Pred", "Gesamt_CO2_Mio_t"]])

# Plot
plt.plot(df["Jahr"], df["CO2_pro_t"], label="Historisch")
plt.plot(future["Jahr"], future["CO2_pro_t_Pred"], '--', label="Prognose")
plt.ylabel("kg CO₂ pro t Rohstahl")
plt.grid(); plt.legend(); plt.show()

# Speichern der Prognose-Datei mit einem relativen Pfad
future.to_csv("verbrauch_pro__kopf/co2_prognose_2024-2035.csv", index=False, sep=",")

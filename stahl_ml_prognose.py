# stahl_ml_prognose.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==== 1. Beispiel-Daten laden ====
data = {
    "Jahr": list(range(2010, 2024)),
    "Stahlpreis_USD_CO2": [600, 620, 590, 580, 570, 510, 500, 520, 550, 560, 620, 680, 660, 650],
    "Tonnen_CO2": [2100, 2150, 2130, 2110, 2100, 2080, 2070, 2050, 2025, 2000, 1980, 1960, 1950, 1930],
    "Produktion_Mio_Tonnen": [1433, 1490, 1470, 1530, 1570, 1620, 1650, 1700, 1780, 1820, 1860, 1900, 1895, 1870]
}
df = pd.DataFrame(data)

# ==== 2. Separate Modelle für Preis & CO2 erstellen ====

# a) CO2-Modell (nur Jahr)
co2_X = df[["Jahr"]].values
co2_y = df["Tonnen_CO2"].values
co2_model = LinearRegression().fit(co2_X, co2_y)

# b) Preis-Modell (nur Jahr)
price_X = df[["Jahr"]].values
price_y = df["Stahlpreis_USD_CO2"].values
price_model = LinearRegression().fit(price_X, price_y)

# ==== 3. Hauptmodell: Produktion in Abhängigkeit von Preis & CO2 ====
X = df[["Jahr", "Stahlpreis_USD_CO2", "Tonnen_CO2"]].values
y = df["Produktion_Mio_Tonnen"].values

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
prod_model = LinearRegression().fit(X_train, y_train)

# ==== 4. Prognose für 2024-2030 ====
future_years = np.arange(2024, 2031).reshape(-1, 1)
future_prices = price_model.predict(future_years)
future_co2 = co2_model.predict(future_years)

# Zusammenstellen
future_df = pd.DataFrame({
    "Jahr": future_years.flatten(),
    "Stahlpreis_USD_CO2": future_prices,
    "Tonnen_CO2": future_co2
})

# Prognose der Produktion
future_poly = poly.transform(future_df)
production_preds = prod_model.predict(future_poly)
future_df["Produktion_Mio_Tonnen"] = np.round(production_preds, 1)

# ==== 5. Ergebnis anzeigen ====
print("\n🔮 Prognose 2024–2030:")
print(future_df)

# ==== 6. Plot (optional) ====
plt.plot(df["Jahr"], df["Produktion_Mio_Tonnen"], label="Historisch")
plt.plot(future_df["Jahr"], future_df["Produktion_Mio_Tonnen"], label="Prognose", linestyle="--")
plt.xlabel("Jahr")
plt.ylabel("Produktion (Mio. Tonnen)")
plt.title("Stahlproduktion: Historie & Prognose")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



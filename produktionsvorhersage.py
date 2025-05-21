# 📦 Pakete importieren
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 📁 Daten laden
df = pd.read_csv(r"C:\Users\hogu2\Desktop\Projektabschluss\verbrauch_pro__kopf\gesamtproduktion.csv")

# 🎯 Features und Ziel definieren
X = df["Jahr"].values.reshape(-1, 1)
y = df["Produktion_Mio_Tonnen"].values

# 🔀 Daten aufteilen (z. B. 80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# 📊 Vorhersage auf Testdaten
y_pred = model.predict(X_test)

# 🧾 Fehlerbewertung
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("📉 Testdaten-Fehler:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# 🔮 Prognose auf zukünftige Jahre (2024–2035)
future_years = np.arange(2024, 2036).reshape(-1, 1)
future_preds = model.predict(future_years)

# 📈 Visualisierung
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color="blue", label="Trainingsdaten")
plt.scatter(X_test, y_test, color="green", label="Testdaten")
plt.plot(X, model.predict(X), color="black", label="Modell")
plt.plot(future_years, future_preds, color="red", linestyle="--", label="Prognose 2024–2035")
plt.xlabel("Jahr")
plt.ylabel("Produktion (Mio. Tonnen)")
plt.title("Lineare Regression: Stahlproduktion weltweit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

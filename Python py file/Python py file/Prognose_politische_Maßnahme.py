#!/usr/bin/env python
# coding: utf-8

# In[194]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Ursprüngliche Daten
data_pfad =r"C:\Users\User\Desktop\Abschlussprojekte Weiterbildung\CSV Datei\Prognose Politische Maßnahme nach Regioin.csv"
df = pd.read_csv(data_pfad)

# Unabhängige Variable (X) und Zielvariable (y)
X = df[["Jahr"]]
y = df.drop(columns=["Jahr"])  # Abhängige Variablen

# Modell für jede Region separat trainieren
prognosen = {}
jahre_zukunft = pd.DataFrame({"Jahr": [2025, 2030, 2035]})

for region in y.columns:
    model = LinearRegression()
    model.fit(X, y[region])  # Trainiere das Modell für jede Region separat
    prognosen[region] = model.predict(jahre_zukunft)  # Prognose für 2025, 2030, 2035

# Ergebnisse speichern und auf ganze Zahlen runden
prognose_df = pd.DataFrame(prognosen, index=jahre_zukunft["Jahr"]).round(0).astype(int)
print("Prognose für 2025, 2030, 2035:")
print(prognose_df)

# Visualisierung der Prognose
plt.figure(figsize=(10, 6))
for region in y.columns:
    plt.plot(df["Jahr"], df[region], marker="o", label=f"Echte Daten ({region})")  # Echte Werte
    plt.plot(jahre_zukunft["Jahr"], prognose_df[region], marker="x", linestyle="dashed", label=f"Prognose ({region})")  # Prognosen

plt.xlabel("Jahr")
plt.ylabel("Maßnahmen (%) Steuervergünstigung")
plt.title("Prognose der politischen Maßnahmen zur Förderung des Stahlrecyclings bis 2035 nach Region")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





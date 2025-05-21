#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# CSV-Datei laden
datei_pfad=r"C:\Users\User\Desktop\Abschlussprojekte Weiterbildung\CSV Datei\Prognose der Stahlproduktion mit grünem Wasserstoff.csv"
df=pd.read_csv(datei_pfad,sep=",") # Falls tab-getrennt, Trennerzeichen setzen

# Daten vorbereiten
X=df[["Jahr"]] #Unabhängige Variable(Jahre)
y=df.drop(columns=["Jahr"]) # Abhängige variablen (Maßnahmen)

#Trainings- und Tetdaten aufteilen
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Modell erstellen und trainieren
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#Prognose für zukünftige Jahre bis 2050
jahre_zukunft=pd.DataFrame({"Jahr":[2030,2040,2050]})
prognose=model.predict(jahre_zukunft)

# Ergebnisse anzeigen
print("Prognose für 2030,2040,2050:")
print(pd.DataFrame(prognose,columns=y.columns,index=jahre_zukunft["Jahr"]))

# Visualisierung der Prognose
plt.figure(figsize=(10,6))
for region in y.columns:
    plt.plot(df["Jahr"],df[region],marker="o",label=f"Echte Daten({region})")
    plt.plot(jahre_zukunft["Jahr"],prognose[:,y.columns.get_loc(region)],marker="x",linestyle="dashed",label=f"Prognose({region})")

plt.xlabel("Jahr")
plt.ylabel("Maßnahmen (% Steuervergünstigung)")
plt.title("Prognose der Stahlproduktion mit grünem Wasserstoff")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





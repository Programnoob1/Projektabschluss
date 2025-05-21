#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

data_pfad=r"C:\Users\User\Desktop\Abschlussprojekte Weiterbildung\CSV Datei\Prognose der Recyclingquote bis 2050 nach Region.csv"
df=pd.read_csv(data_pfad)

X=df[["Jahr"]]
y=df.drop(columns=["Jahr"])

jahre_zukunft=pd.DataFrame({"Jahr":[2023,2030,2040,2050]})
prognosen={}
for region in y.columns:
    model=make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
    model.fit(X,y[region])
    prognosen[region]=model.predict(jahre_zukunft)

prognose_df=pd.DataFrame(prognosen,index=jahre_zukunft["Jahr"]).round(0).astype(int)

print("Prognose für 2023,2030,2040,2050")
print(prognose_df)

plt.figure(figsize=(10,6))
for region in y.columns:
    plt.plot(df["Jahr"], df[region], marker="o", label=f"Echte Daten ({region})")
    plt.plot(jahre_zukunft["Jahr"], prognose_df[region], marker="x", linestyle="dashed", label=f"Prognose ({region})")
    
plt.xlabel("Jahr")
plt.ylabel("Maßnahmen (%)")
plt.title("Polynomiale Prognose der politischen Maßnahmen bis 2045 nach Region")
plt.legend(loc="best")
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt  # Corrección: era 'pit', debe ser 'plt'
from sklearn.cluster import KMeans  # ✅ Correcto

# 1. Importar el Dataset con los datos de entrenamiento
df_datos_clientes = pd.read_csv("clientes_entrenamiento.csv")
print(df_datos_clientes.info())

# 2. Convertir el Dataframe a un array de Numpy
X = df_datos_clientes.values
print(X)

# 3. Entrenar el modelo de KMeans
modelo = KMeans(n_clusters=2, random_state=1234, n_init=10)  # Corrección: kMeans -> KMeans
modelo.fit(X)

# 4. analisis de datos
df_datos_clientes['cluster'] = modelo.labels_

analisis = df_datos_clientes.groupby('cluster').mean()
print(analisis)

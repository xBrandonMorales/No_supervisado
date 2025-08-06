import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as pit
from sklearn.cluster import kMeans

# 1. Importar el Dataset con los datos de entrenamiento

df_data_clientes = pd.read_csv("clientes_entrenamiento.csv")

print(df_data_clientes.info())

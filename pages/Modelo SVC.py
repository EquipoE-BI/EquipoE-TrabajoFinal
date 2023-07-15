import yfinance as yf
import warnings
import streamlit as st

# Librería ML
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Librerías Manipulación de Datos
import pandas as pd
import numpy as np

# Librería de Visualización
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Ignorar alertas
warnings.filterwarnings("ignore")

# Título
st.set_page_config(page_title="SVC")

st.markdown("# Modelo Predictivo - SVC")
st.sidebar.header("Support Vector Classifier")
st.write(
    """En esta página podrás ver cómo funciona el modelo SVC en la predicción del mercado de valores"""
)

ticker = st.text_input('Etiqueta de cotización', 'EA')
st.write('La etiqueta de cotización actual es', ticker)

tic = yf.Ticker(ticker)
tic

hist = tic.history(period="max", auto_adjust=True)
hist

df = hist
df.info()

# Creamos variables predictivas
df["Open-Close"] = df.Open - df.Close
df["High-Low"] = df.High - df.Low

# Guardar todas las variables predictivas en X
X = df[["Open-Close", "High-Low"]]
X.head()

# Variables objetivas
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 1)

split_percentage = 0.8
split = int(split_percentage*len(df))

# Conjunto de entrenamiento
X_train = X[:split]
y_train = y[:split]

# Conjunto de prueba
X_test = X[split:]
y_test = y[split:]

# SVC
cls = SVC().fit(X_train, y_train)

df['Predicted_Signal'] = cls.predict(X)
# Calcular los retornos diarios
df["Return"] = df.Close.pct_change()
# Calcula retornos de estrategia
df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
# Calcular los retornos acumulativos
df["Cum_Ret"] = df["Return"].cumsum()
st.write("Dataframe con retornos acumulativos")
df
# Plot de los retornos acumulativos de la estrategia
df["Cum_Strategy"] = df["Strategy_Return"].cumsum()
st.write("Dataframe con retornos de estrategia acumulativos")
df


st.write("Gráfica de los retornos de la estrategia vs. los retornos originales")
# Plot de los retornos acumulativos de la estrategia
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Cum_Ret"], color="red", label="Retornos Originales")
ax.plot(df["Cum_Strategy"], color="blue", label="Retornos de Estrategia")
ax.set_title("Retornos Acumulativos de la Estrategia vs. Retornos Originales")
ax.set_xlabel("Fecha")
ax.set_ylabel("Retornos Acumulativos")
ax.legend()
st.pyplot(fig)

st.write("Conclusiones")
st.write(
    """Los retornos acumulativos originales (color rojo) muestran cómo hubieran evolucionado las inversiones si se hubiera mantenido una estrategia de "comprar y mantener" en el mercado sin utilizar ninguna estrategia de trading."""
)
st.write(
    """Los retornos acumulativos de la estrategia (color azul) muestran cómo evolucionan las inversiones utilizando el modelo de regresión para generar señales de compra y venta en el mercado."""
)
st.write(
    """Comparando ambos gráficos, se puede evaluar el desempeño de la estrategia en relación con los retornos originales del mercado. Si los retornos acumulativos de la estrategia superan a los retornos acumulativos originales, se considera que la estrategia ha sido exitosa en generar retornos adicionales. Por otro lado, si los retornos acumulativos de la estrategia son inferiores a los retornos originales, la estrategia puede considerarse menos efectiva en generar ganancias."""
)
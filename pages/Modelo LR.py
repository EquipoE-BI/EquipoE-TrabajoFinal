import yfinance as yf
import warnings
import streamlit as st

# Machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# To ignore warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Regression")

st.markdown("# Regression")
st.sidebar.header("Regression")
st.write(
    """En esta página podrás ver cómo funciona el modelo de regresión en la predicción del mercado de valores"""
)

ticker = st.text_input('Etiqueta de cotización', 'NFLX')
st.write('La etiqueta de cotización actual es', ticker)

tic = yf.Ticker(ticker)
tic

hist = tic.history(period="max", auto_adjust=True)
hist

df = hist
df.info()

# Create predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Save all predictor variables in X
X = df[['Open-Close', 'High-Low']]
X.head()

# Normalize the data
X = (X - X.mean()) / X.std()

# Target variable
y = df['Close'].shift(-1)

split_percentage = 0.8
split = int(split_percentage * len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

df['Predicted_Close'] = np.concatenate([train_pred, test_pred])
df['Predicted_Signal'] = np.where(df['Predicted_Close'].shift(-1) > df['Close'], 1, -1)

# Calculate daily returns
df['Return'] = df.Close.pct_change()
# Calculate strategy returns
df['Strategy_Return'] = df.Return * df.Predicted_Signal
# Calculate cumulative returns
df['Cum_Ret'] = df['Return'].cumsum()
st.write("Dataframe con retornos acumulativos")
df
# Plot cumulative strategy returns
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
st.write("Dataframe con retornos de estrategia acumulativos")
df

st.write("Plot Strategy Returns vs Original Returns")
fig = plt.figure()
plt.plot(df['Cum_Ret'], color='red')
plt.plot(df['Cum_Strategy'], color='blue')
st.pyplot(fig)

st.write("Haz llegado hasta el final de esta sección. Gracias")

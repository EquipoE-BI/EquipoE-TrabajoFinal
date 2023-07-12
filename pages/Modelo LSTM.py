import yfinance as yf
import warnings
import streamlit as st

# Machine learning
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# For LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# To ignore warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="LSTM")

st.markdown("# LSTM")
st.sidebar.header("LSTM")
st.write(
    """En esta página podrás ver cómo funciona el modelo LSTM en la predicción del mercado de valores"""
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

# Variables objetivas
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage * len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

# Reshape the data for LSTM input
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# LSTM model
model = Sequential()
model.add(LSTM(units=32, input_shape=(1, X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

df['Predicted_Signal'] = np.concatenate([train_pred.flatten(), test_pred.flatten()])
# Calculate daily returns
df['Return'] = df.Close.pct_change()
# Calculate strategy returns
df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
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

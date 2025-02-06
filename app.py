import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the trained model
model = load_model('C:/Users/varni/OneDrive/Desktop/bitcoin/Bitcoin_Price_prediction_Model.keras')

# Streamlit App Header
st.header('Bitcoin Price Prediction Model')

# Fetch Bitcoin Data
st.subheader('Bitcoin Price Data')
data = pd.DataFrame(yf.download('BTC-USD', '2015-01-01', '2024-04-08'))

# Flatten MultiIndex columns (if any)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Reset index to have a proper Date column
data = data.reset_index()
st.write(data)

# Plot Bitcoin Price Line Chart
st.subheader('Bitcoin Line Chart')
st.line_chart(data[['Date', 'Close']].set_index('Date'))

# Preprocessing the Data
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Volume'], inplace=True, errors='ignore')

train_data = data[:-100]
test_data = data[-200:]

# Scaling the Data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Preparing Test Data
base_days = 100
x_test, y_test = [], []
for i in range(base_days, test_data_scaled.shape[0]):
    x_test.append(test_data_scaled[i - base_days:i])
    y_test.append(test_data_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predicting the Prices
st.subheader('Predicted vs Original Prices')
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

# Convert to DataFrame for Display
pred_df = pd.DataFrame(pred, columns=['Predicted Price'])
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
actual_df = pd.DataFrame(y_actual, columns=['Original Price'])
chart_data = pd.concat((pred_df, actual_df), axis=1)

st.write(chart_data)

# Plot Predicted vs Original Prices
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(chart_data)

# Future Predictions
st.subheader('Predicted Future Bitcoin Prices')

m = y_test
future_predictions = []
future_days = 5

for _ in range(future_days):
    m = m.reshape(-1, 1)
    latest_data = m[-base_days:].reshape(1, base_days, 1)
    pred = model.predict(latest_data)
    m = np.append(m, pred)
    future_predictions.append(pred)

# Reshape and Inverse Transform Future Predictions
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Plot Future Predictions
st.line_chart(pd.DataFrame(future_predictions, columns=['Future Predicted Price']))

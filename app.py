import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st



import yfinance as yf

start = '2010-01-01'
end = '2019-12-31'


st.title('Stock Trend Prediction')

user_input  = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)


st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('closing price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)





split_percentage = 0.7

split_index = int(len(df) * split_percentage)
train_data = df.iloc[:split_index][['Close']]
test_data = df.iloc[split_index:][['Close']]





from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler(feature_range = (0,1))

training_array = Scaler.fit_transform(train_data)




model = load_model('model.h5')

daysOf_100_Data = train_data.tail(100)
final_dataFrame = daysOf_100_Data.append(test_data, ignore_index = True)
input_data = Scaler.fit_transform(final_dataFrame)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

    
x_test,y_test = np.array(x_test), np.array(y_test)

y_predict = model.predict(x_test)

_scaler = Scaler.scale_

factor = 1/ _scaler[0]
y_predict = y_predict * factor
y_test = y_test * factor


st.subheader('Predictions vs original')
plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'original_price')
plt.plot(y_predict, 'r', label = 'Predicted_Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)
  

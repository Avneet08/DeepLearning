# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 08:20:11 2018

@author: Avneet
"""

# part 1 data preprocessing

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing train set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = dataset_train.iloc[:, 1:2].values

# feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
train_set_scale=sc.fit_transform(train_set)

# building a specific datastructure with 60 timesteps and 1 output
# X-train will contain 60 prev days
# Y -train will have the info for next day
 X_train=[]
 y_train=[]
 for i in range(60,1258):
     X_train.append(train_set_scale[i-60:i,0])
     y_train.append(train_set_scale[i,0])
     
X_train,y_train=np.array(X_train),np.array(y_train)

# Reshaping the data 
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


#part2 building rnn

# importing the keras libraries
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense

# initialization the rnn
regressor = Sequential()
# adding first lstm layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# adding a second layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# third layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# 4th layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units=1))


# compiling the rnn

regressor.compile(optimizer='adam', loss = 'mean_squared_error')

# fitting the rnn int the train_set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# part3 predictions and analysis of results

# getting the real stoock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


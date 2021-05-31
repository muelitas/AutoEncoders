'''/**************************************************************************
    File: from_tutorial.py
    Author: Mario Esparza
    Created on: 05/22/2021
    Last edit on: 05/22/2021
    
    Strictly following this tutorial:
    https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352

    I tried the following:
    
    300 epochs, result is MSE-average of 5 different runs:
        LSTM, LayerNorm, relu = 6.553951015824205e-06
        LSTM, LayerNorm, gelu = 0.00020415089871280524
        LSTM, LayerNorm, tanh = 2.5725508325840007e-05
        LSTM, No-LayerNorm, relu = 0.00013076274216886906
        
 (best) GRU, LayerNorm, relu = 3.809240525075671e-06
        GRU, LayerNorm, gelu = 3.47865120702526e-05
        GRU, LayerNorm, tanh = 1.3336048079030414e-05
        GRU, No-LayerNorm, relu = 5.362894689740379e-05
        
***************************************************************************'''
# lstm autoencoder to recreate a timeseries
import gc

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras import Input, Model
from utils import AutoEncoder

'''
A UDF to convert input data into 3-D
array as required for LSTM network.
'''
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

# define input timeseries
timeseries = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       [0.1**3, 0.2**3, 0.3**3, 0.4**3, 0.5**3, 0.6**3, 0.7**3, 0.8**3, 0.9**3]]).transpose()

timesteps = timeseries.shape[0]
n_features = timeseries.shape[1]

timesteps = 3
X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)

n_features = 2
X = np.array(X)
X = X.reshape(X.shape[0], timesteps, n_features)

MSE = 0
for i in range(0, 5):
    #Initialize model
    inpuT= Input(shape = (timesteps, n_features), batch_size=5)
    HP = {'dim1': timesteps, 'dim2': n_features}
    autoencoder = Model(inpuT, AutoEncoder(inpuT, HP))
    autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.summary()
    
    # fit model
    autoencoder.fit(X, X, epochs=300, batch_size=5, verbose=0)
    # demonstrate reconstruction
    yhat = autoencoder.predict(X, verbose=0)
    # print('---Predicted---')
    # print(np.round(yhat,3))
    # print('---Actual---')
    # print(np.round(X, 3))
    
    #Calculate mse
    mse = ((X - yhat)**2).mean(axis=None)
    MSE += mse
    print(f"MSE between X and yhat is: {mse}")
    
    #Release global state and collect garbage (clear gpu memory)
    del(autoencoder)
    tf.keras.backend.clear_session()
    # print(f"first value of gc collect: {gc.collect()}")
    # print(f"second value of gc collect: {gc.collect()}")
    
print(f"Average MSE: {MSE/5}")
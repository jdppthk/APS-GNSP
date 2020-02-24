import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras import Sequential
from keras.layers import Dense, Lambda
from keras.regularizers import Regularizer
from keras import regularizers

def keras_regressor(X,y, N, lam, num_epochs, bsize, learn_rate):
    model = Sequential()
    model.add(Dense(1,input_dim = X.shape[1],use_bias=False, kernel_regularizer =regularizers.l2(lam)))
    model.compile(loss = 'mean_squared_error',optimizer = keras.optimizers.adam(lr=learn_rate))
    history = model.fit(x=X,y=y,epochs=num_epochs,batch_size = bsize, verbose=2)
    p = model.get_weights()[0]
    return(history, p)





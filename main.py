# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from glob import glob
from scipy.io import loadmat
import math
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from glob import glob
from scipy.io import loadmat
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
print(tf.__version__)

dataset_internet = pd.read_csv('Milano1Day.csv', header=None)
dataset_internet=dataset_internet.values

# Parameter initialization
r=5 # size of spatial data
recurrent_length = 6*6 # 12 10 minutes interval
train_length = 6*14
test_length = 6*8

# creating 3D train data
data_spatial = np.zeros((100,100,24*6))
data_spatial_temp = np.zeros((11,11))
for t in range(24*6):
    data_temp = dataset_internet[dataset_internet[:,1]==t]
    for i in range(100):
        for j in range(100):
            temp = data_temp[data_temp[:,0]==100*i+j+1,2]
            if temp.shape[0]!=0:
                data_spatial[i,j,t] = temp
            else:
                data_spatial[i,j,t] = data_spatial[i,j,t-1]
                
                
# Create y train data
for i in range(r,20):
    for j in range(r,20):
        for t in range(recurrent_length,train_length):
          if t==recurrent_length:
            y_t_ij=data_spatial[i,j,t].reshape(1,1)
          else:
            y_t_ij = np.concatenate((y_t_ij,data_spatial[i,j,t].reshape(1,1)),axis=0)
        if i==r and j==r:
          y_train = y_t_ij
        else:
          y_train = np.concatenate((y_train,y_t_ij), axis =0)
          
          
# create train data
for i in range(r,20):
    for j in range(r,20):
        for t in range(train_length):
            data_temp = data_spatial[i-r:i+r+1,j-r:j+r+1,t].reshape(2*r+1,2*r+1,1)
            if (t==0 and (i==r and j==r)):
                data_t=data_temp
            else:
                data_t = np.concatenate((data_t,data_temp),axis=2)
        for t in range(recurrent_length,train_length):
          if t==recurrent_length:
            data_t_ij=data_t[:,:,t-recurrent_length:t].reshape(1,2*r+1,2*r+1,recurrent_length)
          else:
            data_t_ij = np.concatenate((data_t_ij,data_t[:,:,t-recurrent_length:t].reshape(1,2*r+1,2*r+1,recurrent_length)),axis=0)
        if i==r and j==r:
          data_final = data_t_ij
        else:
          data_final = np.concatenate((data_final,data_t_ij), axis =0)
          
X_train = data_final

# Define the model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(2*r+1, 2*r+1, recurrent_length)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (1,1), activation='relu'),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron
    tf.keras.layers.Dense(1)
])

# Train the model
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(learning_rate=1e-4))

history = model.fit(
      X_train,y_train,
      steps_per_epoch=8,  
      epochs=100,
      verbose=1)
      
# Plot the loss
loss = history.history['loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.show()

# Prediction of the first 20 timesteps
X_pred = model.predict(X_train[0:20,:,:,:])
y_pred = y_train[0:20]

# plotting the predicted result vs actual data
plt.figure()
plt.plot(np.arange(20), X_pred, 'r', label='Training Loss')
plt.plot(np.arange(20), y_pred, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

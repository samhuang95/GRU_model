import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
import tensorflow as tf 
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,GRU
from sklearn.metrics import mean_absolute_error,mean_squared_error

# MSE Calculation formula
def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
    
# preprocessing function
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, single_step=False):
  data = []
  labels = []
  
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size
  
  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    data.append(dataset[indices])
    
    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])
  
  return np.array(data), np.array(labels)

# model program
def gru_model(input_length, input_dim):

    d=0.3
    model= Sequential()
    model.add(GRU(64,input_shape=(input_length, input_dim),return_sequences=True))
    model.add(Dropout(d))

    model.add(GRU(32,input_shape=(input_length, input_dim),return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(1,activation='linear',kernel_initializer="uniform")) #linear / softmax / sigmoid

    # optimizer = tf.keras.optimizers.Adam(lr=0.00005)
    model.compile(loss='mse',optimizer='adam') #loss=mse/categorical_crossentropy
    return model 

def gru_model1(input_length, input_dim):

    d=0.3
    model= Sequential()
    model.add(GRU(128,input_shape=(input_length, input_dim),return_sequences=True))
    model.add(Dropout(d))

    model.add(GRU(64,input_shape=(input_length, input_dim),return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(1,activation='linear',kernel_initializer="uniform"))

    # optimizer = tf.keras.optimizers.Adam(lr=0.00005)
    model.compile(loss='mse',optimizer='adam')#loss=mse/categorical_crossentropy
    return model 

# =================================================================
# 【Splite and Scaling data】
df = pd.read_csv('./train_data.csv')
y =df["total_sales"]
x =df.drop("total_sales", axis=1)

scaler=MinMaxScaler(feature_range=(0,1))
y=scaler.fit_transform(y.to_frame())

scaler1=MinMaxScaler(feature_range=(0,1))
x=scaler1.fit_transform(x)

# splite the part if train, test and vaild data
x,y=multivariate_data( x ,y , 0 ,None, 10 , 1 ,single_step=True)
split =0.9
# This parameter will affect the length of future time.
x_,y_  = x[0:int(split*len(x))] , y[0:int(split*len(x))]

x_test ,y_test   = x[int(split*len(x)):] , y[int(split*len(x)):]
split= 0.8
x_train,y_train  =x_[:int(split*len(x_))] , y_[:int(split*len(x_))]
x_vaild,y_vaild  =x_[int(split*len(x_)):] , y_[int(split*len(x_)):]

trainindex= df.index[10:len(x_train)+10]
valindex = df.index[len(x_train)+10:len(x_train)+10+len(x_vaild)]

# =================================================================
# 【Basic program and 】
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=300, monitor = 'val_loss')
    ]
filepath="lstm.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min',save_best_only=True)

call_backlist = [ my_callbacks,checkpoint]
callbacks=call_backlist
gru1 = gru_model1(10,6)
# Need to adjust the dimension, my data is 6 features

historygru1 = gru1.fit( x_train, y_train, batch_size=30,shuffle=False , epochs=1000,validation_data=(x_vaild,y_vaild),callbacks=call_backlist)
gru1.summary()

gru1train  = gru1.predict(x_train)
gru1val = gru1.predict(x_vaild)
gru1pre = gru1.predict(x_test)
plt.plot(historygru1.history['loss'], alpha=0.7)
plt.plot(historygru1.history['val_loss'], alpha=0.7)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

pre = gru1.predict(x_train)
pre1=gru1.predict(x_vaild)
fc=np.concatenate((pre,pre1))
yreal=np.concatenate((y_train,y_vaild))
plt.figure(facecolor='white')
pd.Series(fc.reshape(-1)).plot(color='blue', label='Predict1', alpha=0.7)
pd.Series(yreal.reshape(-1)).plot(color='red', label='Original', alpha=0.7)
plt.legend()
plt.show()

lstm1pre = gru1.predict(x_test)
pre= scaler.inverse_transform(lstm1pre)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))
print(pre)
plt.figure()
plt.plot(pre, alpha=0.7)
plt.plot(y_test, alpha=0.7)
plt.title('sales predicated')
plt.ylabel('sales')
plt.xlabel('day')
plt.legend(['pre', 'Test'], loc='upper left')
plt.show()

print(root_mean_squared_error(pre,y_test))

plt.figure()
plt.plot(pre)
plt.title('sales predicated')
plt.ylabel('sales')
plt.xlabel('day')

plt.show()

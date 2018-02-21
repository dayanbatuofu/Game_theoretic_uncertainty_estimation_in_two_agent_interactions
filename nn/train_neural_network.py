from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
import csv

# Read file
with open('train_data_50000.txt') as f:
    reader = csv.reader(f, delimiter="\t")
    d = array(list(reader))


# Input initialization
inp_dim = 6
out_dim = 40

data_size = 50000
validation_ind = 40000
test_ind = 45000

# Import the data
train_x = array(d[0:validation_ind, 0:inp_dim])
train_y = array(d[0:validation_ind, inp_dim:inp_dim+out_dim])

val_x = array(d[validation_ind:test_ind, 0:inp_dim])
val_y = array(d[validation_ind:test_ind, inp_dim:inp_dim+out_dim])

test_x = array(d[test_ind:data_size, 0:inp_dim])
test_y = array(d[test_ind:data_size, inp_dim:inp_dim+out_dim])

train_x = train_x.astype('float32')
val_x   = val_x.astype('float32')
test_x  = test_x.astype('float32')

train_y = train_y.astype('float32')
val_y   = val_y.astype('float32')
test_y  = test_y.astype('float32')

# Network Model
model = Sequential()

model.add(Dense(50, input_shape=(inp_dim,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))  # it's optional
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))  # it's optional
model.add(Dense(out_dim))
model.add(Activation('sigmoid'))  # you may change the activation here based on your need


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='min')
model.fit(train_x, train_y, batch_size=validation_ind, epochs=100, verbose=1, validation_data=(val_x, val_y),
          callbacks=[early_stop], initial_epoch=10)  # please read the tutorial for the guide of parameter tuning

model.save('action_prediction_model.h5')  # save the model

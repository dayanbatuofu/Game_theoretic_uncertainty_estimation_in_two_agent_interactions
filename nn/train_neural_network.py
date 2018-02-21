from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
import csv

#################################################
# try to scale input data
s_other_y          = [0, 1]
s_self_x           = [-2, 2]
s_self_y           = [0, 1]
s_desired_other_x  = [-2, 2]
s_desired_other_y  = [0, 1]
c_other            = [20, 100]
#################################################

# Read file
with open('train_data_10000.txt') as f:
    reader = csv.reader(f, delimiter="\t")
    d = array(list(reader))


# Input initialization
inp_dim = 6
out_dim = 40

data_size = 10000
validation_ind = 8000
test_ind = 9000

# Import the data
train_x = d[0:validation_ind, 0:inp_dim].astype('float32')
train_x[:,0] = (train_x[:,0] - s_other_y[0])/(s_other_y[1]-s_other_y[0])
train_x[:,1] = (train_x[:,1] - s_self_x[0])/(s_self_x[1]-s_self_x[0])
train_x[:,2] = (train_x[:,2] - s_self_y[0])/(s_self_y[1]-s_self_y[0])
train_x[:,3] = (train_x[:,3] - s_desired_other_x[0])/(s_desired_other_x[1]-s_desired_other_x[0])
train_x[:,4] = (train_x[:,4] - s_desired_other_y[0])/(s_desired_other_y[1]-s_desired_other_y[0])
train_x[:,5] = (train_x[:,5] - c_other[0])/(c_other[1]-c_other[0])
train_x = array(train_x)
train_y = array(d[0:validation_ind, inp_dim:inp_dim+out_dim])

val_x = d[validation_ind:test_ind, 0:inp_dim].astype('float32')
val_x[:,0] = (val_x[:,0] - s_other_y[0])/(s_other_y[1]-s_other_y[0])
val_x[:,1] = (val_x[:,1] - s_self_x[0])/(s_self_x[1]-s_self_x[0])
val_x[:,2] = (val_x[:,2] - s_self_y[0])/(s_self_y[1]-s_self_y[0])
val_x[:,3] = (val_x[:,3] - s_desired_other_x[0])/(s_desired_other_x[1]-s_desired_other_x[0])
val_x[:,4] = (val_x[:,4] - s_desired_other_y[0])/(s_desired_other_y[1]-s_desired_other_y[0])
val_x[:,5] = (val_x[:,5] - c_other[0])/(c_other[1]-c_other[0])
val_x = array(val_x)
val_y = array(d[validation_ind:test_ind, inp_dim:inp_dim+out_dim])

test_x = d[test_ind:data_size, 0:inp_dim].astype('float32')
test_x[:,0] = (test_x[:,0] - s_other_y[0])/(s_other_y[1]-s_other_y[0])
test_x[:,1] = (test_x[:,1] - s_self_x[0])/(s_self_x[1]-s_self_x[0])
test_x[:,2] = (test_x[:,2] - s_self_y[0])/(s_self_y[1]-s_self_y[0])
test_x[:,3] = (test_x[:,3] - s_desired_other_x[0])/(s_desired_other_x[1]-s_desired_other_x[0])
test_x[:,4] = (test_x[:,4] - s_desired_other_y[0])/(s_desired_other_y[1]-s_desired_other_y[0])
test_x[:,5] = (test_x[:,5] - c_other[0])/(c_other[1]-c_other[0])
test_x = array(test_x)
test_y = array(d[test_ind:data_size, inp_dim:inp_dim+out_dim])

train_y = train_y.astype('float32')
val_y   = val_y.astype('float32')
test_y  = test_y.astype('float32')

# Network Model
model = Sequential()

model.add(Dense(16, input_shape=(inp_dim,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))  # it's optional
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))  # it's optional
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))  # it's optional
model.add(Dense(out_dim))
model.add(Activation('sigmoid'))  # you may change the activation here based on your need


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='min')
model.fit(train_x, train_y, batch_size=validation_ind, epochs=5000, verbose=1, validation_data=(val_x, val_y),
          callbacks=[], initial_epoch=10)  # please read the tutorial for the guide of parameter tuning

model.save('action_prediction_model.h5')  # save the model

out = model.predict(train_x)

a = 1

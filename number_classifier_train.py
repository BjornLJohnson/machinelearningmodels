#!/usr/bin/env python
# coding: utf-8

import h5py
import cv2
import numpy as np
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

train_dataset = h5py.File('/home/bjorn/dev/data/train_signs.h5', "r")
print(str(list(train_dataset.keys())))
train_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_y = backend.one_hot(np.array(train_dataset["train_set_y"][:]), 6) # your train set labels

test_dataset = h5py.File('/home/bjorn/dev/data/test_signs.h5', "r")
test_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_y = backend.one_hot(np.array(test_dataset["test_set_y"][:]), 6) # your test set labels


print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

im_bgr = cv2.cvtColor(test_x_orig[12, :, :, :], cv2.COLOR_RGB2BGR)
cv2.imwrite('/home/bjorn/images/test_number.jpg', im_bgr)

# Standardize data to have feature values between 0 and 1.
train_x = train_x_orig/255
test_x = test_x_orig/255

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

model_name = 'number_classifier'

model = Sequential()
model.add(Conv2D(8, kernel_size=4, strides=1, padding='same', 
            input_shape=(64, 64, 3), data_format='channels_last',
            use_bias=True, activation='relu'))
model.add(MaxPool2D(pool_size=8, strides=8, padding='same', data_format='channels_last'))
model.add(Conv2D(16, kernel_size=2, strides=1, padding='same', use_bias=True, activation='relu', data_format='channels_last'))
model.add(MaxPool2D(pool_size=4, strides=4, padding='same', data_format='channels_last'))
model.add(Flatten(data_format='channels_last'))
model.add(Dense(6, use_bias=True, activation=None))

model.compile(loss='mean_squared_error',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=20, steps_per_epoch=64, use_multiprocessing=True)

# Save the weights
model.save_weights(model_name + '_weights.h5')

# Save the model architecture
with open(model_name + '_arch.json', 'w') as f:
    f.write(model.to_json())

# train_score = model.evaluate(train_x, train_y)
# test_score = model.evaluate(test_x, test_y)
# print("Train Score " + str(train_score))
# print("Test Score " + str(test_score))
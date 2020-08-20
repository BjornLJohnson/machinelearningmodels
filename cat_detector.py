#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

train_dataset = h5py.File('/home/bjorn/dev/data/train_catvnoncat.h5', "r")
train_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_y = np.array(train_dataset["train_set_y"][:]) # your train set labels

test_dataset = h5py.File('/home/bjorn/dev/data/test_catvnoncat.h5', "r")
test_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

# train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
# test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = np.transpose(train_x_flatten/255)
test_x = np.transpose(test_x_flatten/255)

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


model = Sequential()
model.add(Dense(20, input_dim=num_px*num_px*3, use_bias=True, activation='relu'))
model.add(Dense(7, use_bias=True, activation='relu'))
model.add(Dense(5, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=True, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          epochs=1000)

train_score = model.evaluate(train_x, train_y)
test_score = model.evaluate(test_x, test_y)
print("Train Score " + str(train_score))
print("Test Score " + str(test_score))
# def show_image(index, x, y, classes):
#     # The following code will show you an image in the dataset. Feel free to change the index and re-run the cell multiple times to see other images. 

#     # Example of a picture
#     plt.imshow(x[index])
#     plt.show(block=False)
#     print ("y = " + str(y[0,index]) + ". It's a " + classes[y[0,index]].decode("utf-8") +  " picture.")
# # Show the 50th image
# show_image(50, train_x_orig, train_y, classes)


# < The model can be summarized as: ***INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT***
# 
# 
# - The input is a (64,64,3) image which is flattened to a vector of size $(12288,1)$. 
# - The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ of size $(n^{[1]}, 12288)$.
# - You then add a bias term and take its relu to get the following vector: $[a_0^{[1]}, a_1^{[1]},..., a_{n^{[1]}-1}^{[1]}]^T$.
# - You then repeat the same process.
# - You multiply the resulting vector by $W^{[2]}$ and add your intercept (bias). 
# - Finally, you take the sigmoid of the result. If it is greater than 0.5, you classify it to be a cat.
#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks: Application
# 
# Welcome to Course 4's second assignment! In this notebook, you will:
# 
# - Implement helper functions that you will use when implementing a TensorFlow model
# - Implement a fully functioning ConvNet using TensorFlow 
# 
# **After this assignment you will be able to:**
# 
# - Build and train a ConvNet in TensorFlow for a classification problem 
# 
# We assume here that you are already familiar with TensorFlow. If you are not, please refer the *TensorFlow Tutorial* of the third week of Course 2 ("*Improving deep neural networks*").

# ## 1.0 - TensorFlow model
# 
# In the previous assignment, you built helper functions using numpy to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. 
# 
# As usual, we will start by loading in the packages. 

# In[33]:

import math
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from conv_net_utils import *

#get_ipython().magic('matplotlib inline')
np.random.seed(1)


# Run the next cell to load the "SIGNS" dataset you are going to use.

# In[34]:

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


# As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.
# 
# <img src="images/SIGNS.png" style="width:800px;height:300px;">
# 
# The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 

# In[35]:

# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# In Course 2, you had built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.
# 
# To get started, let's examine the shapes of your data. 

# In[36]:

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

X, Y = create_placeholders(64, 64, 3, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))


# **Expected Output**
# 
# <table> 
# <tr>
# <td>
#     X = Tensor("Placeholder:0", shape=(?, 64, 64, 3), dtype=float32)
# 
# </td>
# </tr>
# <tr>
# <td>
#     Y = Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)
# 
# </td>
# </tr>
# </table>


tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))


# ** Expected Output:**
# 
# <table> 
# 
#     <tr>
#         <td>
#         W1 = 
#         </td>
#         <td>
# [ 0.00131723  0.14176141 -0.04434952  0.09197326  0.14984085 -0.03514394 <br>
#  -0.06847463  0.05245192]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#         W2 = 
#         </td>
#         <td>
# [-0.08566415  0.17750949  0.11974221  0.16773748 -0.0830943  -0.08058 <br>
#  -0.00577033 -0.14643836  0.24162132 -0.05857408 -0.19055021  0.1345228 <br>
#  -0.22779644 -0.1601823  -0.16117483 -0.10286498]
#         </td>
#     </tr>
# 
# </table>

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = " + str(a))


# **Expected Output**:
# 
# <table> 
#     <td> 
#     Z3 =
#     </td>
#     <td>
#     [[-0.44670227 -1.57208765 -1.53049231 -2.31013036 -1.29104376  0.46852064] <br>
#  [-0.17601591 -1.57972014 -1.4737016  -2.61672091 -1.00810647  0.5747785 ]]
#     </td>
# </table>


tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
    print("cost = " + str(a))


# **Expected Output**: 
# 
# <table>
#     <td> 
#     cost =
#     </td> 
#     
#     <td> 
#     2.91034
#     </td> 
# </table>

# ## 1.4 Model 
# 
# Finally you will merge the helper functions you implemented above to build a model. You will train it on the SIGNS dataset. 
# 
# You have implemented `random_mini_batches()` in the Optimization programming assignment of course 2. Remember that this function returns a list of mini-batches. 
# 
# **Exercise**: Complete the function below. 
# 
# The model below should:
# 
# - create placeholders
# - initialize parameters
# - forward propagate
# - compute the cost
# - create an optimizer
# 
# Finally you will create a session and run a for loop  for num_epochs, get the mini-batches, and then for each mini-batch you will
# optimize the function. [Hint for initializing the variables](https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer)

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters


# Run the following cell to train your model for 100 epochs. Check if your cost after epoch 0 and 5 matches our output.
# If not, stop the cell and go back to your code!

_, _, parameters = model(X_train, Y_train, X_test, Y_test)


# **Expected output**: although it may not match perfectly, your expected output should be close to ours and your cost value should decrease.
# 
# <table> 
# <tr>
#     <td> 
#     **Cost after epoch 0 =**
#     </td>
# 
#     <td> 
#       1.917929
#     </td> 
# </tr>
# <tr>
#     <td> 
#     **Cost after epoch 5 =**
#     </td>
# 
#     <td> 
#       1.506757
#     </td> 
# </tr>
# <tr>
#     <td> 
#     **Train Accuracy   =**
#     </td>
# 
#     <td> 
#       0.940741
#     </td> 
# </tr> 
# 
# <tr>
#     <td> 
#     **Test Accuracy   =**
#     </td>
# 
#     <td> 
#       0.783333
#     </td> 
# </tr> 
# </table>

# Congratulations! You have finised the assignment and built a model that recognizes SIGN language with almost
# 80% accuracy on the test set. If you wish, feel free to play around with this dataset further. You can actually
# improve its accuracy by spending more time tuning the hyperparameters, or using regularization (as this model clearly has a high variance). 
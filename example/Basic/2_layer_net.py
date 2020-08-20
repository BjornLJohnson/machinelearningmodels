#!/usr/bin/env python
# coding: utf-8

# # Deep Neural Network for Image Classification: Application
# 
# When you finish this, you will have finished the last programming assignment of Week 4, and also the last programming assignment of this course! 
# 
# You will use use the functions you'd implemented in the previous assignment to build a deep network, and apply it to cat vs non-cat classification. Hopefully, you will see an improvement in accuracy relative to your previous logistic regression implementation.  
# 
# **After this assignment you will be able to:**
# - Build and apply a deep neural network to supervised learning. 
# 
# Let's get started!

# ## 1 - Packages

# Let's first import all the packages that you will need during this assignment. 
# - [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
# - [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.
# - dnn_app_utils provides the functions implemented in the "Building your Deep Neural Network: Step by Step" assignment to this notebook.
# - np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work.

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from neural_net_utils import *

#get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')

np.random.seed(1)

# ## 2 - Dataset
# 
# You will use the same "Cat vs non-Cat" dataset as in "Logistic Regression as a Neural Network" (Assignment 2). The model you had built had 70% test accuracy on classifying cats vs non-cats images. Hopefully, your new model will perform a better!
# 
# **Problem Statement**: You are given a dataset ("data.h5") containing:
#     - a training set of m_train images labelled as cat (1) or non-cat (0)
#     - a test set of m_test images labelled as cat and non-cat
#     - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
# 
# Let's get more familiar with the dataset. Load the data by running the cell below.

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# Show the 50th image
show_image(50, train_x_orig, train_y, classes)

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


# As usual, you reshape and standardize the images before feeding them to the network. The code is given in the cell below.
# 
# <img src="images/imvectorkiank.png" style="width:450px;height:300px;">
# 
# <caption><center> <u>Figure 1</u>: Image to vector conversion. <br> </center></caption>

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# $12,288$ equals $64 \times 64 \times 3$ which is the size of one reshaped image vector.

# ## 3 - Architecture of your model

# Now that you are familiar with the dataset, it is time to build a deep neural network to distinguish cat images from non-cat images.
# 
# You will build two different models:
# - A 2-layer neural network
# - An L-layer deep neural network
# 
# You will then compare the performance of these models, and also try out different values for $L$. 
# 
# Let's look at the two architectures.
# 
# ### 3.1 - 2-layer neural network
# 
# <img src="images/2layerNN_kiank.png" style="width:650px;height:400px;">
# <caption><center> <u>Figure 2</u>: 2-layer neural network. <br> The model can be summarized as: ***INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT***. </center></caption>
# 
# <u>Detailed Architecture of figure 2</u>:
# - The input is a (64,64,3) image which is flattened to a vector of size $(12288,1)$. 
# - The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ of size $(n^{[1]}, 12288)$.
# - You then add a bias term and take its relu to get the following vector: $[a_0^{[1]}, a_1^{[1]},..., a_{n^{[1]}-1}^{[1]}]^T$.
# - You then repeat the same process.
# - You multiply the resulting vector by $W^{[2]}$ and add your intercept (bias). 
# - Finally, you take the sigmoid of the result. If it is greater than 0.5, you classify it to be a cat.
# 
# ### 3.2 - L-layer deep neural network
# 
# It is hard to represent an L-layer deep neural network with the above representation. However, here is a simplified network representation:
# 
# <img src="images/LlayerNN_kiank.png" style="width:650px;height:400px;">
# <caption><center> <u>Figure 3</u>: L-layer neural network. <br> The model can be summarized as: ***[LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID***</center></caption>
# 
# <u>Detailed Architecture of figure 3</u>:
# - The input is a (64,64,3) image which is flattened to a vector of size (12288,1).
# - The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ and then you add the intercept $b^{[1]}$. The result is called the linear unit.
# - Next, you take the relu of the linear unit. This process could be repeated several times for each $(W^{[l]}, b^{[l]})$ depending on the model architecture.
# - Finally, you take the sigmoid of the final linear unit. If it is greater than 0.5, you classify it to be a cat.
# 
# ### 3.3 - General methodology
# 
# As usual you will follow the Deep Learning methodology to build the model:
#     1. Initialize parameters / Define hyperparameters
#     2. Loop for num_iterations:
#         a. Forward propagation
#         b. Compute cost function
#         c. Backward propagation
#         d. Update parameters (using parameters, and grads from backprop) 
#     4. Use trained parameters to predict labels
# 
# Let's now implement those two models!

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

# Run the cell below to train your parameters. See if your model runs. The cost should be decreasing. 
# It may take up to 5 minutes to run 2500 iterations. Check if the "Cost after iteration 0" matches the expected output below

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


# **Expected Output**:
# <table> 
#     <tr>
#         <td> **Cost after iteration 0**</td>
#         <td> 0.6930497356599888 </td>
#     </tr>
#     <tr>
#         <td> **Cost after iteration 100**</td>
#         <td> 0.6464320953428849 </td>
#     </tr>
#     <tr>
#         <td> **...**</td>
#         <td> ... </td>
#     </tr>
#     <tr>
#         <td> **Cost after iteration 2400**</td>
#         <td> 0.048554785628770206 </td>
#     </tr>
# </table>

# Good thing you built a vectorized implementation! Otherwise it might have taken 10 times longer to train this.
# 
# Now, you can use the trained parameters to classify images from the dataset. To see your predictions on the training and test sets, run the cell below.

predictions_train = predict(train_x, train_y, parameters)


# **Expected Output**:
# <table> 
#     <tr>
#         <td> **Accuracy**</td>
#         <td> 1.0 </td>
#     </tr>
# </table>


predictions_test = predict(test_x, test_y, parameters)


# **Expected Output**:
# 
# <table> 
#     <tr>
#         <td> **Accuracy**</td>
#         <td> 0.72 </td>
#     </tr>
# </table>

# **Note**: You may notice that running the model on fewer iterations (say 1500) gives better accuracy on the test set. This is called "early stopping" and we will talk about it in the next course. Early stopping is a way to prevent overfitting. 
# 
# Congratulations! It seems that your 2-layer neural network has better performance (72%) than the logistic regression implementation (70%, assignment week 2). Let's see if you can do even better with an $L$-layer model.


# ##  6) Results Analysis
# 
# First, let's take a look at some images the L-layer model labeled incorrectly. This will show a few mislabeled images. 

print_mislabeled_images(classes, test_x, test_y, pred_test)


# **A few types of images the model tends to do poorly on include:** 
# - Cat body in an unusual position
# - Cat appears against a background of a similar color
# - Unusual cat color and species
# - Camera Angle
# - Brightness of the picture
# - Scale variation (cat is very large or small in image) 

# ## 7) Test with your own image (optional/ungraded exercise) ##
# 
# Congratulations on finishing this assignment. You can use your own image and see the output of your model. To do that:
#     1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
#     2. Add your image to this Jupyter Notebook's directory, in the "images" folder
#     3. Change your image's name in the following code
#     4. Run the code and check if the algorithm is right (1 = cat, 0 = non-cat)!

# ## START CODE HERE ##
# my_image = "my_image.jpg" # change this to the name of your image file 
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# ## END CODE HERE ##

# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# my_image = my_image/255.
# my_predicted_image = predict(my_image, my_label_y, parameters)

# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
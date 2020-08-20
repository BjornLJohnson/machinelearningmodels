#!/usr/bin/env python
# coding: utf-8

import sys
import cv2
import numpy as np
from keras.models import model_from_json

def main():
    model_name = sys.argv[1]
    input_path = sys.argv[2]

    

    # Model reconstruction from JSON file
    with open('models/' + model_name + '_arch.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('models/' + model_name + '_weights.h5')

    # Get necessary input image shape
    shape = model.layers[0].input_shape[1:3]

    # Load input image from file and size appropriately
    input_data = load_data(input_path, shape)

    predictions = model.predict(input_data)

    print('predictions: ' + str(predictions))

def load_data(input_path, shape) :
    input_image = cv2.imread(input_path)   # reads an image in the BGR format

    height = input_image.shape[0]
    width = input_image.shape[1]

    # crops the image to be square
    if not height==width :
        min_dim = min(height, width)
        x_min = int((width-min_dim)/2)
        x_max = int((width+min_dim)/2)
        y_min = int((height-min_dim)/2)
        y_max = int((height+min_dim)/2)

        input_image = input_image[x_min:x_max, y_min:y_max]

    # resizes to the input size of the network
    resized_image = cv2.resize(input_image, shape, interpolation=cv2.INTER_AREA)

    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # converts to rgb and normalizes pixels to values 0-1
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    input_data = np.expand_dims(image_rgb, axis=0)/255

    return input_data

main()
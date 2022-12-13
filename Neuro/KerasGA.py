from keras import Sequential
from keras.layers import Dense
import numpy as np


def get_weights(model):
    # Initialize an empty list to store the weights
    weights = []

    # Loop over all layers in the model
    for layer in model.layers:
        # Get the weights for the current layer
        layer_weights = layer.get_weights()

        # Loop over the weights for the current layer
        for weight in layer_weights:
            # Flatten the weights and add them to the list
            weights += weight.flatten().tolist()

    # Return the list of flattened weights
    return weights


def set_weights(model, weights):
    # Initialize an index variable to keep track of the current position in the weights list
    index = 0

    # Loop over all layers in the model
    for layer in model.layers:
        # Get the number of weights for the current layer
        num_weights = layer.count_params()

        # Get the weights for the current layer as a flat list
        layer_weights = weights[index:index + num_weights]

        # Set the weights for the current layer
        layer.set_weights([layer_weights])

        # Update the index variable
        index += num_weights

model = Sequential()
model.add(Dense(4, input_dim = 1, activation = 'linear', name = 'layer_1'))
model.add(Dense(1, activation = 'linear', name = 'layer_2'))
model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['mse'])
w = get_weights(model)
print(w)
set_weights(model, w)
w = get_weights(model)
print(w)
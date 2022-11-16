"""
Basic neural network architecture that can be easily evolved.
"""
import random
import numpy as np
def sig(x):
 return 1/(1 + np.exp(-x))
class Network:
    def __init__(self, layers):
        self.inputs = []
        self.layers = layers
    def predict(self):
        first = 1
        for layer in self.layers:
            if first:
                start_layer = layer
                first = 0
            layer.from_inputs(start_layer.outputs)
            start_layer = layer
        outs = start_layer.outputs
        return outs


class Layer:
    def __init__(self, length, next_length, nconnections):  # TODO: make all these accept a rate
        self.next_length = next_length  # the length of the next layer
        self.length = length
        self.connections = nconnections
        self.weights = np.ones(next_length)
        self.outputs = np.ones(next_length)
        self.connections = np.asarray(random.choices(range(0, next_length), k=(next_length) * nconnections))
        self.connections = np.split(self.connections, nconnections)  # list of list (conections) len(list) will = len(weights)


    def from_inputs(self, inputs):  # TODO: make the best version of this possible
        for i, con in enumerate(self.connections):
            for n in con:
                val = inputs[i]
                self.outputs[n] += val  # im only 85% sure this is all correct
        self.outputs = sig(np.multiply(self.outputs, self.weights))
        return self.outputs


class EndLayer:
    def __init__(self, length, outputlen, nconnections):  # TODO: make all these accept a rate
        self.length = length
        self.weights = np.ones(outputlen)
        self.outputs = np.ones(outputlen)
        self.connections = np.asarray(random.choices(range(0, outputlen), k=(outputlen) * nconnections))
        self.connections = np.split(self.connections, nconnections)  # list of list (conections) len(list) will = len(weights)
    def from_inputs(self, inputs):
        out = np.multiply(inputs, self.weights)
        self.outputs = out
        return out
class StartLayer:
    def __init__(self, inputs, next):  # TODO: make all these accept a rate
        self.weights = np.ones(next)
        self.outputs = np.ones(next)
        self.connections = random.choices(range(0, next), k=next)
        self.inputs = inputs

    def from_inputs(self, n):
        for i, con in enumerate(self.connections):
            self.outputs[con] += self.inputs[i]  # im only 85% sure this is all correct
        self.outputs = np.multiply(self.outputs, self.weights)
        return self.outputs

x = [0, 1, 1, 0]
want = 1
net = Network(layers = [
    StartLayer(x, 4),
    Layer(40, 3, 4),
    Layer(3, 2, 3),
    EndLayer(1, 1, 1),
])

print(net.predict())

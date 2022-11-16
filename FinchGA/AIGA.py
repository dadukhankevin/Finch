import random
import numpy as np


def sig(x):
    return 1 / (1 + np.exp(-x))


class Network:
    def __init__(self, layers=[]):
        self.layers = layers

    def predict(self, input):
        connections = np.split(np.zeros(len(input)), len(input))
        for layer in self.layers:
            input, connections = layer.run(input, connections)
        return input, connections

    def get_complexity(self):
        r = 1
        for layer in self.layers:
            r *= layer.connections.size
        return r


class Dense:
    def __init__(self, input_length, output_length, synapses, activation=None):
        self.weights = np.ones(input_length)
        self.weighed = np.ones(input_length)
        self.activation = activation
        # for the next layer
        self.connections = np.asarray(random.choices(range(0, output_length), k=(input_length) * synapses))
        self.connections = self.connections.reshape((-1, synapses))

    def run(self, inputs, connections):
        for i, connection_group in enumerate(connections):
            for connection in connection_group:
                val = inputs[i - 1]
                self.weighed[int(connection)] += val
        if self.activation:
            self.weighed = sig(np.multiply(self.weighed, self.weights))
        else:
            self.weighed = np.multiply(self.weighed, self.weights)
        return self.weighed, self.connections


def get_all_weights(network):  # TODO: get a better way
    all = np.array([])
    for i in network.layers:
        all = np.append(all, i.weights)
        input(i.weights)
    return all


def get_from_weight(network, weights):
    for i, layer in enumerate(network.layers):
        layer.weights = weights[0: len(layer.weights)]
        weights = weights[len(layer.weights) - 1:-1]
    return network


class ImageNetwork:
    def __init__(self, images, classes):
        self.images = images
        self.size = images[0].size
        self.classes = classes
        self.net = None

    def get_network(self):
        net = Network(layers=[
            Dense(self.size, self.size * 2, 3, None),
            Dense(self.size * 2, self.size * 2, 4, sig),
            Dense(self.size * 2, self.classes.size, 3, sig),
            Dense(self.classes.size, self.classes.size, 3, activation=sig)
        ])
        self.net = net
        return net


def evaluate(network, x, y, amount):
    both = list(zip(x, y))
    choices = random.choices(both, k=amount)
    error = []
    for choice in choices:
        a = network.predict(choice[0])
        wanted = choice[1]
        error.append(np.square(np.subtract(a, wanted)).mean())
    return sum(error)/len(error)


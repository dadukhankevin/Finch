class Network:
    def __init__(self, synapses):
        self.synapses = synapses


class Neuron:
    def __init__(self, weight):
        self.weight = weight

    def receive(self, signal):
        pass


class Synapse:
    def __init__(self, neurons):
        self.neurons = neurons

import copy
import random
import numpy as np
import random as r
import matplotlib.pyplot as plt

def fitness(net):
    a = net.rec(1)
    if a > 9 and a < 10:
        return 1
    else:
        return 0


class SuperNetworkPopulation:
    def __init__(self, network, amount, fitnessfunc):
        self.network = network
        self.amount = amount
        self.fitnessfunc = fitnessfunc
        self.allnets = [copy.deepcopy(network)]
        for i in range(amount):
            network.mutate(.9)
            self.allnets.append(copy.deepcopy(network))

    def determine_fitness(self):
        fitnesses = []
        for net in self.allnets:
            fitnesses.append((fit([1,2], net), net))
        return list(sorted(fitnesses))
    def train(self, epochs):
        hist = []
        for i in range(epochs):
            all = self.determine_fitness()
            n = copy.deepcopy(all[-1][1])
            n.mutate(.2)
            hist.append(all[-1][0])
            self.allnets.append(n)
            self.allnets.pop(0)
        all = self.determine_fitness()

        return all[-1][1], hist

class Network:
    def __init__(self, subnets=[]):
        self.name = random.randint(-100, 100)
        self.weights = np.random.uniform(0, 1, len(subnets))
        self.subnets = subnets
        self.sum = 0
        self.age = 0
        self.mutate_age = 0
        self.found = 0

    def remake_weights(self):

        self.weights = np.random.uniform(0, 1, len(self.subnets))
    def rec(self, i, weight):
        self.sum += i * weight

        for i, n in enumerate(self.subnets):
            n.rec(self.sum, self.weights[i])

    def send(self):
        self.sum *= self.weight
        for i in self.subnets:
            i.rec(self.sum)

    def mutate(self, percent):
        global mutate_age

        for i in self.subnets:
            i.mutate(percent)
        if r.uniform(0, 1) < percent and len(self.weights) >= 1:
            self.weights[r.randint(0, len(self.weights)) - 1] = r.uniform(0, 1)

    def reset(self):

        self.sum = 0
        for i in self.subnets:
            i.reset()


class FirstLayer:
    def __init__(self, width, depth, output):
        self.outputs = [0]*output
        self.layers = np.ones((depth, width)).tolist()
        for i, n in enumerate(self.outputs):
            self.outputs[i] = copy.deepcopy(Network())
        for layer in self.layers:
            for i, nw in enumerate(layer):
                layer[i] = copy.deepcopy(Network())
        for l, layer in enumerate(self.layers):
            for i, nw in enumerate(layer):
                try:
                    layer[i].subnets = self.layers[l + 1]
                except IndexError:
                    layer[i].subnets = self.outputs
                layer[i].remake_weights()
        self.subnets = self.layers[0]
        self.weights = np.random.uniform(0, 1, width)

    def rec(self, lst):
        for i, n in enumerate(self.subnets):
            n.rec(lst[i], self.weights[i])


        outs = []
        for i in self.outputs:
            outs.append(i.sum)
        return outs

    def reset(self):
        for i in self.subnets:
            i.reset()

    def mutate(self, percent):
        global mutate_age

        for i in self.subnets:
            i.mutate(percent)
        if r.uniform(0, 1) < percent:
            self.weights[r.randint(0, len(self.weights))-1] += r.uniform(-.1, .1)
def fit(inputs, network):
    a = network.rec(inputs)
    if sum(a) > 100:
        return 100/sum(a)
    else:
        return sum(a)/100

net = FirstLayer(2, 3, 5)
a = net.rec([1,19])
print(a)

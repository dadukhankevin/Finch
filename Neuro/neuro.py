import copy
import numpy as np
import random as r
import matplotlib.pyplot as plt
max_age = 100000000000
mutate_age = 20000000000

best_net = None
best_fitness = 0
def modifybest(a):
    global best_fitness
    r = ""
    print(best_fitness)
    if a >best_fitness:
        best_fitness = a
        r = "changed"
    return r
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
            self.allnets.append(copy.deepcopy(network))

    def determine_fitness(self):
        pass


class Network:
    def __init__(self, subnets=[], name=""):
        self.name = name
        self.weights = np.random.uniform(0, 1, len(subnets))
        self.subnets = subnets
        self.sum = 0
        self.age = 0
        self.mutate_age = 0
        self.found = 0
    def rec(self, i, weight):
        self.sum += i
        self.age += 1
        if self.age < max_age:
            for i, n in enumerate(self.subnets):
                n.rec(self.sum * weight, self.weights[i])



    def send(self):
        self.sum *= self.weight
        for i in self.subnets:
            i.rec(self.sum)

    def mutate(self, percent):
        global mutate_age

        for i in self.subnets:
            i.mutate(percent)
        if r.uniform(0, 1) < percent and len(self.weights) >= 1:
            self.weights[r.randint(0, len(self.weights)) -1] = r.uniform(0, 1)
    def reset(self):
        self.sum = 0
        for i in self.subnets:
            i.reset()

class FirstLayer:
    def __init__(self, nets):
        self.subnets = nets
        self.weights = np.random.uniform(0, 1, len(nets))
        print(self.weights)
    def rec(self, lst):
        for i, n in enumerate(self.subnets):
            n.rec(lst[i], self.weights[i])
    def reset(self):
        for i in self.subnets:
            i.reset()
    def mutate(self, percent):
        global mutate_age

        for i in self.subnets:
            i.mutate(percent)
        if r.uniform(0, 1) < percent:
            self.weights[r.randint(0, len(self.weights))] += r.uniform(-.1, .1)
        




end = Network(name="end")

middle_1 = Network([end])
middle_2 = Network([end])
a = Network([middle_1])
b = Network([middle_1, middle_2])
c = Network([middle_2])

net = FirstLayer([a, b, c])

#best_net = copy.deepcopy(net)
net.rec([0,1,1])
print(end.sum)
net.reset()
net.mutate(.5)
net.rec([0,1,1])
print(end.sum)

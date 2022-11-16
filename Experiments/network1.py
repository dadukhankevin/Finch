import random

import numpy as np

from Finch.FinchGA.AIGA import *
from Finch.FinchGA.Layers import OverPoweredMutation, Parents, Kill, DetermineFitnessAndSort, GenerateData, KeepLength, Function, SortFitness
from Finch.FinchGA.GenePools import FloatPool
from Finch.FinchGA.Environments import SequentialEnvironment as SE
from Finch.FinchGA.generic import Fittness
import copy
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
def vectorize(i):
    ar = np.zeros(len(x)+1)
    ar[x.index(i)] = 1
    return ar


def close(x,y):
    if x>y:
        return y/x
    else:
        return x/y
def evaluate(x, y, model):
    past = []
    for i, ind in enumerate(x):
        past.append(close(model.predict([x[i]])[0][0], y[i]))
    return np.mean(past)
model = Network([
    Dense(1, 8, 10, activation=None),
    Dense(8, 1, 2, activation=sig),
    Dense(1, 1, 5, activation=None),  # must end with this
])
#b = np.array(get_all_weights(a))
#input(get_from_weight(a, b))
def fitness_layer2(weights):
    global model
    model = get_from_weight(copy.deepcopy(model), weights)
    return evaluate(x, y, model)


pool = FloatPool(0, 1, fitfunc=fitness_layer2)
env1 = SE(layers=[
    GenerateData(pool, 20, len(get_all_weights(model))),
    OverPoweredMutation(pool, 20, -1, fitness_function=fitness_layer2, range_rate=1, method="smart"),
    SortFitness(),
    Parents(pool, gene_size=1, family_size=3, percent=50, amount=2),
    SortFitness(),
    KeepLength(20),
])
env1.compile(100, fitness_layer2, every=10, stop_threshold=.98)
env1.simulate_env()

model = get_from_weight(model, env1.best_ind.genes)
probs, _ = model.predict(vectorize(4))
print("probs", probs)
probs, _ = model.predict(vectorize(2))
print("probs", probs)
env1.plot()





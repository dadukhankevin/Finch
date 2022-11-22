import random

import numpy as np

from Finch.FinchGA.AIGA import *
from Finch.FinchGA.Layers import OverPoweredMutation, Parents, Kill, DetermineFitnessAndSort, GenerateData, KeepLength, Function, SortFitness, Duplicate, FastMutateTop
from Finch.FinchGA.GenePools import FloatPool
from Finch.FinchGA.Environments import SequentialEnvironment as SE
from Finch.FinchGA.generic import Fittness
import copy
x = [1, 2, 3, 4, 5]
y = [1, 0, 1, 0, 1]
def vectorize(i):
    ar = np.zeros(len(x)+1)
    ar[x.index(i)] = 1
    return ar


def close(x,y):
    return abs(x-y)
def evaluateh(x, y, model):
    past = []
    for i, ind in enumerate(x):
        past.append(close(model.predict([x[i]])[0][0], y[i]))
    return np.min(past)
model = Network([
    Dense(1, 100, 2, activation=relu),
    Dense(100, 4, 2, activation=sig),
    Dense(4, 4, 2, activation=sig),
    Dense(4, 1, 2, activation=sig),
    Dense(1, 1, 1, activation=sig)  # must end with this
])
#b = np.array(get_all_weights(a))
#input(get_from_weight(a, b))
def fitness_layer2(weights):
    global model
    model = get_from_weight(model, copy.deepcopy(weights))
    a = model.predict([1])[0][0]
    b = model.predict([0])[0][0]
    return ((1-a)*b)


pool = FloatPool(0, 1, fitfunc=fitness_layer2)
env1 = SE(layers=[
    GenerateData(pool, 8, len(get_all_weights(model))),
    FastMutateTop(pool, amount=8, individual_mutation_amount=20),
    SortFitness(),
    Parents(pool, gene_size=1, family_size=1, percent=50, amount=2, every=10),
    SortFitness(),
    KeepLength(8),
])
env1.compile(400, fitness_layer2, every=100) #stop_threshold=.86)
env1.simulate_env()

model = get_from_weight(model, env1.best_ind.genes)
probs = model.predict([1])
print("probs", probs)
probs = model.predict([0])
print("probs", probs)
env1.plot()





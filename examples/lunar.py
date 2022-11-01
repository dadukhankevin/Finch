from Finch.FinchGA.GA import *
import matplotlib.pyplot as plt
from Finch.FinchGA.EvolveRates import *
import random
from textblob import TextBlob
import math
import numpy as np
#random.seed(10) #to reproduce
#((x*(y/z)+w**h)*t)/m=10

#(((a*a)/b*2)-c*d)-e+f=200
def mean(l):
    return np.mean(l)
def fitness(ind):
    a = ind[0]
    b = ind[1]
    c = ind[2]
    d = ind[3]
    e = ind[4]
    f = ind[5]
    g = ind[6]
    h = ind[7]
    i = ind[8]
    try:
        result = (((((a * a) / b * 2) - c * d) - (math.factorial(abs(e)) + f))/g*10)*102.4+g/(math.cos(h)+i)
        points = 200 / result
    except:
        return -10, ind
    if points >1:

        points = result/200
    return points, ind
# Similar to keras.Sequential
environment = Environment(classes=[
    #Kill(Rates(100,0).constant),
    Data(arrlen=100, length=9),

    DataMutate(percent_mutate=15, select_percent=100, delay=0),
    Duplicate(5),
    Parents(num_children=2, gene_size=1, delay=0, kill_parents=False, percent=10), #Mates them together randomly and kills the parents.
    KeepLength(amt=1000), #Keeps the len() of each generation at a max of 50

    Narrow(delay=100)
    ])
environment2 = Environment(classes=[
    Kill(Rates(100,0).constant),
    Data(arrlen=100, length=9),

    #DataMutate(percent_mutate=15, select_percent=100, delay=0),
    #Duplicate(5),
    #Parents(num_children=2, gene_size=1, delay=0, kill_parents=False, percent=10), #Mates them together randomly and kills the parents.
    #KeepLength(amt=500), #Keeps the len() of each generation at a max of 50

    #Narrow(delay=100)
    ])
vocab = list(range(-1000,1000))
vocab.pop(vocab.index(0))
Data , history = environment.compile(epochs=300, func=fitness, verbose=True, every=20, lettrs=vocab)
Data , history2 = environment2.compile(epochs=300, func=fitness, verbose=True, every=20, lettrs=vocab)

print(Data[-1])
print(Data[0])
plt.plot(history, history2)
plt.show()

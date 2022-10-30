from FinchGA.finchstring import *
from FinchGA.EvolveRates import *
from FinchGA.generic import *
import matplotlib.pyplot as plt
import random
from textblob import TextBlob
#random.seed(10) #to reproduce

def fitness(generation):
    # A basic function that insensitive generations with the most occurrences of the letter a.
    #input("".join(generation))
    ret = generation
    points = 0
    words = "".join(generation).split(" ")
    n = 0
    for i in words:
        p = TextBlob(i).polarity
        if p > .02:
            points += p + n
            n += .1

    return points, ret

killrate = Rates(.50, .005)
PercentageM = Rates(100, .0000001) #decreases from 100
Small = Rates(10, .00000001)
# Similar to keras.Sequential
environment = Environment(classes=[
    Data(arrlen=200, length = 10),

    DataMutate(small_function=Small.slow, big_function=PercentageM.slow, delay=0),
    #ParentOpposites(amount=3,num_children=2, gene_size=4, delay=50, kill_parents=True, percent=100), #Mates them together randomly and kills the parents.
    Parents(num_children=1, gene_size=1, delay=0, kill_parents=False, percent=100), #Mates them together randomly and kills the parents.
    KeepLength(amt=50), #Keeps the len() of each generation at a max of 50
    ])

_, history = environment.compile(epochs=500, func=fitness, verbose=True, every=10, lettrs=list("qwert ui opbas dfghj klx cvbnm"))
plt.plot(history)
plt.show()
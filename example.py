from FinchGA.finchstring import *
from FinchGA.EvolveRates import *
from FinchGA.generic import *
import matplotlib.pyplot as plt
import random
from textblob import TextBlob
random.seed(10)
max = None


def fit1(generation):  # A basic function that insensitive generations with the most occurrences of the letter a.
    points = generation.count("h")
    points += generation.count("e")
    points += generation.count("l")
    points += generation.count("o")
    return points, generation
def fit2(generation):
    points = generation.count("he")
    points += generation.count("el")
    points += generation.count("ll")
    points += generation.count("lo ")
    points += generation.count(" ")
    return points, generation



Fit = Fittness([fit1, fit2,], [90, 10])

words = list("asdfghjkl")
killrate = Rates(.50, .005)
PercentageM = Rates(100, .0000001)
Small = Rates(2, .00000001)
# Similar to keras.Sequential
environment = Environment(classes=[
    Kill(killrate.slow) # Kills the bottom 90% of the population
    ,
    String(10, 105) # Creates a new generation of
    ,
    Duplicate(6)
    ,
    KeepLength(50)
    ,
    Narrow(delay=0)
    ,
    StringMutate(small_function=Small.slow, big_function=PercentageM.slow, delay=0)
    # mutates 99 percent of the generation and within those it mutates it changes 1% f1 the letters.
    ,
    TopN(number=30, delay=200)
    ,
    Parents(num_children=2, gene_size=4, delay=100, kill_parents=True, percent=100),

])
data, history = environment.compile(400, func=Fit.func, verbose=True, every=10, lettrs=list("qwertyui opasdfghjklxcvbnm"))
plt.plot(history)
plt.show()

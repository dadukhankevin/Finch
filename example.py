from FinchGA.finchstring import *
from FinchGA.EvolveRates import *
from FinchGA.generic import *
import matplotlib.pyplot as plt
import random
from textblob import TextBlob
random.seed(10)
max = None


def fit1(generation):
    # A basic function that insensitive generations with the most occurrences of the letter a.
    #i#nput("".join(generation))
    points = "".join(generation).count("a")

    return points, generation




Fit = Fittness([fit1], [90])

words = list("ab")
killrate = Rates(.50, .005)
PercentageM = Rates(100, .0000001)
Small = Rates(2, .00000001)
# Similar to keras.Sequential
environment = Environment(classes=[
    Data(arrlen=10, length=20),
    DataMutate(small_function=Small.slow, big_function=PercentageM.slow, delay=0),
    Parents(num_children=2, gene_size=4, delay=100, kill_parents=True, percent=100),
    KeepLength(50),
])
data, history = environment.compile(epochs=1000, func=Fit.func, verbose=True, every=10, lettrs=list("ab"))
plt.plot(history)
plt.show()

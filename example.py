from FinchGA.finchstring import *
from FinchGA.EvolveRates import *
import matplotlib.pyplot as plt
max = None

def fit(generation):  # A basic function that insensitive generations with the most occurrences of the letter a.
    points = generation.count("a")
    return points, generation

words = list("asdfghjkl")
killrate = Rates(.50, .005)
PercentageM = Rates(100, .0000001)
Small = Rates(2, .00000001)
print(Small.constant())
environment = Environment()

# Similar to keras.Sequential
environment.add(
    Kill(killrate.slow)  # Kills the bottom 90% of the population
)

environment.add(
    String(30, 100)  # Creates a new generation of
)
environment.add(
    Duplicate(3)
)
environment.add(
    KeepLength(10)
)
environment.add(
    Narrow()
)
environment.add(
    StringMutate(small_function=Small.slow,big_function=PercentageM.slow)
    # mutates 99 percent of the generation and within those it mutates it changes 1% of the letters.
)

environment.add(
    Parent(2, gene_size=8)
)

data, history = environment.compile(100, func=fit, verbose=True, every=3, lettrs=["a", "b", "c"])
plt.plot(history)
plt.show()


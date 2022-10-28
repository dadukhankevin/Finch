from FinchGA.finchstring import *
from FinchGA.EvolveRates import *

max = None

def fit(generation):  # A basic function that insensitive generations with the most occurrences of the letter a.
    generation = [int(i) for i in generation]
    points = sum(generation)
    generation = [str(i) for i in generation]
    return points, generation

words = ['1','2','3','4']
killrate = Rates(.50, .005)
PercentageM = Rates(100, .0000001)
Small = Rates(2, .00000001)
print(Small.constant())
environment = Environment()  # Similar to keras.Sequential
environment.add(
    Kill(killrate.slow)  # Kills the bottom 90% of the population
)

environment.add(
    String(30, 1000, fit, letters=words)  # Creates a new generation of
)
environment.add(
    Duplicate(3)
)
environment.add(
    KeepLength(10)
)
environment.add(
    StringMutate(small_function=Small.slow,big_function=PercentageM.slow, letters=words)
    # mutates 99 percent of the generation and within those it mutates it changes 1% of the letters.
)

environment.add(
    Parent(10, gene_size=3)
)

data, history = environment.compile(1000, func=fit, verbose=True)
plt.plot(history)
plt.show()


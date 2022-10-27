from FinchGA.finchstring import *
from FinchGA.EvolveRates import *

max = None

def fit(generation):  # A basic function that insensitive generations with the most occurrences of the letter a.
    points = generation.count("z")
    return points, generation


killrate = Rates(.50, .005)
PercentageM = Rates(100, .0000001)
Small = Rates(2, .00000001)
print(Small.constant())
environment = Environment()  # Similar to keras.Sequential
environment.add(
    Kill(killrate.slow)  # Kills the bottom 90% of the population
)

environment.add(
    String(100, 100, fit, letters="qwertyuiopasdfghjklzxcvbnm")  # Creates a new generation of
)
environment.add(
    Duplicate(20)
)
environment.add(
    KeepLength(100)
)
environment.add(
    StringMutate(small_function=Small.slow,big_function=PercentageM.slow, letters="qwertyuiopasdfghjklzxcvbnm")
    # mutates 99 percent of the generation and within those it mutates it changes 1% of the letters.
)

environment.add(
    Parent(10, gene_size=1)
)

data, history = environment.compile(1000, func=fit, verbose=True)
plt.plot(history)
plt.show()
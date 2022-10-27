from Finch.FinchGA.finchstring import *

max = 100
def fit(generation):  # A basic function that insensitive generations with the most occurrences of the letter a.
    points = generation.count("a")
    if max:
        points = (points/max)*100
    return points, generation


environment = Environment()  # Similar to keras.Sequential
environment.add(
    Kill(.2)  # Kills the bottom 90% of the population
)
environment.add(
    String(100, 100, fit, letters="qwertyuiopasdfghjklzxcvbnm")  # Creates a new generation of
)
environment.add(
    StringMutate(percentage=99, small_percent=1, letters="qwertyuiopasdfghjklzxcvbnm")  #mutates 99 percent of the generation and within those it mutates it changes 1% of the letters.
)

environment.add(
    Parent(5, gene_size=1)
)
data, history = environment.compile(1000, func=fit, verbose=True)
plt.plot(history)
plt.show()

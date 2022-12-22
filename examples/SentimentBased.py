"""
This example generates real words because the fitness function incentives text that has higher positive
or negative sentiment at all (and so is made up of real words). Of course this only has many weaknesses since it is built
on sentiment and not "how real is this sentence".
"""
from textblob import TextBlob
from Finch.FinchGenetics import *
import matplotlib.pyplot as plt


def fitness(sentence):
    real = "".join(sentence).split(" ")
    points = 0
    for word in real:
        points += abs(TextBlob(word).polarity) * (len(word) / 2)  # length of the word is also given priority
    return points


querty = np.asarray(list("qwertyuiopasdfghjklzxcvbnm     "))
pool = GenePool(querty, fitness, mx=100, mn=0, shape=(50,))

env = Environment([
    Generate(pool, 1000),
    Parents(pool, delay=1, gene_size=5, family_size=5, percent=.04, method="random"),
    OverPoweredMutation(pool, iters=5, index=-1, method="random", fitness_function=fitness),
    SortFitness(),
    KeepLength(990),
])

data, hist = env.evolve(100, verbose=10)
print("Look for real words: ")
print("".join(data[-1].genes))

plt.plot(hist)
plt.show()

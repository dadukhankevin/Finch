from Finch.FinchGenetics import *
from random import randint


def ensure_length(s, length):
    if len(s) < length:
        s = s.ljust(length, "|")
    return s


def gen():
    return np.array(list(ensure_length(str(randint(0, 1000)), 20)))

def mutate(data):
    return data

def fit(string):
    return len("".join(string).replace("|", ""))

def parent(string):
    return string
pool = CustomGenePool(gen_function=gen, fitness_func=fit, shape=(20,))

env = Environment([
    Generate(pool, 6),
    Parents(pool, delay=1, gene_size=1, family_size=2, percent=.5, method="best", amount=10, every=100),
    Function(parent),
    Function(mutate),
    SortFitness(),
    KeepLength(3),
])

data, history = env.evolve(50, verbose=1)
genes = data[-1].genes
print("Look for real words: ")
print("".join(data[-1].genes))

plt.plot(history)
plt.show()

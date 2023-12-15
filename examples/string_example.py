from Finch3.environments import Sequential
from Finch3.genepools import IntPool
from Finch3.layers import *
from Finch3.tools import individualselectors as selectors
import random as r
import numpy.random as nr

r.seed(100)
nr.seed(100)

def translate(array):
    return "".join([chr(i) for i in array])


def fit(individual):
    translated_string = translate(individual.genes)
    points = translated_string.count("a")
    return points


# Config section
length = 100
pool_minimum = ord('a')
pool_maximum = ord('z')
population_size = 101
mutate_amount = 70
gene_selection = 2
parent_count = 20
selection_rank_factor = 40
children_count = 4
max_population = 100
evolution_steps = 2000

selector = selectors.RankBasedSelection(selection_rank_factor, amount_to_select=parent_count)

# Creating the FloatPool
pool = IntPool(length=length, minimum=pool_minimum, maximum=pool_maximum)

# Creating the Sequential environment
environment = Sequential(layers=[
    Populate(pool, population=population_size),
    Mutate(individual_selection=mutate_amount, gene_selection=gene_selection),
    ParentNPoint(selector.select, children=children_count, refit=True, points=4),
    SortByFitness(),
    CapPopulation(max_population=max_population),
])


if __name__ == "__main__":
    # Compiling the environment with the fitness function
    environment.compile(fitness_function=fit, verbose_every=100)
    environment.evolve(evolution_steps)

    # Printing the best individual
    print("Here is the best individual:\n", translate(environment.best_ever.genes))

    # Plotting the environment
    environment.plot()

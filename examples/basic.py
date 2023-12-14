from Finch3.environments import Sequential
from Finch3.genepools import FloatPool
from Finch3.layers import *


def fit(individual):
    return sum(individual.genes)  # better fitness the higher the


# Config section
length = 100
pool_minimum = 0
pool_maximum = 10
population_size = 1001
individual_selection = 2
gene_selection = 10
parent_count = 2
children_count = 2
max_population = 1000
evolution_steps = 1000

# Creating the FloatPool
pool = FloatPool(length=length, minimum=pool_minimum, maximum=pool_maximum)

# Creating the Sequential environment
environment = Sequential(layers=[
    Populate(pool, population=population_size),
    Mutate(individual_selection=individual_selection, gene_selection=gene_selection),
    ParentSimple(parent_count, children=children_count),
    SortByFitness(),
    CapPopulation(max_population=max_population),
])

# Compiling the environment with the fitness function
environment.compile(fitness_function=fit)
environment.evolve(evolution_steps)

# Printing the best individual
print("Here is the best individual:\n", environment.best_ever.genes)

# Plotting the environment
environment.plot()

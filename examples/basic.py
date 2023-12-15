from Finch3.environments import Sequential
from Finch3.genepools import FloatPool
from Finch3.layers import *


def fit(individual):
    return sum(individual.genes)  # -(individual.genes[-1]*4)  <-- You can add something like that to make it
    # interesting


# Config section
length = 100
pool_minimum = 0
pool_maximum = 10
population_size = 100
amount_to_mutate = 10
gene_selection = 5
parent_count = 20
children_count = 2
max_population = 99
evolution_steps = 1000
min_mutation = -2
max_mutation = 2

# Creating the FloatPool
pool = FloatPool(length=length, minimum=pool_minimum, maximum=pool_maximum)

# Creating the Sequential environment
environment = Sequential(layers=[
    Populate(pool, population=population_size),
    FloatMutateRange(individual_selection=amount_to_mutate, gene_selection=gene_selection,
                     min_mutation=min_mutation, max_mutation=max_mutation, keep_within_genepool_bounds=True),
    ParentSimple(parent_count, children=children_count),
    SortByFitness(),
    CapPopulation(max_population=max_population),
])


if __name__ == "__main__":
    # Compiling the environment with the fitness function
    environment.compile(fitness_function=fit)
    environment.evolve(evolution_steps)

    # Printing the best individual
    print("Here is the best individual:\n", environment.best_ever.genes)

    # Plotting the environment
    environment.plot()
